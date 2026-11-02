import argparse
import cProfile
import json
import os
import pstats
import resource
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from functools import lru_cache

import numpy as np

from cs336_basics.bpe_state import BpeState
from cs336_basics.common import gpt2_bytes_to_unicode
from cs336_basics.peek_memory_sampler import PeekMemorySampler
from cs336_basics.pretokenization import pretokenize, split_text_to_chunks, pretokenize_text, _build_tuple_bytes, \
    merge_pretoken, get_byte_pairs, PAT

BYTES_COUNT = 256  # this is really an overkill since in UTF-8 192, 193 and >=245 bytes are not used, but we learn those if at inference time invalid input was somehow passed

VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.txt"
CONFIG_FILE = "tokenizer_config.json"


class BpeTokenizer:
    vocab: dict[int, bytes]
    id_vocab: dict[bytes, int]
    merges: list[tuple[bytes, bytes]]
    merges_rank: dict[tuple[bytes, bytes], int]
    special_tokens: list[bytes]

    def __init__(self):
        self.vocab = {}
        self.id_vocab = {}
        self.merges = []
        self.special_tokens = []

    @classmethod
    def from_vocab_merges_path(
            cls,
            vocab_path: str | os.PathLike,
            merges_path: str | os.PathLike,
            special_tokens: list[str] | None = None,
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls.from_resources(vocab, merges, special_tokens)

    @classmethod
    def from_resources(cls, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> "BpeTokenizer":
        bpe_tokenizer: BpeTokenizer = BpeTokenizer()
        bpe_tokenizer.merges = merges
        bpe_tokenizer.vocab = vocab
        bpe_tokenizer.special_tokens = [x.encode('utf-8') for x in special_tokens] if special_tokens else []
        bpe_tokenizer.special_tokens.sort(key=lambda x: len(x), reverse=True)
        bpe_tokenizer.id_vocab = {id: byte for byte, id in vocab.items()}
        bpe_tokenizer.merges_rank = {pair: i for i, pair in enumerate(merges)}
        return bpe_tokenizer

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> None:  # (vocab, merges)
        merges: list[tuple[bytes, bytes]] = []
        vocab: dict[int, bytes] = {x: special_tokens[x].encode('utf-8') for x in range(len(special_tokens))}
        self.special_tokens = [x.encode('utf-8') for x in special_tokens]
        self.special_tokens.sort(key=lambda x: len(x), reverse=True)
        special_tokens_count: int = len(special_tokens)
        for x in range(special_tokens_count, BYTES_COUNT + special_tokens_count):
            vocab[x] = bytes([x - special_tokens_count])
        initial_vocab_size = len(vocab)
        num_merges: int = vocab_size - initial_vocab_size
        split_special_token: bytes = self.special_tokens[0]
        pretoken_vocab: Counter[tuple[bytes, ...]] = pretokenize(input_path, split_special_token=split_special_token)

        t0 = time.perf_counter()
        bpe_state: BpeState = BpeState(pretoken_vocab)
        for i in range(num_merges):
            pair: tuple[bytes, bytes] = bpe_state.get_highest_count_pair()
            merges.append(pair)
            new_index: int = special_tokens_count + BYTES_COUNT + i
            merged_pair: bytes = pair[0] + pair[1]
            vocab[new_index] = merged_pair
            bpe_state.merge(pair)
        t1 = time.perf_counter()
        print(f"training: {t1 - t0:.2f}s")
        self.vocab = vocab
        self.id_vocab = {id: byte for byte, id in vocab.items()}
        self.merges = merges
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        encoded_ids: list[int] = []
        encoded_text: bytes = text.encode('utf-8')
        chunks: list[bytes] = split_text_to_chunks(encoded_text, self.special_tokens)
        for chunk in chunks:
            if chunk in self.special_tokens:
                encoded_ids.append(self.id_vocab[chunk])
            else:
                pretokens: list[bytes] = pretokenize_text(chunk)
                for pretoken in pretokens:
                    encoded_ids += self._encode_pretoken(pretoken)
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Streaming encode. The buffer holds bytes that are not yet safe to emit.
        #
        # 1. Pull the next chunk from the iterable and append it to the buffer.
        #
        # 2. Split the buffer on special tokens. split() returns them interleaved:
        #       b'a<|eot|>b'  ->  [b'a', b'<|eot|>', b'b']
        #
        # 3. Every chunk EXCEPT the last is settled — a matched special token bounds it
        #    on the right, so nothing arriving later can change it. Process each one
        #    completely and forget it:
        #       a) it is a special token  ->  emit its encoded id
        #       b) otherwise              ->  pretokenize and emit ALL encoded pretokens
        #
        # 4. The last chunk is open-ended: more input could still extend it. Three cases,
        #    tested in this order (a chunk can be both a whole token and a prefix of a
        #    longer one, e.g. <|eot|> vs <|eot|><|eot|>):
        #       a) it is a proper prefix of some special token
        #             ->  emit nothing, leave it in the buffer, go to 1
        #       b) it equals a special token
        #             ->  emit its id, buffer becomes empty
        #       c) neither
        #             ->  pretokenize; emit all but the final pretoken, and leave that
        #                 final pretoken in the buffer (it may still grow)
        #
        # 5. When the iterable is exhausted, flush: split the buffer and emit every chunk
        #    in full — special tokens as ids, text chunks fully pretokenized. Nothing is
        #    withheld, since no more input is coming.
        buffer: bytes = b''
        for input_chunk in iterable:
            # 1.
            buffer += input_chunk.encode('utf-8')
            # 2.
            split_chunks: list[bytes] = split_text_to_chunks(buffer, self.special_tokens)
            chunk_idx: int = 0
            while chunk_idx < len(split_chunks):
                last_chunk: bool = chunk_idx == len(split_chunks) - 1
                current_chunk: bytes = split_chunks[chunk_idx]
                if not last_chunk:
                    # 3. a)
                    special_token_found: bool = False
                    for special_token in self.special_tokens:
                        if current_chunk == special_token:
                            special_token_found = True
                            break
                    if special_token_found:
                        yield self.id_vocab[current_chunk]
                        buffer = buffer[len(current_chunk):]
                        chunk_idx += 1
                    # 3. b)
                    else:
                        pretokens: list[bytes] = PAT.findall(current_chunk)
                        for pretoken in pretokens:
                            encoded_ids: list[int] = self._encode_pretoken(pretoken)
                            for idx in encoded_ids:
                                yield idx
                        buffer = buffer[len(current_chunk):]
                        chunk_idx += 1
                else:
                    special_token_start: bool = False
                    special_token_found: bool = False
                    for special_token in self.special_tokens:
                        if special_token.startswith(current_chunk):
                            if special_token != current_chunk:  # 4. a)
                                special_token_start = True
                                # We can break here because we sorted the special tokens from longest to shortest guaranteeing that first match is enough
                                # If we didn't we would land in the elif and wrongly identify as a special shorter token before reading enough buffer
                                break
                            elif special_token == current_chunk:  # 4. b)
                                special_token_found = True
                                break
                    # 4. a)
                    if special_token_start:
                        break
                    # 4. b)
                    elif special_token_found:
                        yield self.id_vocab[current_chunk]
                        buffer = buffer[len(current_chunk):]
                        chunk_idx += 1
                    # 4. c)
                    else:
                        pretokens: list[bytes] = PAT.findall(current_chunk)
                        for pretoken in pretokens[:-1]:
                            encoded_ids: list[int] = self._encode_pretoken(pretoken)
                            for idx in encoded_ids:
                                yield idx
                            buffer = buffer[len(pretoken):]
                        chunk_idx += 1
        # 5.
        split_chunks: list[bytes] = split_text_to_chunks(buffer, self.special_tokens)
        chunk_idx: int = 0
        while chunk_idx < len(split_chunks):
            current_chunk: bytes = split_chunks[chunk_idx]
            special_token_found: bool = False
            for special_token in self.special_tokens:
                if current_chunk == special_token:
                    special_token_found = True
                    break
            if special_token_found:
                yield self.id_vocab[current_chunk]
                buffer = buffer[len(current_chunk):]
                chunk_idx += 1
            else:
                pretokens: list[bytes] = PAT.findall(current_chunk)
                for pretoken in pretokens:
                    encoded_ids: list[int] = self._encode_pretoken(pretoken)
                    for idx in encoded_ids:
                        yield idx
                buffer = buffer[len(current_chunk):]
                chunk_idx += 1

    @lru_cache(maxsize=None)
    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        tuple_bytes: tuple[bytes, ...] = _build_tuple_bytes(pretoken)
        while True:
            byte_pairs: set[tuple[bytes, bytes]] = get_byte_pairs(tuple_bytes)
            candidates: list[tuple[bytes, bytes]] = [x for x in byte_pairs if x in self.merges_rank]
            if not candidates:
                break
            pair: tuple[bytes, bytes] = min(candidates, key=self.merges_rank.get)
            tuple_bytes = merge_pretoken(pair, tuple_bytes)
        return [self.id_vocab[x] for x in tuple_bytes]

    def decode(self, ids: list[int]) -> str:
        bytes_list: list[bytes] = list(map(self.vocab.get, ids))
        text = b"".join(bytes_list).decode('utf-8', errors='replace')
        return text

    def save(self, vocab_path: str, merges_path: str) -> None:
        encoded_vocab: dict[str, int] = {
            bytes_to_gpt2_unicode(bytes_item): vocab_index
            for bytes_item, vocab_index in self.id_vocab.items()
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(encoded_vocab, f, ensure_ascii=False, indent=4)
        with open(merges_path, "w", encoding="utf-8") as f:
            for merge in self.merges:
                f.write(f"{bytes_to_gpt2_unicode(merge[0])} {bytes_to_gpt2_unicode(merge[1])}\n")

    def save_dir(self, tokenizer_dir: str | os.PathLike, input_path: str | None = None) -> None:
        os.makedirs(tokenizer_dir, exist_ok=True)
        self.save(os.path.join(tokenizer_dir, VOCAB_FILE), os.path.join(tokenizer_dir, MERGES_FILE))
        config = {
            "vocab_size": len(self.vocab),
            "special_tokens": [x.decode("utf-8") for x in self.special_tokens],
            "input_path": input_path,
        }
        with open(os.path.join(tokenizer_dir, CONFIG_FILE), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_dir(cls, tokenizer_dir: str | os.PathLike) -> "BpeTokenizer":
        config = read_tokenizer_config(tokenizer_dir)
        return cls.from_vocab_merges_path(
            os.path.join(tokenizer_dir, VOCAB_FILE),
            os.path.join(tokenizer_dir, MERGES_FILE),
            config["special_tokens"],
        )

    def max_length_token(self) -> tuple[int, bytes]:
        return max(self.vocab.items(), key=lambda x: len(x[1]))


def read_tokenizer_config(tokenizer_dir: str | os.PathLike) -> dict:
    with open(os.path.join(tokenizer_dir, CONFIG_FILE)) as f:
        return json.load(f)


def bytes_to_gpt2_unicode(bytes_token: bytes) -> str:
    return ''.join([gpt2_bytes_to_unicode()[token] for token in bytes_token])


def train_command(args: argparse.Namespace) -> None:
    bpe_tokenizer = BpeTokenizer()
    if args.profile:
        with cProfile.Profile() as pr:
            with PeekMemorySampler() as mem:
                bpe_tokenizer.train(args.input, args.vocab_size, args.special_tokens)
                print(f"peak: {mem.peak_mib():.0f} MiB")
            print(f"Max worker memory used {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}")
        pstats.Stats(pr).sort_stats("cumulative").print_stats(40)
    else:
        bpe_tokenizer.train(args.input, args.vocab_size, args.special_tokens)

    bpe_tokenizer.save_dir(args.out_dir, input_path=args.input)
    print(f"Saved tokenizer to {args.out_dir}")
    print(f"Longest token: {bpe_tokenizer.max_length_token()}")


def encode_command(args: argparse.Namespace) -> None:
    bpe_tokenizer = BpeTokenizer.from_dir(args.tokenizer_dir)
    # uint16 halves the file size, but only fits ids below 65,536
    dtype = np.uint16 if len(bpe_tokenizer.vocab) <= np.iinfo(np.uint16).max + 1 else np.uint32
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.input) as f:
        ids = np.fromiter(bpe_tokenizer.encode_iterable(f), dtype=dtype, count=-1)
    np.save(args.output, ids)
    print(f"Saved {len(ids)} tokens ({ids.dtype}) to {args.output}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = arg_parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument("--input", type=str, required=True)
    train_parser.add_argument("--vocab_size", type=int, required=True)
    train_parser.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    train_parser.add_argument("--out_dir", type=str, required=True)
    train_parser.add_argument("--profile", action="store_true")
    train_parser.set_defaults(func=train_command)

    encode_parser = subparsers.add_parser("encode", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    encode_parser.add_argument("--tokenizer_dir", type=str, required=True)
    encode_parser.add_argument("--input", type=str, required=True)
    encode_parser.add_argument("--output", type=str, required=True)
    encode_parser.set_defaults(func=encode_command)

    args = arg_parser.parse_args()
    args.func(args)
