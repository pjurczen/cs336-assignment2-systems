import cProfile
import mmap
import os
import pstats
import time
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from functools import partial
from multiprocessing import Pool
from typing import BinaryIO, Optional

import regex as re
from line_profiler import profile

PAT = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def split_text_to_chunks(text: bytes, sorted_special_tokens: list[bytes]) -> list[bytes]:
    """
    Splits text by special tokens (any match counts) and returns chunk with trailing special token that caused that split.
    Special tokens must be sorted from longest to shortest in the input param.
    """
    if not sorted_special_tokens:
        return [text]
    special_tokens = [re.escape(x) for x in sorted_special_tokens]
    split_pattern = re.compile(b'(' + b'|'.join(special_tokens) + b')')
    chunks = split_pattern.split(text)
    return [c for c in chunks if c]


def pretokenize_text(text: bytes) -> list[bytes]:
    """
    Pretokenize a text string to separate pretokens.
    """
    return PAT.findall(text)


def _find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = mmap.PAGESIZE  # Read ahead by 16k bytes at a time - Apple silicon memory page size

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _calculate_pretokens(text: bytes) -> tuple[tuple[bytes, ...], ...]:
    return map(_build_tuple_bytes, PAT.findall(text))


@lru_cache(maxsize=None)
def _build_tuple_bytes(word_bytes: bytes) -> tuple[bytes, ...]:
    return tuple(word_bytes[i:i + 1] for i in range(len(word_bytes)))


@profile
def _pretokenize_chunk(start_idx: int, end_idx: int, path: str, split_special_token: bytes) -> Counter[
    tuple[bytes, ...]]:
    frequencies: Counter[tuple[bytes, ...]] = Counter({})
    split_pattern = re.compile(re.escape(split_special_token))
    page_size: int = mmap.ALLOCATIONGRANULARITY  # Apple silicon memory allocation size
    diff_to_full_page_size: int = start_idx % page_size
    full_page_start_in_file: int = start_idx - diff_to_full_page_size
    start_in_mm: int = diff_to_full_page_size
    pages_length: int = end_idx - full_page_start_in_file
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), pages_length, access=mmap.ACCESS_READ, offset=full_page_start_in_file) as mm:
            for match in split_pattern.finditer(mm, start_in_mm):
                chunk = mm[start_in_mm: match.start()]
                pretokens = _calculate_pretokens(chunk)
                frequencies.update(pretokens)
                start_in_mm = match.end()
            tail = mm[start_in_mm:]
            if tail:
                pretokens = _calculate_pretokens(tail)
                frequencies.update(pretokens)
    return frequencies


def _merge_frequencies(dict1: Counter[tuple[bytes, ...]], dict2: Counter[tuple[bytes, ...]]) -> Counter[
    tuple[bytes, ...]]:
    dict1.update(dict2)
    return dict1


def pretokenize(path: str, num_processes: Optional[int] = None, split_special_token: bytes = b"<|endoftext|>") -> \
        Counter[tuple[bytes, ...]]:
    num_processes = num_processes or os.cpu_count()
    frequencies: Counter[tuple[bytes, ...]] = Counter({})
    with open(path, "rb") as f:
        t0 = time.perf_counter()
        boundaries: list[int] = _find_chunk_boundaries(f, num_processes, split_special_token)
        t1 = time.perf_counter()
        print(f"boundaries: {t1 - t0:.2f}s")
        chunk_indices: Iterator[tuple[int, int]] = zip(boundaries[:-1], boundaries[1:])
        _pretokenize_partial = partial(_pretokenize_chunk, path=path, split_special_token=split_special_token)
        with Pool(num_processes) as p:
            chunk_frequencies_list: list[Counter[tuple[bytes, ...]]] = p.starmap(_pretokenize_partial, chunk_indices)
        t2 = time.perf_counter()
        print(f"pretokenize (parallel): {t2 - t1:.2f}s")
        for chunk_frequencies in chunk_frequencies_list:
            frequencies = _merge_frequencies(frequencies, chunk_frequencies)
        t3 = time.perf_counter()
        print(f"merge: {t3 - t2:.2f}s")
    return frequencies


@profile
def merge_pretoken(pair: tuple[bytes, bytes], pretoken: tuple[bytes, ...]) -> tuple[bytes, ...]:
    new_pretoken: tuple[bytes, ...] = ()
    i: int = 0
    pretoken_len: int = len(pretoken)
    while i < pretoken_len:
        if i < pretoken_len - 1 and (pretoken[i], pretoken[i + 1]) == pair:
            new_token = pretoken[i] + pretoken[i + 1]
            new_pretoken += (new_token,)
            i += 2
        else:
            new_pretoken += (pretoken[i],)
            i += 1
    return new_pretoken


def get_byte_pairs(pretoken: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
    pairs: set[tuple[bytes, bytes]] = set()
    for byte_pair in zip(pretoken, pretoken[1:]):
        pairs.add(byte_pair)
    return pairs


def profile_pretokenization(path: str):
    boundaries = _find_chunk_boundaries(open(path, "rb"), 4, b"<|endoftext|>")
    start, end = boundaries[0], boundaries[1]

    with cProfile.Profile() as pr:
        _pretokenize_chunk(start, end, path, b"<|endoftext|>")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(40)


if __name__ == "__main__":
    # profile_pretokenization("../data/TinyStoriesV2-GPT4-valid.txt")
    result = pretokenize("../data/TinyStoriesV2-GPT4-train.txt")
    sorted_items = sorted(result.items(), key=lambda kv: kv[1], reverse=True)
    print("=== TOP 10 ===\n")
    for k, c in sorted_items[:10]:
        print(f"{c:>8}  {k!r}\n")
    print("\n=== BOTTOM 10 ===\n")
    for k, c in sorted_items[-10:]:
        print(f"{c:>8}  {k!r}\n")
    print(f"\nTotal unique pre-tokens: {len(result)}\n")
    print(f"Total token count: {sum(result.values())}\n")
