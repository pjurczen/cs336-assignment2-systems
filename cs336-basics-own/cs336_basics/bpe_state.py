from collections import Counter, defaultdict

from line_profiler import profile

from cs336_basics.pretoken_counter import PretokenCounter
from cs336_basics.pretokenization import merge_pretoken


class BpeState:
    pretoken_vocab: dict[tuple[bytes, ...], int]
    pretoken_counter: PretokenCounter
    pairs_to_pretokens_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]

    def __init__(self, pretoken_vocab: Counter[tuple[bytes, ...]]):
        self.pretoken_vocab = pretoken_vocab
        self.pretoken_counter = PretokenCounter.from_counter(self._count_adjacent_pairs(pretoken_vocab))
        self.pairs_to_pretokens_index = self._calculate_pairs_to_pretokens_index(pretoken_vocab)

    def _count_adjacent_pairs(self, pretoken_vocab: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
        counts: dict[tuple[bytes, bytes], int] = {}
        for pretoken, count in pretoken_vocab.items():
            for byte_pair in zip(pretoken, pretoken[1:]):
                if byte_pair not in counts:
                    counts[byte_pair] = count
                else:
                    counts[byte_pair] += count
        return counts

    def _calculate_pairs_to_pretokens_index(self, pretoken_vocab: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
        pairs_to_pretokens_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
        for pretoken in pretoken_vocab.keys():
            i: int = 0
            pretoken_len: int = len(pretoken)
            while i < pretoken_len - 1:
                pairs_to_pretokens_index[(pretoken[i], pretoken[i + 1])].add(pretoken)
                i += 1
        return pairs_to_pretokens_index

    def get_highest_count_pair(self) -> tuple[bytes, bytes]:
        return self.pretoken_counter.highest()

    @profile
    def merge(self, pair: tuple[bytes, bytes]) -> None:
        # iterate over pretokens that contain this pair only
        if pair not in self.pairs_to_pretokens_index:
            return
        for pretoken in list(self.pairs_to_pretokens_index[pair]):
            count: int = self.pretoken_vocab[pretoken]
            new_pretoken = merge_pretoken(pair, pretoken)
            old_counts: dict[tuple[bytes, bytes], int] = self._count_byte_pairs(pretoken)
            new_counts: dict[tuple[bytes, bytes], int] = self._count_byte_pairs(new_pretoken)
            count_deltas: dict[tuple[bytes, bytes], int] = self.get_count_deltas(old_counts, new_counts)
            for merged_pair, delta in count_deltas.items():
                self.pretoken_counter.add(merged_pair, count * delta)
            for old_pair in old_counts.keys():
                bucket: set[tuple[bytes, ...]] = self.pairs_to_pretokens_index[old_pair]
                bucket.remove(pretoken)
                if not bucket:
                    del self.pairs_to_pretokens_index[old_pair]
            for new_pair in new_counts.keys():
                self.pairs_to_pretokens_index[new_pair].add(new_pretoken)
            del self.pretoken_vocab[pretoken]
            self.pretoken_vocab[new_pretoken] += count

    @classmethod
    def _count_byte_pairs(cls, pretoken: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
        counts: dict[tuple[bytes, bytes], int] = {}
        for byte_pair in zip(pretoken, pretoken[1:]):
            if byte_pair not in counts:
                counts[byte_pair] = 1
            else:
                counts[byte_pair] += 1
        return counts

    @classmethod
    def get_count_deltas(cls, old_counts: dict[tuple[bytes, bytes], int], new_counts: dict[tuple[bytes, bytes], int]) -> dict[tuple[bytes, bytes], int]:
        count_deltas: dict[tuple[bytes, bytes], int] = new_counts.copy()
        for new_pair, new_count in new_counts.items():
            if new_pair in old_counts:
                count_deltas[new_pair] -= old_counts[new_pair]
                if count_deltas[new_pair] == 0:
                    del count_deltas[new_pair]
        for old_pair, old_count in old_counts.items():
            if old_pair not in new_counts:
                count_deltas[old_pair] = -old_count
        return count_deltas
