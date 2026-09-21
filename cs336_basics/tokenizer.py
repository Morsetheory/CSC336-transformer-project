from __future__ import annotations

import json
import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path
from cs336_basics.pretokenization import iter_pretokens


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        # Ensure provided special tokens can always be encoded as single tokens.
        next_id = (max(self.vocab.keys()) + 1) if self.vocab else 0
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.byte_to_id:
                self.vocab[next_id] = tok_bytes
                self.byte_to_id[tok_bytes] = next_id
                next_id += 1

        # Merge ranks are stored by ID pairs so encode works on integer IDs.
        self.merge_ranks: dict[tuple[int, int], int] = {}
        for rank, (left_bytes, right_bytes) in enumerate(self.merges):
            left_id = self.byte_to_id[left_bytes]
            right_id = self.byte_to_id[right_bytes]
            self.merge_ranks[(left_id, right_id)] = rank

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)

        # Preferred format from this repo's BPE training script.
        if vocab_path.suffix == ".pkl" and merges_path.suffix == ".pkl":
            with open(vocab_path, "rb") as vocab_file:
                vocab = pickle.load(vocab_file)
            with open(merges_path, "rb") as merges_file:
                merges = pickle.load(merges_file)
            return cls(vocab, merges, special_tokens)

        # Fallback for JSON vocab + text merges.
        with open(vocab_path, encoding="utf-8") as vocab_file:
            raw_vocab: dict[str, int] = json.load(vocab_file)
        with open(merges_path, encoding="utf-8") as merges_file:
            raw_merges = [
                tuple(line.strip().split(" "))
                for line in merges_file
                if line.strip() and not line.startswith("#")
            ]

        vocab = {idx: token.encode("utf-8") for token, idx in raw_vocab.items()}
        merges = [(left.encode("utf-8"), right.encode("utf-8")) for left, right in raw_merges]
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _merge_pair_in_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        a, b = pair
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _encode_pretoken(self, pretok: str) -> list[int]:
        # Convert raw bytes to tokenizer IDs; do not assume id == byte value.
        ids = [self.byte_to_id[bytes([b])] for b in pretok.encode("utf-8")]
        while len(ids) >= 2:
            best_pair = None
            best_rank = None
            for a, b in zip(ids[:-1], ids[1:]):
                rank = self.merge_ranks.get((a, b))
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = (a, b)
            if best_pair is None:
                break
            new_id = self.byte_to_id[self.vocab[best_pair[0]] + self.vocab[best_pair[1]]]
            ids = self._merge_pair_in_ids(ids, best_pair, new_id)
        return ids

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable((text,)))

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for token, is_special in iter_pretokens(iterable, self.special_tokens):
            if is_special:
                yield self.byte_to_id[token.encode("utf-8")]
            else:
                yield from self._encode_pretoken(token)


# Backward compatibility if older code imports lowercase class name.
tokenizer = Tokenizer
