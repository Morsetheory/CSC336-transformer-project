from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer on TinyStories."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
        help="Path to TinyStories training text file.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="Total vocabulary size (includes all special tokens).",
    )
    parser.add_argument(
        "--special-token",
        type=str,
        default="<|endoftext|>",
        help="Special token to include in the tokenizer vocabulary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tinystories_bpe_10k"),
        help="Directory to write vocab/merges artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=[args.special_token],
    )
    elapsed = time.time() - start

    vocab_path = args.output_dir / "vocab.pkl"
    merges_path = args.output_dir / "merges.pkl"
    metadata_path = args.output_dir / "metadata.json"

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_path": str(args.input_path),
                "vocab_size": args.vocab_size,
                "special_tokens": [args.special_token],
                "num_vocab_items": len(vocab),
                "num_merges": len(merges),
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
        )

    print(f"Saved vocab to: {vocab_path}")
    print(f"Saved merges to: {merges_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Training time: {elapsed:.2f}s")
    print(f"Vocab size: {len(vocab)}")
    print(f"Merge count: {len(merges)}")


if __name__ == "__main__":
    main()
