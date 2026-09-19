from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.decoder import generate, load_model
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained TinyStories model.")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("artifacts/tinystories_10k_4l_512d.pt"))
    parser.add_argument("--tokenizer-vocab", type=Path, default=Path("artifacts/tinystories_bpe_10k/vocab.pkl"))
    parser.add_argument("--tokenizer-merges", type=Path, default=Path("artifacts/tinystories_bpe_10k/merges.pkl"))
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--output-file", type=Path, default=None)

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--seq-len", type=int, default=256)
    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return "mps"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    tokenizer = Tokenizer.from_files(
        str(args.tokenizer_vocab),
        str(args.tokenizer_merges),
        special_tokens=["<|endoftext|>"],
    )
    model = load_model(
        args.checkpoint_path,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        max_seq_len=args.seq_len,
        device=device,
    )

    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.seq_len,
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
