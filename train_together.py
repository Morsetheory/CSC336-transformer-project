from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.experiment_tracking import ExperimentTracker
from cs336_basics.layers import (
    CausalMultiHeadSelfAttention_with_RoPe,
    Embedding,
    Linear,
    MyAdamLike,
    SwiGLU,
    gradient_clipping,
    load_checkpoint,
    rmsnorm,
    save_checkpoint,
    cross_entropy,
    get_lr_cosine_schedule,

)
from cs336_basics.tokenizer import Tokenizer

try:
    import wandb
except Exception:  # pragma: no cover - wandb is optional at runtime.
    wandb = None
 

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_theta: float, max_seq_len: int):
        super().__init__()
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.ln1 = rmsnorm(d_model)
        self.attn = CausalMultiHeadSelfAttention_with_RoPe(d_model=d_model, num_heads=num_heads)
        self.ln2 = rmsnorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x),
            token_positions=token_positions,
            theta=self.rope_theta,
            max_seq_len=self.max_seq_len,
        )
        x = x + self.ffn(self.ln2(x))
        return x


class ByteTransformerLM(nn.Module):
    """A byte-level LM built from cs336_basics.layers components."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.token_embed = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope_theta=rope_theta,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = rmsnorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(x)
        batch_size, seq_len = x.shape
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        for block in self.blocks:
            h = block(h, token_positions)
        h = self.ln_final(h)
        return self.lm_head(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level LM on memmapped user data.")

    data = parser.add_argument_group("data")
    data.add_argument("--train-bin", type=Path, default=None, help="Path to training tokens (.bin, uint16).")
    data.add_argument("--val-bin", type=Path, default=None, help="Path to validation tokens (.bin, uint16).")
    data.add_argument("--text", type=str, default=None, help="Raw text to auto-prepare into memmap data.")
    data.add_argument("--text-file", type=Path, default=None, help="Path to raw text to auto-prepare.")
    data.add_argument(
        "--tokenizer-vocab",
        type=Path,
        default=None,
        help="Optional tokenizer vocab artifact to encode raw text before training.",
    )
    data.add_argument(
        "--tokenizer-merges",
        type=Path,
        default=None,
        help="Optional tokenizer merges artifact to encode raw text before training.",
    )
    data.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="Special token(s) for tokenizer loading. Can be passed multiple times.",
    )
    data.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("artifacts/training_together_data"),
        help="Output dir for auto-prepared memmap data.",
    )
    data.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction when auto-preparing.")

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, default=256)
    model.add_argument("--d-model", type=int, default=256)
    model.add_argument("--num-layers", type=int, default=2)
    model.add_argument("--num-heads", type=int, default=8)
    model.add_argument("--d-ff", type=int, default=1024)
    model.add_argument("--rope-theta", type=float, default=10000.0)

    optim = parser.add_argument_group("optimizer")
    optim.add_argument("--lr", type=float, default=3e-4)
    optim.add_argument("--min-lr", type=float, default=3e-5)
    optim.add_argument("--weight-decay", type=float, default=0.01)
    optim.add_argument("--beta1", type=float, default=0.9)
    optim.add_argument("--beta2", type=float, default=0.95)
    optim.add_argument("--grad-clip", type=float, default=1.0)

    train = parser.add_argument_group("training")
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--seq-len", type=int, default=128)
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--eval-batches", type=int, default=20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    io = parser.add_argument_group("checkpointing")
    io.add_argument("--checkpoint-path", type=Path, default=Path("artifacts/checkpoint.pt"))
    io.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint path.")
    io.add_argument("--save-every", type=int, default=200)

    logging = parser.add_argument_group("logging")
    logging.add_argument("--log-every", type=int, default=20)
    logging.add_argument("--experiment-dir", type=Path, default=Path("artifacts/experiments"))
    logging.add_argument("--run-name", type=str, default=None)
    logging.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    logging.add_argument("--wandb-project", type=str, default="cs336-training-together")
    logging.add_argument("--wandb-run-name", type=str, default=None)

    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_memmap_from_text(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.text is None and args.text_file is None:
        raise ValueError("Provide either --train-bin/--val-bin or --text/--text-file.")

    if args.text is not None:
        raw_text = args.text
    else:
        raw_text = args.text_file.read_text(encoding="utf-8")

    if (args.tokenizer_vocab is None) != (args.tokenizer_merges is None):
        raise ValueError("Provide both --tokenizer-vocab and --tokenizer-merges, or neither.")

    if args.tokenizer_vocab is not None:
        tokenizer = Tokenizer.from_files(
            vocab_filepath=str(args.tokenizer_vocab),
            merges_filepath=str(args.tokenizer_merges),
            special_tokens=args.special_token,
        )
        token_ids = np.asarray(tokenizer.encode(raw_text), dtype=np.uint16)
        recommended_vocab_size = len(tokenizer.vocab)
    else:
        token_ids = np.frombuffer(raw_text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)
        recommended_vocab_size = 256

    if token_ids.size < 2:
        raise ValueError("Input text is too short to train a next-token model.")

    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    split = max(1, int((1.0 - args.val_frac) * token_ids.size))
    split = min(split, token_ids.size - 1)
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]

    train_path = args.prepared_dir / "train.bin"
    val_path = args.prepared_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "vocab_size_recommended": recommended_vocab_size,
        "num_train_tokens": int(train_ids.size),
        "num_val_tokens": int(val_ids.size),
    }
    (args.prepared_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return train_path, val_path

 
def open_memmaps(train_bin: Path, val_bin: Path) -> tuple[np.memmap, np.memmap]:
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    return train_data, val_data


def get_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_starts = len(data) - seq_len
    if num_starts <= 0:
        raise ValueError(f"Dataset too short ({len(data)} tokens) for seq_len={seq_len}.")
    starts = torch.randint(0, num_starts, (batch_size,))
    x = np.stack([data[s : s + seq_len] for s in starts.tolist()], axis=0).astype(np.int64, copy=False)
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts.tolist()], axis=0).astype(np.int64, copy=False)
    xb = torch.from_numpy(x).to(device=device, non_blocking=True)
    yb = torch.from_numpy(y).to(device=device, non_blocking=True)
    return xb, yb


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: np.memmap,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_batches):
        xb, yb = get_batch(data, batch_size, seq_len, device)
        logits = model(xb)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        losses.append(float(loss.item()))
    model.train(was_training)
    return float(np.mean(losses))


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not args.wandb:
        return
    if wandb is None:
        raise RuntimeError("wandb logging requested but `wandb` is not importable.")
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={key: _wandb_config_value(value) for key, value in config.items()},
    )


def _wandb_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _wandb_config_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wandb_config_value(item) for item in value]
    return value


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    return MyAdamLike(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )


def get_scheduled_lr(args: argparse.Namespace, step: int) -> float:
    warmup_steps = max(1, args.warmup_steps)
    cosine_cycle_steps = max(args.steps, warmup_steps + 1)
    return float(  
        get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=warmup_steps,
            cosine_cycle_iters=cosine_cycle_steps,
        )
    )


def main() -> None:
    args = parse_args()
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    if args.train_bin is None or args.val_bin is None:
        train_bin, val_bin = prepare_memmap_from_text(args)
    else:
        train_bin, val_bin = args.train_bin, args.val_bin

    train_data, val_data = open_memmaps(train_bin, val_bin)
    meta_path = train_bin.parent / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as meta_file:
            meta = json.load(meta_file)
        recommended_vocab_size = meta.get("vocab_size_recommended")
        if recommended_vocab_size is not None and int(recommended_vocab_size) != args.vocab_size:
            raise ValueError(
                f"--vocab-size={args.vocab_size} does not match prepared data "
                f"vocab_size_recommended={recommended_vocab_size} from {meta_path}."
            )
    if len(train_data) <= args.seq_len:
        raise ValueError(
            f"Training dataset is too short ({len(train_data)} tokens) for seq_len={args.seq_len}. "
            "Provide more data or reduce --seq-len."
        )
    can_run_validation = len(val_data) > args.seq_len
    if not can_run_validation:
        print(
            f"Validation dataset is too short ({len(val_data)} tokens) for seq_len={args.seq_len}; "
            "validation will be skipped."
        )

    model = ByteTransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        max_seq_len=args.seq_len,
    ).to(device)
    optimizer = build_optimizer(args, model)

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at step={start_step}")

    tracker = ExperimentTracker(
        args.experiment_dir,
        config={
            **vars(args),
            "resolved_device": str(device),
            "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "start_step": start_step,
        },
        run_name=args.run_name,
    )
    print(f"Experiment artifacts will be written to {tracker.run_dir}")

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    maybe_init_wandb(args, vars(args))

    train_start_time = time.perf_counter()
    completed = False

    try:
        for step in range(start_step + 1, args.steps + 1):
            step_start_time = time.perf_counter()
            xb, yb = get_batch(train_data, args.batch_size, args.seq_len, device)
            logits = model(xb)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

            current_lr = get_scheduled_lr(args, step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()

            step_wallclock = time.perf_counter() - train_start_time
            step_duration = time.perf_counter() - step_start_time
            tokens_seen = step * args.batch_size * args.seq_len

            if step % args.log_every == 0 or step == 1:
                train_loss = float(loss.item())
                train_ppl = math.exp(min(20.0, train_loss))
                print(f"step={step:6d} lr={current_lr:.6g} train_loss={train_loss:.4f} train_ppl={train_ppl:.2f}")
                tracker.log_metrics(
                    split="train",
                    step=step,
                    wallclock_seconds=step_wallclock,
                    loss=train_loss,
                    ppl=train_ppl,
                    lr=current_lr,
                    tokens_seen=tokens_seen,
                    step_time_seconds=step_duration,
                )
                if args.wandb:
                    wandb.log(
                        {
                            "step": step,
                            "wallclock_seconds": step_wallclock,
                            "train/lr": current_lr,
                            "train/loss": train_loss,
                            "train/ppl": train_ppl,
                            "train/tokens_seen": tokens_seen,
                            "train/step_time_seconds": step_duration,
                        }
                    )

            if can_run_validation and (step % args.eval_every == 0 or step == args.steps):
                eval_start_time = time.perf_counter()
                val_loss = evaluate(
                    model=model,
                    data=val_data,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    eval_batches=args.eval_batches,
                    device=device,
                )
                eval_duration = time.perf_counter() - eval_start_time
                eval_wallclock = time.perf_counter() - train_start_time
                val_ppl = math.exp(min(20.0, val_loss))
                print(f"step={step:6d} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
                tracker.log_metrics(
                    split="val",
                    step=step,
                    wallclock_seconds=eval_wallclock,
                    loss=val_loss,
                    ppl=val_ppl,
                    tokens_seen=tokens_seen,
                    eval_time_seconds=eval_duration,
                )
                if args.wandb:
                    wandb.log(
                        {
                            "step": step,
                            "wallclock_seconds": eval_wallclock,
                            "val/loss": val_loss,
                            "val/ppl": val_ppl,
                            "val/tokens_seen": tokens_seen,
                            "val/eval_time_seconds": eval_duration,
                        }
                    )

            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=args.checkpoint_path)
                checkpoint_wallclock = time.perf_counter() - train_start_time
                tracker.log_checkpoint(
                    step=step,
                    wallclock_seconds=checkpoint_wallclock,
                    checkpoint_path=args.checkpoint_path,
                )
                if args.wandb:
                    wandb.log(
                        {
                            "step": step,
                            "wallclock_seconds": checkpoint_wallclock,
                            "checkpoint_path": str(args.checkpoint_path),
                        }
                    )
        completed = True
    finally:
        tracker.finish(status="completed" if completed else "failed", final_step=args.steps if completed else start_step)
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
