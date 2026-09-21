from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.layers import softmax

if TYPE_CHECKING:
    from train_together import ByteTransformerLM

def load_model(
    checkpoint_path: str | Path,
    *,
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    max_seq_len: int,
    device: torch.device | str = "cpu",
) -> "ByteTransformerLM":
    device = torch.device(device)
    from train_together import ByteTransformerLM

    model = ByteTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = softmax(logits / temperature, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Include the token that first reaches the requested probability mass.
        keep_sorted = torch.ones_like(sorted_probs, dtype=torch.bool)
        keep_sorted[..., 1:] = cumulative_probs[..., :-1] < top_p

        filtered_sorted_probs = torch.where(keep_sorted, sorted_probs, torch.zeros_like(sorted_probs))
        filtered_probs = torch.zeros_like(probs).scatter(-1, sorted_indices, filtered_sorted_probs)
        probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_seq_len: int,
    device: torch.device | str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_of_text_token: str = "<|endoftext|>",
) -> str:
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

    device = torch.device(device)
    model.eval()

    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        raise ValueError("prompt must encode to at least one token")

    tokens = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    eot_id = None
    eot_bytes = end_of_text_token.encode("utf-8")
    if eot_bytes in tokenizer.byte_to_id:
        eot_id = tokenizer.byte_to_id[eot_bytes]

    for _ in range(max_new_tokens):
        context = tokens[:, -max_seq_len:]
        logits = model(context)
        next_token_logits = logits[:, -1, :]
        next_token = _sample_from_logits(
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
        )
        tokens = torch.cat([tokens, next_token], dim=1)

        if eot_id is not None and next_token.item() == eot_id:
            break

    return tokenizer.decode(tokens[0].tolist())
