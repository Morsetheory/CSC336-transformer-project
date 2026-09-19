from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Optional
import math
import torch
import torch.nn as nn
import os 
from typing import IO, Any, BinaryIO
from einops import rearrange, einsum
import numpy.typing as npt
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections import defaultdict
import regex as re

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        # Truncated normal: values drawn near mean, clipped to [a, b]
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std= math.sqrt(2 / (self.in_features + self.out_features)),  # often ~ 1/sqrt(in_features)
            a=- 3 * math.sqrt(2 / (self.in_features + self.out_features)),   # often ~ -2*std
            b= 3 * math.sqrt(2 / (self.in_features + self.out_features))     # often ~ +2*std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        y = x @ self.weight.T
        return y

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Weight shape: (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Truncated normal: values drawn near mean, clipped to [a, b]
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std= 1,  # often ~ 1/sqrt(num_embeddings)
            a=- 3 * math.sqrt(2 / (self.num_embeddings + self.embedding_dim)),   # often ~ -2*std
            b= 3 * math.sqrt(2 / (self.num_embeddings + self.embedding_dim))     # often ~ +2*std
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...,) of integers in [0, num_embeddings-1] -> (..., embedding_dim)
        y = self.weight[token_ids]
        return y
    
class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Weight shape: (d_model,)
        self.weight = nn.Parameter(torch.empty(d_model))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize to all ones
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model) -> (..., d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)  # ensure same dtype for computation
        norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        weight = self.weight.to(x.dtype)  # ensure same dtype for multiplication
        y = x / norm * weight
        y = y.to(in_dtype)  # convert back to original dtype
        return y

class SwiGLU(nn.Module):
    def __init__(self, d_model: int,
    d_ff: int, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))
        self.reset_parameters()
    
    def reset_parameters(self):
        # Truncated normal: values drawn near mean, clipped to [a, b]
        nn.init.trunc_normal_(
            self.w1,
            mean=0.0,
            std= math.sqrt(2 / (self.d_model + self.d_ff)),  # often ~ 1/sqrt(in_features)
            a=- 3 * math.sqrt(2 / (self.d_model + self.d_ff)),   # often ~ -2*std
            b= 3 * math.sqrt(2 / (self.d_model + self.d_ff))     # often ~ +2*std
        )

        nn.init.trunc_normal_(
            self.w2,
            mean=0.0,
            std= math.sqrt(2 / (self.d_model + self.d_ff)),  # often ~ 1/sqrt(in_features)
            a=- 3 * math.sqrt(2 / (self.d_model + self.d_ff)),   # often ~ -2*std
            b= 3 * math.sqrt(2 / (self.d_model + self.d_ff))     # often ~ +2*std
        )

        nn.init.trunc_normal_(
            self.w3,
            mean=0.0,
            std= math.sqrt(2 / (self.d_model + self.d_ff)),  # often ~ 1/sqrt(in_features)
            a=- 3 * math.sqrt(2 / (self.d_model + self.d_ff)),   # often ~ -2*std
            b= 3 * math.sqrt(2 / (self.d_model + self.d_ff))     # often ~ +2*std
        )
    
    def forward(self, x: torch.Tensor):
        a = x @ self.w1.T
        b = a * torch.sigmoid(a)
        c = b * (x @ self.w3.T)
        d = c @ self.w2.T
        return d



class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")

        self.d_k = d_k
        self.theta = float(theta)
        self.max_seq_len = int(max_seq_len)

        # Precompute angle table once: (max_seq_len, d_k/2)
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
        angles = torch.outer(positions, inv_freq)

        # Fixed RoPE tables: move with module device/dtype, but do not persist in checkpoints.
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected last dim {self.d_k}, got {x.shape[-1]}")
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                f"token_positions last dim ({token_positions.shape[-1]}) must equal seq_len ({x.shape[-2]})"
            )

        pos = token_positions.to(device=x.device, dtype=torch.long)
        while pos.ndim < x.ndim - 1:
            # Allows broadcasting over extra dims like attention heads.
            pos = pos.unsqueeze(-2)

        if pos.numel() > 0:
            max_pos = int(pos.max().item())
            if max_pos >= self.max_seq_len:
                raise ValueError(
                    f"token_positions has value {max_pos}, but max_seq_len is {self.max_seq_len}"
                )

        cos_cached = self.cos_cached.to(device=x.device)
        sin_cached = self.sin_cached.to(device=x.device)
        cos = cos_cached[pos].to(dtype=x.dtype)
        sin = sin_cached[pos].to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(d_k, theta, max_seq_len)
    return rope(in_query_or_key, token_positions)



def _split_heads(x: torch.Tensor, num_heads, head_dim) -> torch.Tensor:
        # (..., seq_len, d_model) -> (..., num_heads, seq_len, head_dim)
        x = x.unflatten(-1, (num_heads, head_dim))
        return x.movedim(-2, -3)

def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        # (..., num_heads, seq_len, head_dim) -> (..., seq_len, d_model)
        x = x.movedim(-3, -2)
        return x.flatten(-2, -1)

def run_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Given key (K), query (Q), and value (V) tensors, return
        the output of your scaled dot product attention implementation.

        Args:
            Q (torch.Tensor): Query tensor
            K (torch.Tensor): Key tensor
            V (torch.Tensor): Values tensor
            mask (torch.Tensor | None): Mask tensor
        Returns:
            torch.Tensor: Output of SDPA
        """
        d_k = Q.shape[-1]
        scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)
        if mask is not None:
            keep = mask.to(dtype=torch.bool, device=scores.device)
            if keep.ndim == scores.ndim - 1:
                # Broadcast (..., queries, keys) mask across the head axis.
                keep = keep.unsqueeze(-3)
            scores = scores.masked_fill(~keep, float("-inf"))
        probs = softmax(scores, dim=-1)
        return probs @ V

def _causal_mask(query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        q_idx = torch.arange(query_len, device=device).unsqueeze(-1)
        k_idx = torch.arange(key_len, device=device).unsqueeze(0)
        return k_idx <= q_idx

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        q = _split_heads(self.q_proj(x), self.num_heads, self.head_dim)
        k = _split_heads(self.k_proj(x), self.num_heads, self.head_dim)
        v = _split_heads(self.v_proj(x),self.num_heads, self.head_dim)

        if mask is None:
            keep = _causal_mask(q.shape[-2], k.shape[-2], q.device)
            keep = keep.view((1,) * (q.ndim - 2) + keep.shape)
        else:
            keep = mask

        out = run_scaled_dot_product_attention(q, k, v, keep)
        out = _merge_heads(out)
        return self.output_proj(out)
    
class CausalMultiHeadSelfAttention_with_RoPe(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, token_positions: Int[Tensor, " ... sequence_length"], theta: float, max_seq_len: int = 2048, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        q = _split_heads(self.q_proj(x), self.num_heads, self.head_dim)
        k = _split_heads(self.k_proj(x), self.num_heads, self.head_dim)
        q = run_rope(self.head_dim, theta, max_seq_len, q, token_positions)
        k = run_rope(self.head_dim, theta, max_seq_len, k, token_positions)
        v = _split_heads(self.v_proj(x),self.num_heads, self.head_dim)

        if mask is None:
            keep = _causal_mask(q.shape[-2], k.shape[-2], q.device)
            keep = keep.view((1,) * (q.ndim - 2) + keep.shape)
        else:
            keep = mask

        out = run_scaled_dot_product_attention(q, k, v, keep)
        out = _merge_heads(out)
        return self.output_proj(out)
    
class MyAdamLike(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:  # Initialize optimizer state for this parameter.
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                grad = p.grad
                m, v = state["m"], state["v"]

                state["t"] += 1
                t = state["t"]

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                step_size = lr / bias_correction1
                denom = v.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)

                if wd != 0:
                    p.add_(p, alpha=-lr * wd)
                p.addcdiv_(m, denom, value=-step_size)
        return loss


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["iteration"])

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    x_shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    target_logits = x_shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    log_denom = torch.logsumexp(x_shifted, dim=-1)
    loss = -(target_logits - log_denom)
    return loss.mean()

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    if warmup_iters <= it <= cosine_cycle_iters:
        increment = 1/2 * (1 + math.cos((it - warmup_iters) * math.pi/(cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
        return min_learning_rate + increment
    else:
        return min_learning_rate
    
def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    x_shifted = in_features - in_features.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x_shifted)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps = 1e-6) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    total_l2 = torch.sqrt(sum((p.grad**2).sum() for p in params))
    
    if total_l2 >= max_l2_norm:
        scale = max_l2_norm / (total_l2 + eps)
        for p in params:
            p.grad.mul_(scale)

from typing import TYPE_CHECKING

from cs336_basics.tokenizer import Tokenizer

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
    end_of_text_token: str = "<|endoftext|>",
) -> str:
    device = torch.device(device)
    model.eval()

    input_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    eot_id = None
    eot_bytes = end_of_text_token.encode("utf-8")
    if eot_bytes in tokenizer.byte_to_id:
        eot_id = tokenizer.byte_to_id[eot_bytes]

    for _ in range(max_new_tokens):
        context = tokens[:, -max_seq_len:]
        logits = model(context)
        next_token_logits = logits[:, -1, :]

        if temperature == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_token_logits / temperature
            probs = softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

        if eot_id is not None and next_token.item() == eot_id:
            break

    return tokenizer.decode(tokens[0].tolist())



    
