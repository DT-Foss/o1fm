"""Attention and normalization primitives for O1FM."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _check_finite(x: Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must be finite")


def patchify(
    x: Tensor, patch_len: int, *, pad_value: float = 0.0
) -> tuple[Tensor, int]:
    """Split a ``[B, C, T]`` tensor into ``[B, C, P, patch_len]`` patches."""
    if x.ndim != 3:
        raise ValueError("expected [batch, channels, time]")
    if patch_len <= 0:
        raise ValueError("patch_len must be positive")
    batch, channels, time = x.shape
    pad = (-time) % patch_len
    if pad:
        x = F.pad(x, (0, pad), value=float(pad_value))
    patches = x.reshape(batch, channels, -1, patch_len)
    return patches, pad


def revin(
    x: Tensor,
    mask: Tensor,
    *,
    eps: float = 1e-6,
    reverse: bool = False,
    mean: Tensor | None = None,
    std: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-variate RevIN over observed positions.

    Args:
      x: ``[B, C, T]`` tensor.
      mask: Boolean tensor with the same shape, ``True`` for observed values.
      eps: Numerical floor for the standard deviation.
      reverse: If ``True``, denormalize using ``mean`` and ``std``.
      mean/std: Optional broadcastable statistics for reverse mode.
    """
    if x.ndim != 3 or mask.shape != x.shape:
        raise ValueError("x and mask must both have shape [batch, channels, time]")
    if not bool(mask.dtype == torch.bool):
        raise ValueError("mask must be boolean")
    _check_finite(x, name="values")

    if reverse:
        if mean is None or std is None:
            raise ValueError("reverse RevIN requires mean and std")
        return x * std + mean, mean, std

    observed = mask.to(dtype=x.dtype)
    count = observed.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (x * observed).sum(dim=-1, keepdim=True) / count
    centered = (x - mean) * observed
    var = (centered.square().sum(dim=-1, keepdim=True) / count).clamp_min(eps * eps)
    std = var.sqrt()
    normed = torch.where(mask, (x - mean) / std, torch.zeros_like(x))
    return normed, mean, std


def sinusoidal_positions(
    length: int, dim: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if length < 1:
        raise ValueError("length must be positive")
    if dim < 2 or dim % 2:
        raise ValueError("dim must be an even positive integer")
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding


def sinkhorn_transport(
    logits: Tensor,
    *,
    key_mask: Tensor | None = None,
    iterations: int = 6,
    epsilon: float = 1e-6,
) -> Tensor:
    """Approximate doubly-stochastic transport over the last two dimensions."""
    if logits.ndim != 4:
        raise ValueError("logits must have shape [batch, heads, rows, cols]")
    work = logits.to(dtype=torch.float32)
    work = work - work.amax(dim=(-1, -2), keepdim=True)
    weights = work.exp()

    if key_mask is not None:
        if key_mask.shape != logits.shape[:1] + logits.shape[-1:]:
            raise ValueError("key_mask must have shape [batch, cols]")
        weights = weights * key_mask[:, None, None, :].to(dtype=weights.dtype)

    for _ in range(max(1, iterations)):
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
        weights = weights / weights.sum(dim=-2, keepdim=True).clamp_min(epsilon)

    if key_mask is not None:
        no_valid = ~key_mask.any(dim=-1)
        if bool(no_valid.any()):
            rows = logits.shape[-2]
            eye = torch.eye(rows, device=logits.device, dtype=weights.dtype).view(
                1, 1, rows, rows
            )
            weights = torch.where(no_valid[:, None, None, None], eye, weights)

    return weights.to(dtype=logits.dtype)


class RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        _check_finite(x, name="RMSNorm input")
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / norm) * self.scale


class SwiGLU(nn.Module):
    """Two-layer SwiGLU feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class TemporalSelfAttention(nn.Module):
    """Strictly causal attention over patches for each variate."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, *, key_mask: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("expected [batch, variates, patches, dim]")
        if key_mask.shape != x.shape[:3]:
            raise ValueError("key_mask must match x without the feature axis")
        batch, variates, patches, dim = x.shape
        work = self.norm(x).reshape(batch * variates, patches, dim)
        mask = key_mask.reshape(batch * variates, patches)

        q, k, v = self.qkv(work).chunk(3, dim=-1)
        q = q.view(batch * variates, patches, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch * variates, patches, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch * variates, patches, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal = torch.tril(
            torch.ones(patches, patches, device=x.device, dtype=torch.bool)
        )
        allowed = causal.unsqueeze(0).unsqueeze(0) & mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        out = (
            torch.matmul(weights, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch * variates, patches, dim)
        )
        return self.out(out).view(batch, variates, patches, dim)


class CrossVariateSinkhornAttention(nn.Module):
    """Full cross-variate mixing with Sinkhorn-normalized transport."""

    def __init__(self, dim: int, heads: int, *, sinkhorn_iterations: int = 6) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.sinkhorn_iterations = sinkhorn_iterations
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, *, key_mask: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("expected [batch, variates, patches, dim]")
        if key_mask.shape != x.shape[:3]:
            raise ValueError("key_mask must match x without the feature axis")
        batch, variates, patches, dim = x.shape
        work = self.norm(x).permute(0, 2, 1, 3).reshape(batch * patches, variates, dim)
        mask = key_mask.permute(0, 2, 1).reshape(batch * patches, variates)

        q, k, v = self.qkv(work).chunk(3, dim=-1)
        q = q.view(batch * patches, variates, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch * patches, variates, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch * patches, variates, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        transport = sinkhorn_transport(
            scores,
            key_mask=mask,
            iterations=self.sinkhorn_iterations,
        )
        out = (
            torch.matmul(transport, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch * patches, variates, dim)
        )
        out = self.out(out).view(batch, patches, variates, dim).permute(0, 2, 1, 3)
        return out


class ForecastBlock(nn.Module):
    """Alternating temporal and cross-variate block."""

    def __init__(self, dim: int, heads: int, *, sinkhorn_iterations: int = 6) -> None:
        super().__init__()
        self.temporal_norm = RMSNorm(dim)
        self.cross_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.temporal = TemporalSelfAttention(dim, heads)
        self.cross = CrossVariateSinkhornAttention(
            dim, heads, sinkhorn_iterations=sinkhorn_iterations
        )
        self.ffn = SwiGLU(dim, hidden_dim=dim * 2)

    def forward(self, x: Tensor, *, key_mask: Tensor) -> Tensor:
        x = x + self.temporal(self.temporal_norm(x), key_mask=key_mask)
        x = x + self.cross(self.cross_norm(x), key_mask=key_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
