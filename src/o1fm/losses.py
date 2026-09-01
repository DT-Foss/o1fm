"""Losses for monotone quantile forecasting."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _prepare_mask(mask: Tensor | None, target: Tensor) -> Tensor:
  if mask is None:
    return torch.ones_like(target, dtype=torch.bool)
  if mask.shape != target.shape:
    raise ValueError("mask must match target shape")
  return mask.to(dtype=torch.bool)


def pinball_loss(predictions: Tensor, target: Tensor, quantiles: Tensor | list[float], *, mask: Tensor | None = None) -> Tensor:
  """Mean pinball loss over the quantile axis."""
  if predictions.ndim != target.ndim + 1:
    raise ValueError("predictions must have one extra quantile dimension")
  if predictions.shape[:-1] != target.shape:
    raise ValueError("predictions and target must align except for quantiles")
  q = torch.as_tensor(quantiles, dtype=predictions.dtype, device=predictions.device)
  if q.ndim != 1 or q.numel() != predictions.shape[-1]:
    raise ValueError("quantiles must match the prediction quantile axis")
  if not bool(torch.isfinite(predictions).all() and torch.isfinite(target).all()):
    raise ValueError("predictions and target must be finite")
  if not bool(torch.all((q > 0) & (q < 1))):
    raise ValueError("quantiles must lie in (0, 1)")

  valid = _prepare_mask(mask, target).unsqueeze(-1)
  errors = target.unsqueeze(-1) - predictions
  loss = torch.maximum(q * errors, (q - 1.0) * errors)
  loss = torch.where(valid, loss, torch.zeros_like(loss))
  denom = valid.sum().clamp_min(1).to(dtype=loss.dtype) * q.numel()
  return loss.sum() / denom


def monotone_penalty(predictions: Tensor) -> Tensor:
  """Penalty for quantile crossings."""
  if predictions.ndim < 1:
    raise ValueError("predictions must include a quantile axis")
  if predictions.shape[-1] < 2:
    return predictions.new_zeros(())
  deltas = predictions[..., 1:] - predictions[..., :-1]
  return F.relu(-deltas).mean()


def forecast_loss(
  predictions: Tensor,
  target: Tensor,
  quantiles: Tensor | list[float],
  *,
  mask: Tensor | None = None,
  monotone_weight: float = 0.1,
) -> Tensor:
  """Composite objective for forecast training."""
  return pinball_loss(predictions, target, quantiles, mask=mask) + float(monotone_weight) * monotone_penalty(predictions)
