"""Forecasting model built on causal temporal and Sinkhorn variate mixing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor, nn

from .attention import ForecastBlock, RMSNorm, patchify, revin, sinusoidal_positions


DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True, slots=True)
class ForecastBatch:
  """Model input bundle.

  values:
    ``[batch, variates, time]`` tensor containing all roles packed together.
  observed_mask:
    Boolean tensor with the same shape. ``True`` marks observed values.
  role_ids:
    One integer role per variate. Supported roles are:
    ``0 = target``, ``1 = past_only_covariate``, ``2 = known_future_covariate``.
  horizon:
    Number of forecast steps at the suffix of ``time``.
  """

  values: Tensor
  observed_mask: Tensor
  role_ids: Tensor
  horizon: int


@dataclass(frozen=True, slots=True)
class ForecastResult:
  """Forecast output with a median point forecast."""

  median: Tensor
  quantiles: Tensor


@dataclass(frozen=True, slots=True)
class O1FMConfig:
  """Hyperparameters for the compact forecast backbone."""

  patch_len: int = 32
  d_model: int = 128
  num_heads: int = 4
  num_layers: int = 4
  hidden_mult: int = 2
  sinkhorn_iterations: int = 6
  quantiles: tuple[float, ...] = DEFAULT_QUANTILES
  revin_eps: float = 1e-6
  max_time_scale: int = 10000

  def __post_init__(self) -> None:
    if self.patch_len <= 0:
      raise ValueError("patch_len must be positive")
    if self.d_model <= 0 or self.num_heads <= 0 or self.num_layers <= 0:
      raise ValueError("model dimensions must be positive")
    if self.d_model % self.num_heads:
      raise ValueError("d_model must be divisible by num_heads")
    if self.d_model % 2:
      raise ValueError("d_model must be even for sinusoidal position encoding")
    if len(self.quantiles) < 3 or len(self.quantiles) % 2 == 0:
      raise ValueError("quantiles must contain an odd number of entries")
    if not all(0.0 < q < 1.0 for q in self.quantiles):
      raise ValueError("quantiles must lie in (0, 1)")
    if tuple(self.quantiles) != tuple(sorted(self.quantiles)):
      raise ValueError("quantiles must be strictly increasing")


class MonotoneQuantileHead(nn.Module):
  """Produces strictly monotone quantiles with a middle-point forecast."""

  def __init__(self, dim: int, patch_len: int, quantiles: Iterable[float]) -> None:
    super().__init__()
    quantiles = tuple(float(q) for q in quantiles)
    if len(quantiles) < 3 or len(quantiles) % 2 == 0:
      raise ValueError("quantiles must contain an odd number of entries")
    self.quantiles = quantiles
    self.median_index = len(quantiles) // 2
    self.patch_len = patch_len
    self.median = nn.Linear(dim, patch_len)
    self.lower = nn.Linear(dim, self.median_index * patch_len)
    self.upper = nn.Linear(dim, self.median_index * patch_len)

  def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    batch, variates, patches, dim = x.shape
    median = self.median(x).view(batch, variates, patches, 1, self.patch_len)
    lower = torch.nn.functional.softplus(self.lower(x)).view(
      batch, variates, patches, self.median_index, self.patch_len
    )
    upper = torch.nn.functional.softplus(self.upper(x)).view(
      batch, variates, patches, self.median_index, self.patch_len
    )

    lower_offsets = torch.flip(torch.cumsum(lower, dim=-2), dims=(-2,))
    upper_offsets = torch.cumsum(upper, dim=-2)

    lower_q = median - lower_offsets
    middle_q = median
    upper_q = median + upper_offsets
    quantiles = torch.cat((lower_q, middle_q, upper_q), dim=-2)
    median_point = median.squeeze(-2)
    return median_point, quantiles


class O1FMForecaster(nn.Module):
  """Independent multivariate forecaster with causal temporal and Sinkhorn mixing."""

  def __init__(self, config: O1FMConfig | None = None) -> None:
    super().__init__()
    self.config = config or O1FMConfig()
    self.input_proj = nn.Linear(self.config.patch_len * 2, self.config.d_model)
    self.role_embed = nn.Embedding(3, self.config.d_model)
    self.blocks = nn.ModuleList(
      [
        ForecastBlock(
          self.config.d_model,
          self.config.num_heads,
          sinkhorn_iterations=self.config.sinkhorn_iterations,
        )
        for _ in range(self.config.num_layers)
      ]
    )
    self.final_norm = RMSNorm(self.config.d_model)
    self.head = MonotoneQuantileHead(
      self.config.d_model,
      self.config.patch_len,
      self.config.quantiles,
    )

  def _validate(self, batch: ForecastBatch) -> tuple[Tensor, Tensor, Tensor]:
    values = batch.values
    observed_mask = batch.observed_mask
    role_ids = batch.role_ids
    if values.ndim != 3:
      raise ValueError("values must have shape [batch, variates, time]")
    if observed_mask.shape != values.shape:
      raise ValueError("observed_mask must match values")
    if role_ids.ndim != 1 or role_ids.shape[0] != values.shape[1]:
      raise ValueError("role_ids must have one entry per variate")
    if batch.horizon <= 0 or batch.horizon >= values.shape[-1]:
      raise ValueError("horizon must be positive and shorter than the time axis")
    if not bool(torch.isfinite(values).all()):
      raise ValueError("values must be finite")
    if not bool(observed_mask.dtype == torch.bool):
      raise ValueError("observed_mask must be boolean")
    if not bool(((role_ids >= 0) & (role_ids <= 2)).all()):
      raise ValueError("role_ids must use the 0/1/2 contract")
    return values, observed_mask, role_ids.to(dtype=torch.long)

  def _effective_mask(
    self, observed_mask: Tensor, role_ids: Tensor, horizon: int
  ) -> Tensor:
    context_len = observed_mask.shape[-1] - horizon
    mask = observed_mask.clone()
    non_future_known = role_ids != 2
    mask[:, non_future_known, context_len:] = False
    return mask

  def encode(self, batch: ForecastBatch) -> tuple[Tensor, Tensor, Tensor, int, Tensor]:
    values, observed_mask, role_ids = self._validate(batch)
    batch_size, variates, time = values.shape
    effective_mask = self._effective_mask(observed_mask, role_ids, batch.horizon)
    normed, mean, std = revin(values, effective_mask, eps=self.config.revin_eps)
    patches, pad = patchify(normed, self.config.patch_len, pad_value=0.0)
    mask_patches, _ = patchify(
      effective_mask.to(dtype=values.dtype), self.config.patch_len, pad_value=0.0
    )
    key_mask = mask_patches.bool().any(dim=-1)

    role_embed = self.role_embed(role_ids).unsqueeze(0).unsqueeze(2)
    pos = sinusoidal_positions(
      patches.shape[2],
      self.config.d_model,
      device=values.device,
      dtype=values.dtype,
    ).unsqueeze(0).unsqueeze(0)

    patch_input = torch.cat((patches, mask_patches), dim=-1)
    state = self.input_proj(patch_input) + role_embed + pos
    for block in self.blocks:
      state = block(state, key_mask=key_mask)
    state = self.final_norm(state)
    return state, mean, std, pad, effective_mask

  def forward(self, batch: ForecastBatch) -> ForecastResult:
    values, observed_mask, role_ids = self._validate(batch)
    state, mean, std, pad, effective_mask = self.encode(batch)
    batch_size, variates, time = values.shape
    total_padded = time + pad
    patches = total_padded // self.config.patch_len

    point_patch, quantile_patch = self.head(state)
    point = point_patch.reshape(batch_size, variates, patches * self.config.patch_len)
    quantiles = quantile_patch.permute(0, 1, 2, 4, 3).reshape(
      batch_size,
      variates,
      patches * self.config.patch_len,
      len(self.config.quantiles),
    )

    point = point * std + mean
    quantiles = quantiles * std.unsqueeze(-1) + mean.unsqueeze(-1)

    point = point[..., :time]
    quantiles = quantiles[..., :time, :]

    target_mask = role_ids == 0
    if not bool(target_mask.any()):
      raise ValueError("at least one target variate is required")
    target_point = point[:, target_mask, :]
    target_quantiles = quantiles[:, target_mask, :, :]

    horizon = batch.horizon
    median = target_point[:, :, -horizon:]
    quantile_out = target_quantiles[:, :, -horizon:, :]
    return ForecastResult(median=median, quantiles=quantile_out)
