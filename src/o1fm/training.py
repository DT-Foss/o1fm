"""Tensor adaptation, compact training, and model identity for O1FM."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .contracts import ForecastExample, ForecastInputs, ForecastPrediction
from .losses import forecast_loss
from .model import ForecastBatch, O1FMForecaster


@dataclass(frozen=True, slots=True)
class TrainingReport:
    epochs: int
    examples: int
    first_loss: float
    final_loss: float
    best_loss: float


def _same_shape(inputs: Sequence[ForecastInputs]) -> None:
    if not inputs:
        raise ValueError("at least one ForecastInputs value is required")
    first = inputs[0]
    shape = (
        first.history_length,
        first.horizon,
        first.target_dim,
        first.past_covariate_dim,
        first.known_covariate_dim,
    )
    if any(
        (
            row.history_length,
            row.horizon,
            row.target_dim,
            row.past_covariate_dim,
            row.known_covariate_dim,
        )
        != shape
        for row in inputs
    ):
        raise ValueError("one tensor batch requires aligned forecast contracts")


def tensorize_inputs(
    inputs: Sequence[ForecastInputs],
    *,
    device: str | torch.device = "cpu",
) -> ForecastBatch:
    """Pack strict time-major contracts into the model's role-aware tensor."""

    rows = tuple(inputs)
    if any(not isinstance(row, ForecastInputs) for row in rows):
        raise TypeError("inputs must contain ForecastInputs values")
    _same_shape(rows)
    values = []
    masks = []
    role_ids: np.ndarray | None = None
    for row in rows:
        history = row.history_length
        horizon = row.horizon
        target_future = np.zeros((horizon, row.target_dim), dtype=np.float32)
        past_future = np.zeros((horizon, row.past_covariate_dim), dtype=np.float32)
        target_values = np.concatenate((row.past_target, target_future), axis=0)
        past_values = np.concatenate((row.past_covariates, past_future), axis=0)
        packed_values = np.concatenate(
            (target_values, past_values, row.known_covariates), axis=1
        )
        target_mask = np.concatenate(
            (
                row.past_target_mask,
                np.zeros((horizon, row.target_dim), dtype=np.bool_),
            ),
            axis=0,
        )
        past_mask = np.concatenate(
            (
                row.past_covariate_mask,
                np.zeros((horizon, row.past_covariate_dim), dtype=np.bool_),
            ),
            axis=0,
        )
        packed_mask = np.concatenate(
            (target_mask, past_mask, row.known_covariate_mask), axis=1
        )
        if packed_values.shape[0] != history + horizon:
            raise AssertionError("packed forecast timeline changed length")
        values.append(packed_values.T)
        masks.append(packed_mask.T)
        candidate_roles = np.concatenate(
            (
                np.zeros(row.target_dim, dtype=np.int64),
                np.ones(row.past_covariate_dim, dtype=np.int64),
                np.full(row.known_covariate_dim, 2, dtype=np.int64),
            )
        )
        if role_ids is None:
            role_ids = candidate_roles
        elif not np.array_equal(role_ids, candidate_roles):
            raise AssertionError("forecast role inventory changed inside a batch")
    assert role_ids is not None
    return ForecastBatch(
        values=torch.as_tensor(np.stack(values), dtype=torch.float32, device=device),
        observed_mask=torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device),
        role_ids=torch.as_tensor(role_ids, dtype=torch.long, device=device),
        horizon=rows[0].horizon,
    )


def predict(
    model: O1FMForecaster,
    inputs: ForecastInputs,
    *,
    device: str | torch.device = "cpu",
) -> ForecastPrediction:
    if not isinstance(model, O1FMForecaster):
        raise TypeError("model must be O1FMForecaster")
    model.eval()
    with torch.no_grad():
        result = model(tensorize_inputs((inputs,), device=device))
    point = result.median[0].transpose(0, 1).detach().cpu().numpy()
    quantiles = result.quantiles[0].permute(1, 0, 2).detach().cpu().numpy()
    return ForecastPrediction(
        series_id=inputs.series_id,
        timestamps=inputs.future_timestamps,
        point=point,
        quantiles=quantiles,
        quantile_levels=model.config.quantiles,
    )


def train_forecaster(
    model: O1FMForecaster,
    examples: Sequence[ForecastExample],
    *,
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 7,
    device: str | torch.device = "cpu",
) -> TrainingReport:
    if not isinstance(model, O1FMForecaster):
        raise TypeError("model must be O1FMForecaster")
    rows = tuple(examples)
    if not rows or any(not isinstance(row, ForecastExample) for row in rows):
        raise ValueError("examples must contain ForecastExample values")
    if any(row.target is None for row in rows):
        raise ValueError("training examples require future targets")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("training controls must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    losses: list[float] = []
    order = list(range(len(rows)))
    generator = random.Random(seed)
    for _epoch in range(epochs):
        generator.shuffle(order)
        model.train()
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            selected = [rows[index] for index in order[start : start + batch_size]]
            batch = tensorize_inputs([row.inputs for row in selected], device=device)
            targets = torch.as_tensor(
                np.stack([row.require_target().values.T for row in selected]),
                dtype=torch.float32,
                device=device,
            )
            masks = torch.as_tensor(
                np.stack([row.require_target().mask.T for row in selected]),
                dtype=torch.bool,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = forecast_loss(
                output.quantiles,
                targets,
                model.config.quantiles,
                mask=masks,
                monotone_weight=0.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        losses.append(epoch_loss / max(1, batches))
    return TrainingReport(
        epochs=epochs,
        examples=len(rows),
        first_loss=losses[0],
        final_loss=losses[-1],
        best_loss=min(losses),
    )


def model_identity(model: O1FMForecaster) -> str:
    """Hash config plus exact tensor bytes without pickle or zip metadata."""

    if not isinstance(model, O1FMForecaster):
        raise TypeError("model must be O1FMForecaster")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            asdict(model.config),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name, tensor in sorted(model.state_dict().items()):
        value: Tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def save_checkpoint(model: O1FMForecaster, path: str | Path) -> str:
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(
        {"config": asdict(model.config), "state_dict": model.state_dict()},
        temporary,
    )
    temporary.replace(destination)
    return model_identity(model)


__all__ = [
    "TrainingReport",
    "model_identity",
    "predict",
    "save_checkpoint",
    "tensorize_inputs",
    "train_forecaster",
]
