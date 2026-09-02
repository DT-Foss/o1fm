"""Optional forecasting-foundation teachers behind O1FM's strict contracts.

The module itself only depends on O1FM's core NumPy/PyTorch requirements.
Toto-2 and Chronos-2 are imported when their ``from_pretrained`` constructors
are called, so native O1FM installations never acquire either dependency.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .contracts import ForecastInputs, ForecastPrediction
from .identity import sha256_json
from .model import DEFAULT_QUANTILES, O1FMForecaster
from .training import model_identity
from .training import predict as native_predict

_NATIVE_ADAPTER = "o1fm.native/v1"
_TOTO2_ADAPTER = "o1fm.toto2/v1"
_CHRONOS2_ADAPTER = "o1fm.chronos2/v1"


def _levels(value: Sequence[float]) -> tuple[float, ...]:
    try:
        levels = tuple(float(level) for level in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("quantile levels must be numeric") from exc
    if (
        not levels
        or not all(np.isfinite(level) and 0.0 < level < 1.0 for level in levels)
        or any(right <= left for left, right in zip(levels, levels[1:], strict=False))
        or 0.5 not in levels
    ):
        raise ValueError(
            "quantile levels must be finite, strictly increasing, inside (0, 1), "
            "and include 0.5"
        )
    return levels


def _identifier(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def teacher_identity(
    adapter: str,
    checkpoint: str,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
) -> str:
    """Return the canonical identity of a teacher execution contract."""

    return sha256_json(
        {
            "adapter": _identifier(adapter, field="adapter"),
            "checkpoint": _identifier(checkpoint, field="checkpoint"),
            "quantile_levels": list(_levels(quantile_levels)),
            "schema": "o1fm.teacher-identity/v1",
        }
    )


@runtime_checkable
class ForecastTeacher(Protocol):
    """Common inference surface for native and external forecast teachers."""

    quantile_levels: tuple[float, ...]

    def predict(self, inputs: ForecastInputs) -> ForecastPrediction:
        """Forecast one strict O1FM input contract."""


def _numpy(value: object, *, field: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not an array") from exc
    if np.iscomplexobj(result):
        raise ValueError(f"{field} cannot contain complex values")
    return result


def _single(value: object) -> object:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _prediction(
    inputs: ForecastInputs,
    *,
    quantiles: object,
    levels: tuple[float, ...],
    point: object | None = None,
) -> ForecastPrediction:
    quantile_array = np.asarray(quantiles, dtype=np.float32)
    if point is None:
        point_array = quantile_array[..., levels.index(0.5)]
    else:
        point_array = np.asarray(point, dtype=np.float32)
    result = ForecastPrediction(
        series_id=inputs.series_id,
        timestamps=inputs.future_timestamps,
        point=point_array,
        quantiles=quantile_array,
        quantile_levels=levels,
    )
    result.validate_for(inputs)
    return result


class NativeO1FMTeacher:
    """Use an ``O1FMForecaster`` through the same teacher protocol."""

    adapter = _NATIVE_ADAPTER

    def __init__(
        self,
        model: O1FMForecaster,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(model, O1FMForecaster):
            raise TypeError("model must be O1FMForecaster")
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.quantile_levels = _levels(model.config.quantiles)
        self.checkpoint = f"sha256:{model_identity(model)}"

    @property
    def identity(self) -> str:
        return teacher_identity(self.adapter, self.checkpoint, self.quantile_levels)

    def predict(self, inputs: ForecastInputs) -> ForecastPrediction:
        if not isinstance(inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        return native_predict(self.model, inputs, device=self.device)


class Toto2Teacher:
    """Adapter for Datadog Toto-2's tensor forecasting API."""

    adapter = _TOTO2_ADAPTER

    def __init__(
        self,
        model: object,
        *,
        checkpoint: str = "injected:Toto2Model",
        device: str | torch.device = "cpu",
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    ) -> None:
        if not callable(getattr(model, "forecast", None)):
            raise TypeError("model must expose forecast(inputs, horizon=...)")
        self.model = model
        self.checkpoint = _identifier(checkpoint, field="checkpoint")
        self.device = torch.device(device)
        self.quantile_levels = _levels(quantile_levels)
        knots = getattr(getattr(model, "output_head", None), "knots", None)
        if knots is not None and _levels(knots) != self.quantile_levels:
            raise ValueError("Toto-2 output knots differ from the adapter inventory")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "Datadog/Toto-2.0-22m",
        device: str | torch.device = "cpu",
    ) -> Toto2Teacher:
        checkpoint = _identifier(checkpoint, field="checkpoint")
        try:
            module = importlib.import_module("toto2")
            model_class = module.Toto2Model
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "Toto2Teacher requires the optional `toto-2` package"
            ) from exc
        model = model_class.from_pretrained(
            checkpoint,
            map_location=torch.device(device),
        )
        model = model.to(device).eval()
        return cls(model, checkpoint=checkpoint, device=device)

    @property
    def identity(self) -> str:
        return teacher_identity(self.adapter, self.checkpoint, self.quantile_levels)

    def _payload(self, inputs: ForecastInputs) -> dict[str, torch.Tensor]:
        historical = np.concatenate(
            (inputs.past_target, inputs.past_covariates), axis=1
        )
        historical_mask = np.concatenate(
            (inputs.past_target_mask, inputs.past_covariate_mask), axis=1
        )
        variates = historical.shape[1]
        payload = {
            "target": torch.as_tensor(
                np.array(historical.T, copy=True),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "target_mask": torch.as_tensor(
                np.array(historical_mask.T, copy=True),
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0),
            "series_ids": torch.zeros(
                (1, variates), dtype=torch.long, device=self.device
            ),
        }
        if inputs.known_covariate_dim:
            payload.update(
                {
                    "known_dynamic": torch.as_tensor(
                        np.array(inputs.known_covariates.T, copy=True),
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0),
                    "known_dynamic_mask": torch.as_tensor(
                        np.array(inputs.known_covariate_mask.T, copy=True),
                        dtype=torch.bool,
                        device=self.device,
                    ).unsqueeze(0),
                    "known_dynamic_series_ids": torch.zeros(
                        (1, inputs.known_covariate_dim),
                        dtype=torch.long,
                        device=self.device,
                    ),
                }
            )
        return payload

    def predict(self, inputs: ForecastInputs) -> ForecastPrediction:
        if not isinstance(inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        payload = self._payload(inputs)
        has_missing = not all(
            bool(mask.all())
            for mask in (
                inputs.past_target_mask,
                inputs.past_covariate_mask,
                inputs.known_covariate_mask,
            )
        )
        with torch.no_grad():
            raw = self.model.forecast(
                payload,
                horizon=inputs.horizon,
                decode_block_size=None,
                has_missing_values=has_missing,
            )
        raw = getattr(raw, "quantiles", raw)
        array = _numpy(raw, field="Toto-2 quantiles")
        expected = (
            len(self.quantile_levels),
            1,
            inputs.target_dim + inputs.past_covariate_dim,
            inputs.horizon,
        )
        if array.shape != expected:
            raise ValueError(
                f"Toto-2 quantiles must have shape {expected}, got {array.shape}"
            )
        quantiles = array[:, 0, : inputs.target_dim, :].transpose(2, 1, 0)
        return _prediction(
            inputs,
            quantiles=quantiles,
            levels=self.quantile_levels,
        )


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float32, order="C", copy=True)
    result[~mask] = np.nan
    return result


def _chronos_payload(inputs: ForecastInputs) -> dict[str, object]:
    past_covariates: dict[str, np.ndarray] = {}
    future_covariates: dict[str, np.ndarray] = {}
    for index in range(inputs.past_covariate_dim):
        past_covariates[f"past_covariate_{index}"] = _masked(
            inputs.past_covariates[:, index],
            inputs.past_covariate_mask[:, index],
        )
    for index in range(inputs.known_covariate_dim):
        name = f"known_covariate_{index}"
        past_covariates[name] = _masked(
            inputs.known_covariates[: inputs.history_length, index],
            inputs.known_covariate_mask[: inputs.history_length, index],
        )
        future_covariates[name] = _masked(
            inputs.known_covariates[inputs.history_length :, index],
            inputs.known_covariate_mask[inputs.history_length :, index],
        )
    payload: dict[str, object] = {
        "target": _masked(inputs.past_target, inputs.past_target_mask).T,
    }
    if past_covariates:
        payload["past_covariates"] = past_covariates
    if future_covariates:
        payload["future_covariates"] = future_covariates
    return payload


def _chronos_quantiles(
    value: object,
    *,
    inputs: ForecastInputs,
    quantile_count: int,
) -> np.ndarray:
    array = _numpy(_single(value), field="Chronos-2 quantiles")
    target = inputs.target_dim
    horizon = inputs.horizon
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (target, horizon, quantile_count):
        return array.transpose(1, 0, 2)
    if array.shape == (horizon, target, quantile_count):
        return array
    if array.shape == (quantile_count, target, horizon):
        return array.transpose(2, 1, 0)
    if array.shape == (quantile_count, horizon, target):
        return array.transpose(1, 2, 0)
    raise ValueError(
        "Chronos-2 quantiles have unsupported shape "
        f"{array.shape}; expected target={target}, horizon={horizon}, "
        f"quantiles={quantile_count}"
    )


def _chronos_point(value: object, *, inputs: ForecastInputs) -> np.ndarray:
    array = _numpy(_single(value), field="Chronos-2 point forecast")
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (inputs.target_dim, inputs.horizon):
        return array.T
    if array.shape == (inputs.horizon, inputs.target_dim):
        return array
    raise ValueError(
        "Chronos-2 point forecast has unsupported shape "
        f"{array.shape}; expected target={inputs.target_dim}, "
        f"horizon={inputs.horizon}"
    )


def _frame_column(frame: object, name: object) -> np.ndarray:
    try:
        value = frame[name]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Chronos-2 dataframe lacks column {name!r}") from exc
    if callable(getattr(value, "to_numpy", None)):
        value = value.to_numpy()
    return np.asarray(value)


def _chronos_frame(
    frame: object,
    *,
    inputs: ForecastInputs,
    levels: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray | None]:
    raw_columns = tuple(getattr(frame, "columns", ()))
    if not raw_columns:
        raise ValueError("Chronos-2 dataframe has no columns")
    columns = {str(column): column for column in raw_columns}
    point_column = next(
        (columns[name] for name in ("predictions", "point", "mean") if name in columns),
        None,
    )
    quantile_columns: list[object] = []
    for level in levels:
        key = str(level)
        if key not in columns:
            raise ValueError(f"Chronos-2 dataframe lacks quantile column {key}")
        quantile_columns.append(columns[key])

    if "target_name" in columns:
        labels = _frame_column(frame, columns["target_name"])
        unique = list(dict.fromkeys(labels.tolist()))
        expected = [f"target_{index}" for index in range(inputs.target_dim)]
        if not all(name in unique for name in expected):
            if len(unique) != inputs.target_dim:
                raise ValueError("Chronos-2 dataframe target inventory differs")
            expected = unique
        selections = [np.flatnonzero(labels == name) for name in expected]
    else:
        rows = len(_frame_column(frame, quantile_columns[0]))
        if rows != inputs.target_dim * inputs.horizon:
            raise ValueError("Chronos-2 dataframe row count differs")
        selections = [
            np.arange(index * inputs.horizon, (index + 1) * inputs.horizon)
            for index in range(inputs.target_dim)
        ]

    timestamps = (
        _frame_column(frame, columns["timestamp"])
        if "timestamp" in columns
        else None
    )
    quantiles = np.empty(
        (inputs.horizon, inputs.target_dim, len(levels)), dtype=np.float32
    )
    point = (
        np.empty((inputs.horizon, inputs.target_dim), dtype=np.float32)
        if point_column is not None
        else None
    )
    for target_index, selected in enumerate(selections):
        if selected.shape != (inputs.horizon,):
            raise ValueError("Chronos-2 dataframe horizon differs")
        if timestamps is not None:
            selected = selected[np.argsort(timestamps[selected])]
        for level_index, column in enumerate(quantile_columns):
            quantiles[:, target_index, level_index] = _frame_column(
                frame, column
            )[selected]
        if point is not None:
            point[:, target_index] = _frame_column(frame, point_column)[selected]
    return quantiles, point


class Chronos2Teacher:
    """Adapter for Amazon Chronos-2's multivariate list-of-dicts API."""

    adapter = _CHRONOS2_ADAPTER

    def __init__(
        self,
        pipeline: object,
        *,
        checkpoint: str = "injected:Chronos2Pipeline",
        device: str | torch.device = "cpu",
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    ) -> None:
        if not callable(getattr(pipeline, "predict_quantiles", None)):
            raise TypeError("pipeline must expose predict_quantiles(inputs, ...)")
        self.pipeline = pipeline
        self.checkpoint = _identifier(checkpoint, field="checkpoint")
        self.device = torch.device(device)
        self.quantile_levels = _levels(quantile_levels)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "amazon/chronos-2",
        device: str | torch.device = "cpu",
    ) -> Chronos2Teacher:
        checkpoint = _identifier(checkpoint, field="checkpoint")
        try:
            module = importlib.import_module("chronos")
            pipeline_class = module.Chronos2Pipeline
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "Chronos2Teacher requires the optional `chronos-forecasting` package"
            ) from exc
        pipeline = pipeline_class.from_pretrained(
            checkpoint,
            device_map=str(torch.device(device)),
        )
        return cls(pipeline, checkpoint=checkpoint, device=device)

    @property
    def identity(self) -> str:
        return teacher_identity(self.adapter, self.checkpoint, self.quantile_levels)

    def predict(self, inputs: ForecastInputs) -> ForecastPrediction:
        if not isinstance(inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        raw = self.pipeline.predict_quantiles(
            [_chronos_payload(inputs)],
            prediction_length=inputs.horizon,
            quantile_levels=list(self.quantile_levels),
        )
        if getattr(raw, "columns", None) is not None:
            quantiles, point = _chronos_frame(
                raw,
                inputs=inputs,
                levels=self.quantile_levels,
            )
        else:
            if isinstance(raw, tuple) and len(raw) == 2:
                raw_quantiles, raw_point = raw
            else:
                raw_quantiles, raw_point = raw, None
            quantiles = _chronos_quantiles(
                raw_quantiles,
                inputs=inputs,
                quantile_count=len(self.quantile_levels),
            )
            point = (
                None
                if raw_point is None
                else _chronos_point(raw_point, inputs=inputs)
            )
        return _prediction(
            inputs,
            quantiles=quantiles,
            point=point,
            levels=self.quantile_levels,
        )


__all__ = [
    "Chronos2Teacher",
    "ForecastTeacher",
    "NativeO1FMTeacher",
    "Toto2Teacher",
    "teacher_identity",
]
