from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from o1fm.contracts import ForecastInputs
from o1fm.model import DEFAULT_QUANTILES, O1FMConfig, O1FMForecaster
from o1fm.open_teacher_demo import _module_sha256
from o1fm.teachers import (
    Chronos2Teacher,
    ForecastTeacher,
    NativeO1FMTeacher,
    Toto2Teacher,
    teacher_identity,
)


def _inputs() -> ForecastInputs:
    target = np.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        dtype=np.float32,
    )
    target_mask = np.ones_like(target, dtype=np.bool_)
    target_mask[1, 1] = False
    past = np.array([[5.0], [6.0], [7.0], [8.0]], dtype=np.float32)
    past_mask = np.ones_like(past, dtype=np.bool_)
    past_mask[2, 0] = False
    known = np.arange(7, dtype=np.float32).reshape(7, 1)
    known_mask = np.ones_like(known, dtype=np.bool_)
    known_mask[5, 0] = False
    return ForecastInputs(
        series_id="series-a",
        horizon=3,
        timestamps=np.arange(
            np.datetime64("2026-01-01"),
            np.datetime64("2026-01-08"),
            dtype="datetime64[D]",
        ),
        past_target=target,
        past_target_mask=target_mask,
        past_covariates=past,
        past_covariate_mask=past_mask,
        known_covariates=known,
        known_covariate_mask=known_mask,
    )


def _quantiles_toto(variates: int, horizon: int) -> torch.Tensor:
    result = torch.empty((9, 1, variates, horizon), dtype=torch.float32)
    for level in range(9):
        for variate in range(variates):
            result[level, 0, variate] = (
                torch.arange(horizon, dtype=torch.float32)
                + 10.0 * variate
                + 0.1 * level
            )
    return result


class _FakeToto:
    output_head = SimpleNamespace(knots=list(DEFAULT_QUANTILES))

    def __init__(self) -> None:
        self.call: tuple[dict[str, torch.Tensor], int, dict[str, object]] | None = None

    def forecast(
        self,
        inputs: dict[str, torch.Tensor],
        horizon: int,
        **kwargs: object,
    ) -> torch.Tensor:
        self.call = (inputs, horizon, kwargs)
        return _quantiles_toto(inputs["target"].shape[1], horizon)


def test_toto_maps_masks_covariates_and_quantiles() -> None:
    inputs = _inputs()
    model = _FakeToto()
    teacher = Toto2Teacher(model)

    prediction = teacher.predict(inputs)

    assert isinstance(teacher, ForecastTeacher)
    assert model.call is not None
    payload, horizon, kwargs = model.call
    assert horizon == 3
    assert kwargs == {"decode_block_size": None, "has_missing_values": True}
    assert payload["target"].shape == (1, 3, 4)
    np.testing.assert_array_equal(
        payload["target"][0].cpu().numpy(),
        np.concatenate((inputs.past_target, inputs.past_covariates), axis=1).T,
    )
    np.testing.assert_array_equal(
        payload["target_mask"][0].cpu().numpy(),
        np.concatenate((inputs.past_target_mask, inputs.past_covariate_mask), axis=1).T,
    )
    assert payload["series_ids"].tolist() == [[0, 0, 0]]
    assert payload["known_dynamic"].shape == (1, 1, 7)
    np.testing.assert_array_equal(
        payload["known_dynamic_mask"][0, 0].cpu().numpy(),
        inputs.known_covariate_mask[:, 0],
    )
    assert payload["known_dynamic_series_ids"].tolist() == [[0]]
    assert prediction.quantiles is not None
    assert prediction.quantiles.shape == (3, 2, 9)
    np.testing.assert_allclose(prediction.point, prediction.quantiles[..., 4])
    assert not prediction.point.flags.writeable
    assert not prediction.quantiles.flags.writeable


def test_toto_rejects_crossed_or_wrong_shaped_output() -> None:
    class CrossingToto(_FakeToto):
        def forecast(self, inputs, horizon, **kwargs):  # type: ignore[no-untyped-def]
            result = super().forecast(inputs, horizon, **kwargs)
            result[5] = result[4] - 1.0
            return result

    with pytest.raises(ValueError, match="non-decreasing"):
        Toto2Teacher(CrossingToto()).predict(_inputs())

    class WrongShapeToto(_FakeToto):
        def forecast(self, inputs, horizon, **kwargs):  # type: ignore[no-untyped-def]
            return torch.zeros(9, 1, 2, horizon)

    with pytest.raises(ValueError, match="must have shape"):
        Toto2Teacher(WrongShapeToto()).predict(_inputs())


def test_toto_loader_uses_safetensors_compatible_device_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoadableToto(_FakeToto):
        @classmethod
        def from_pretrained(cls, checkpoint: str, *, map_location: str):
            assert checkpoint == "Datadog/Toto-2.0-22m"
            assert map_location == "cpu"
            return cls()

        def to(self, device):  # type: ignore[no-untyped-def]
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(Toto2Model=LoadableToto),
    )
    teacher = Toto2Teacher.from_pretrained(device="cpu")
    assert isinstance(teacher, Toto2Teacher)


class _FakeChronos:
    def __init__(self) -> None:
        self.call: tuple[list[dict[str, object]], int, list[float]] | None = None

    def predict_quantiles(
        self,
        inputs: list[dict[str, object]],
        *,
        prediction_length: int,
        quantile_levels: list[float],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self.call = (inputs, prediction_length, quantile_levels)
        target_dim = np.asarray(inputs[0]["target"]).shape[0]
        quantiles = torch.empty(
            (target_dim, prediction_length, len(quantile_levels)),
            dtype=torch.float32,
        )
        for target in range(target_dim):
            for level in range(len(quantile_levels)):
                quantiles[target, :, level] = (
                    torch.arange(prediction_length, dtype=torch.float32)
                    + 10.0 * target
                    + 0.1 * level
                )
        median = quantiles[..., quantile_levels.index(0.5)]
        return [quantiles], [median]


def test_chronos_uses_list_dict_contract_and_preserves_masks() -> None:
    inputs = _inputs()
    pipeline = _FakeChronos()
    teacher = Chronos2Teacher(pipeline)

    prediction = teacher.predict(inputs)

    assert isinstance(teacher, ForecastTeacher)
    assert pipeline.call is not None
    records, horizon, levels = pipeline.call
    assert horizon == 3
    assert levels == list(DEFAULT_QUANTILES)
    assert len(records) == 1
    record = records[0]
    target = np.asarray(record["target"])
    assert target.shape == (2, 4)
    assert np.isnan(target[1, 1])
    past = record["past_covariates"]
    future = record["future_covariates"]
    assert isinstance(past, dict)
    assert isinstance(future, dict)
    assert set(past) == {"past_covariate_0", "known_covariate_0"}
    assert set(future) == {"known_covariate_0"}
    assert np.isnan(past["past_covariate_0"][2])
    assert np.isnan(future["known_covariate_0"][1])
    assert prediction.quantiles is not None
    assert prediction.quantiles.shape == (3, 2, 9)
    np.testing.assert_allclose(prediction.point, prediction.quantiles[..., 4])


class _MiniFrame:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self._values = values
        self.columns = tuple(values)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._values[key]


def test_chronos_normalizes_dataframe_output() -> None:
    class FrameChronos:
        def predict_quantiles(self, inputs, **kwargs):  # type: ignore[no-untyped-def]
            levels = kwargs["quantile_levels"]
            target_name = np.repeat(["target_1", "target_0"], 3)
            timestamp = np.tile(np.array([2, 0, 1]), 2)
            values: dict[str, np.ndarray] = {
                "target_name": target_name,
                "timestamp": timestamp,
                "predictions": np.array([12, 10, 11, 2, 0, 1], dtype=np.float32),
            }
            for index, level in enumerate(levels):
                values[str(level)] = values["predictions"] + (index - 4) * 0.1
            return _MiniFrame(values)

    result = Chronos2Teacher(FrameChronos()).predict(_inputs())
    np.testing.assert_allclose(result.point[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.point[:, 1], [10.0, 11.0, 12.0])
    assert result.quantiles is not None
    np.testing.assert_allclose(result.point, result.quantiles[..., 4])


def test_native_teacher_and_identity_contract() -> None:
    torch.manual_seed(3)
    model = O1FMForecaster(
        O1FMConfig(patch_len=2, d_model=8, num_heads=2, num_layers=1)
    )
    teacher = NativeO1FMTeacher(model)
    prediction = teacher.predict(_inputs())
    assert prediction.point.shape == (3, 2)
    assert isinstance(teacher, ForecastTeacher)
    assert len(teacher.identity) == 64

    first = teacher_identity("adapter/v1", "org/model", DEFAULT_QUANTILES)
    second = teacher_identity("adapter/v1", "org/model", tuple(DEFAULT_QUANTILES))
    assert first == second
    assert first != teacher_identity("adapter/v2", "org/model", DEFAULT_QUANTILES)
    assert first != teacher_identity("adapter/v1", "org/model-2", DEFAULT_QUANTILES)


def test_weight_identity_handles_scalar_and_matrix_state() -> None:
    module = torch.nn.Module()
    module.register_parameter("scalar", torch.nn.Parameter(torch.tensor(1.0)))
    module.register_buffer("matrix", torch.arange(6, dtype=torch.float32).reshape(2, 3))
    first = _module_sha256(module)
    assert first is not None and len(first) == 64
    module.scalar.data.add_(1.0)
    assert _module_sha256(module) != first


@pytest.mark.parametrize(
    ("module_name", "constructor", "message"),
    [
        ("toto2", Toto2Teacher.from_pretrained, "toto-2"),
        ("chronos", Chronos2Teacher.from_pretrained, "chronos-forecasting"),
    ],
)
def test_optional_dependencies_are_lazy_and_actionable(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    constructor,
    message: str,
) -> None:
    real_import = importlib.import_module

    def missing(name: str, package: str | None = None):
        if name == module_name:
            raise ModuleNotFoundError(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ImportError, match=message):
        constructor()
