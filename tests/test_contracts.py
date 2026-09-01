from __future__ import annotations

import numpy as np
import pytest

from o1fm.contracts import (
    ForecastExample,
    ForecastInputs,
    ForecastOutput,
    ForecastPrediction,
    ForecastTarget,
)


def _timeline(length: int) -> np.ndarray:
    start = np.datetime64("2025-01-01T00:00:00", "ns")
    return start + np.arange(length) * np.timedelta64(1, "h")


def _inputs() -> ForecastInputs:
    return ForecastInputs(
        series_id="series-a",
        horizon=2,
        timestamps=_timeline(5),
        past_target=[[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]],
        past_target_mask=[[True, False], [True, True], [True, True]],
        past_covariates=[[0.1], [0.2], [0.3]],
        known_covariates=[
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
    )


def test_inputs_are_canonical_immutable_float32_with_explicit_missingness() -> None:
    inputs = _inputs()

    for value in (
        inputs.past_target,
        inputs.past_covariates,
        inputs.known_covariates,
    ):
        assert value.dtype == np.float32
        assert value.flags.c_contiguous
        assert not value.flags.writeable
        assert np.isfinite(value).all()
    for mask in (
        inputs.past_target_mask,
        inputs.past_covariate_mask,
        inputs.known_covariate_mask,
    ):
        assert mask.dtype == np.bool_
        assert mask.flags.c_contiguous
        assert not mask.flags.writeable
    assert inputs.past_target[0, 1] == 0.0
    assert not inputs.past_target_mask[0, 1]
    assert inputs.history_length == 3
    assert inputs.target_dim == 2
    assert inputs.past_covariate_dim == 1
    assert inputs.known_covariate_dim == 2
    assert inputs.past_timestamps.shape == (3,)
    assert inputs.future_timestamps.shape == (2,)

    with pytest.raises(ValueError):
        inputs.past_target[0, 0] = 9.0


def test_model_mapping_has_no_future_target_channel() -> None:
    inputs = _inputs()
    model_inputs = inputs.model_inputs()

    assert set(model_inputs) == {
        "horizon",
        "known_covariate_mask",
        "known_covariates",
        "past_covariate_mask",
        "past_covariates",
        "past_target",
        "past_target_mask",
        "series_id",
        "timestamps",
    }
    assert not any("future_target" in key for key in model_inputs)
    assert model_inputs["past_covariates"].shape[0] == inputs.history_length
    assert model_inputs["known_covariates"].shape[0] == (
        inputs.history_length + inputs.horizon
    )
    with pytest.raises(TypeError):
        model_inputs["future_target"] = np.ones((2, 2), dtype=np.float32)


def test_future_labels_are_separate_and_exactly_aligned() -> None:
    inputs = _inputs()
    target = ForecastTarget(
        series_id="series-a",
        timestamps=inputs.future_timestamps,
        values=[[6.0, 7.0], [8.0, np.nan]],
        mask=[[True, True], [True, False]],
    )
    example = ForecastExample(inputs=inputs, target=target)

    assert example.require_target() is target
    assert target.values.dtype == np.float32
    assert target.values[1, 1] == 0.0
    assert not target.mask[1, 1]

    with pytest.raises(ValueError, match="series_id"):
        ForecastExample(
            inputs,
            ForecastTarget("series-b", inputs.future_timestamps, target.values),
        )
    with pytest.raises(ValueError, match="horizon"):
        ForecastExample(
            inputs,
            ForecastTarget("series-a", inputs.future_timestamps[:1], [[1.0, 2.0]]),
        )
    shifted = inputs.future_timestamps + np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="timestamps"):
        ForecastExample(
            inputs,
            ForecastTarget("series-a", shifted, target.values, target.mask),
        )
    with pytest.raises(ValueError, match="dimensions"):
        ForecastExample(
            inputs,
            ForecastTarget(
                "series-a",
                inputs.future_timestamps,
                [[1.0], [2.0]],
            ),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"series_id": " bad"}, "series_id"),
        ({"horizon": 0, "timestamps": _timeline(3)}, "horizon"),
        (
            {
                "past_target": [[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]],
                "past_target_mask": None,
            },
            "observed",
        ),
        (
            {
                "past_target": [[1.0, np.inf], [2.0, 3.0], [4.0, 5.0]],
                "past_target_mask": [
                    [True, False],
                    [True, True],
                    [True, True],
                ],
            },
            "infinity",
        ),
        ({"past_covariates": [[1.0], [2.0]]}, "exactly 3"),
        ({"known_covariates": [[1.0]] * 4}, "exactly 5"),
        ({"timestamps": _timeline(5)[[0, 1, 1, 3, 4]]}, "increasing"),
        ({"past_target_mask": [[1, 0], [1, 1], [1, 1]]}, "boolean"),
    ),
)
def test_invalid_inputs_fail_closed(changes: dict[str, object], message: str) -> None:
    options: dict[str, object] = {
        "series_id": "series-a",
        "horizon": 2,
        "timestamps": _timeline(5),
        "past_target": [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]],
        "past_target_mask": None,
        "past_covariates": [[0.1], [0.2], [0.3]],
        "known_covariates": [[0.0, 1.0]] * 5,
    }
    options.update(changes)

    with pytest.raises((TypeError, ValueError), match=message):
        ForecastInputs(**options)


def test_optional_covariates_canonicalize_to_empty_feature_axes() -> None:
    inputs = ForecastInputs(
        series_id="series-empty-covariates",
        horizon=2,
        timestamps=_timeline(5),
        past_target=[[1.0], [2.0], [3.0]],
    )

    assert inputs.past_covariates.shape == (3, 0)
    assert inputs.past_covariate_mask.shape == (3, 0)
    assert inputs.known_covariates.shape == (5, 0)
    assert inputs.known_covariate_mask.shape == (5, 0)
    with pytest.raises(ValueError, match="requires"):
        ForecastInputs(
            series_id="series-empty-covariates",
            horizon=2,
            timestamps=_timeline(5),
            past_target=[[1.0], [2.0], [3.0]],
            past_covariate_mask=np.empty((3, 0), dtype=np.bool_),
        )


def test_prediction_contract_canonicalizes_points_and_ordered_quantiles() -> None:
    inputs = _inputs()
    prediction = ForecastPrediction(
        series_id=inputs.series_id,
        timestamps=inputs.future_timestamps,
        point=[[6.0, 7.0], [8.0, 9.0]],
        quantiles=[
            [[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]],
            [[7.0, 8.0, 9.0], [8.0, 9.0, 10.0]],
        ],
        quantile_levels=[0.1, 0.5, 0.9],
    )

    assert isinstance(prediction, ForecastOutput)
    assert prediction.point.dtype == np.float32
    assert prediction.quantiles.dtype == np.float32
    assert prediction.quantile_levels.dtype == np.float32
    assert prediction.point.shape == (2, 2)
    assert prediction.quantiles.shape == (2, 2, 3)
    assert prediction.horizon == 2
    assert prediction.target_dim == 2
    assert not prediction.point.flags.writeable
    assert not prediction.quantiles.flags.writeable
    prediction.validate_for(inputs)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"quantile_levels": None}, "together"),
        ({"quantile_levels": [0.5, 0.1, 0.9]}, "ordered"),
        ({"quantile_levels": [0.0, 0.5, 0.9]}, "inside"),
        (
            {
                "quantiles": [
                    [[7.0, 6.0, 8.0], [6.0, 7.0, 8.0]],
                    [[7.0, 8.0, 9.0], [8.0, 9.0, 10.0]],
                ]
            },
            "non-decreasing",
        ),
        ({"point": [[6.0, np.nan], [8.0, 9.0]]}, "observed"),
    ),
)
def test_invalid_prediction_contracts_fail_closed(
    changes: dict[str, object],
    message: str,
) -> None:
    options: dict[str, object] = {
        "series_id": "series-a",
        "timestamps": _timeline(2),
        "point": [[6.0, 7.0], [8.0, 9.0]],
        "quantiles": [
            [[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]],
            [[7.0, 8.0, 9.0], [8.0, 9.0, 10.0]],
        ],
        "quantile_levels": [0.1, 0.5, 0.9],
    }
    options.update(changes)
    with pytest.raises(ValueError, match=message):
        ForecastPrediction(**options)


def test_prediction_alignment_rejects_foreign_inputs() -> None:
    inputs = _inputs()
    prediction = ForecastPrediction(
        "series-a",
        inputs.future_timestamps,
        [[1.0, 2.0], [3.0, 4.0]],
    )
    wrong_series = ForecastInputs(
        series_id="series-b",
        horizon=2,
        timestamps=_timeline(5),
        past_target=[[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]],
    )
    with pytest.raises(ValueError, match="series_id"):
        prediction.validate_for(wrong_series)
