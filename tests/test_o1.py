from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from o1fm.o1 import O1Config, O1ConflictError, O1ErrorStream, O1IntegrityError


def _quantiles(offset: float = 0.0) -> dict[float, np.ndarray]:
    return {
        0.1: np.array([0.0, 2.0]) + offset,
        0.5: np.array([1.0, 3.0]) + offset,
        0.9: np.array([2.0, 4.0]) + offset,
    }


def test_o1_tracks_error_pinball_coverage_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "errors.json"
    stream = O1ErrorStream(path, stream_id="series:a")

    update = stream.observe(
        np.array([1.0, 3.0]),
        np.array([0.0, 4.0]),
        quantile_forecasts=_quantiles(),
        observation_id="settlement-1",
    )

    assert update.batch_mae == pytest.approx(1.0)
    assert update.batch_bias == pytest.approx(0.0)
    assert update.interval_coverage == pytest.approx(1.0)
    metrics = stream.metrics()
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["interval_coverage"] == pytest.approx(1.0)
    assert metrics["mean_interval_width"] == pytest.approx(2.0)
    assert metrics["pinball"][(0.1).hex()] == pytest.approx(0.1)
    assert metrics["pinball"][(0.5).hex()] == pytest.approx(0.0)
    assert metrics["pinball"][(0.9).hex()] == pytest.approx(0.1)

    restored = O1ErrorStream(path, stream_id="series:a")
    assert restored.state == stream.state
    assert restored.metrics() == metrics
    assert len(tuple((tmp_path / "errors.json.events").glob("*.json"))) == 1


def test_o1_regime_cusum_and_idempotent_observations(tmp_path: Path) -> None:
    config = O1Config(
        ew_decay=0.5,
        cusum_decay=1.0,
        cusum_drift=0.0,
        cusum_threshold=0.5,
    )
    stream = O1ErrorStream(tmp_path / "state.json", config=config)
    first = stream.observe([0.0], [0.0], scale=1.0, observation_id="one")
    second = stream.observe([10.0], [0.0], scale=1.0, observation_id="two")
    duplicate = stream.observe([10.0], [0.0], scale=1.0, observation_id="two")

    assert not first.regime_changed
    assert second.regime_changed
    assert second.regime_generation == 1
    assert duplicate.duplicate
    assert duplicate.sequence == second.sequence
    assert stream.state.regime_changes == 1
    assert stream.state.cusum == 0.0
    assert len(tuple((tmp_path / "state.json.events").glob("*.json"))) == 2

    with pytest.raises(O1ConflictError):
        stream.observe([11.0], [0.0], scale=1.0, observation_id="two")


def test_o1_rejects_crossed_or_nonfinite_forecasts_without_state_change(
    tmp_path: Path,
) -> None:
    stream = O1ErrorStream(tmp_path / "state.json")
    with pytest.raises(ValueError, match="non-decreasing"):
        stream.observe(
            [1.0, 2.0],
            [1.0, 2.0],
            quantile_forecasts={
                0.1: [2.0, 1.0],
                0.9: [1.0, 3.0],
            },
        )
    with pytest.raises(ValueError, match="finite"):
        stream.observe([1.0], [np.inf])
    assert stream.state.sequence == 0


def test_o1_events_are_authoritative_and_tamper_evident(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    stream = O1ErrorStream(path)
    stream.observe([1.0], [0.0], observation_id="one")
    path.unlink()

    restored = O1ErrorStream(path)
    assert restored.state.sequence == 1
    event = next((tmp_path / "state.json.events").glob("*.json"))
    document = json.loads(event.read_text())
    document["body"]["sequence"] = 2
    event.chmod(0o600)
    event.write_text(json.dumps(document, separators=(",", ":"), sort_keys=True))

    with pytest.raises(O1IntegrityError):
        O1ErrorStream(path)
