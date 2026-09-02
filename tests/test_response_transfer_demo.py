from __future__ import annotations

from pathlib import Path

import numpy as np

from o1fm.cli import build_parser
from o1fm.contracts import ForecastInputs, ForecastPrediction
from o1fm.response_transfer_demo import run_response_transfer_demo
from o1fm.teachers import ForecastTeacherCapabilities


class _Teacher:
    quantile_levels = (0.1, 0.5, 0.9)
    identity = "9" * 64
    checkpoint = "fake:transfer"
    capabilities = ForecastTeacherCapabilities(True, True, True, True)

    def predict(self, inputs: ForecastInputs) -> ForecastPrediction:
        last = np.asarray(inputs.past_target[-1], dtype=np.float32)
        point = np.repeat(last[None, :], inputs.horizon, axis=0)
        quantiles = point[..., None] + np.asarray((-1.0, 0.0, 1.0), np.float32)
        return ForecastPrediction(
            series_id=inputs.series_id,
            timestamps=inputs.future_timestamps,
            point=point,
            quantiles=quantiles,
            quantile_levels=self.quantile_levels,
        )


def test_transfer_demo_cold_falls_back_then_releases_parent_surface(
    tmp_path: Path,
) -> None:
    result = run_response_transfer_demo(
        teacher=_Teacher(),
        state_dir=tmp_path / "state",
        seed=41,
    )

    assert result["cartographer"]["candidate_count"] == 60
    assert result["cartographer"]["selected_count"] == 48
    assert result["cartographer"]["design_rank"] == 48
    identities = result["identities"]
    assert identities["r0_topology_sha256"] != identities["r1_topology_sha256"]
    assert identities["r0_exact_key_sha256"] != identities["r1_exact_key_sha256"]
    assert identities["shared_inner_crystal"] is True
    assert result["domains"]["r1_cold_audits"]["actions"] == {
        "foundation": 9,
        "crystal": 0,
    }
    assert result["domains"]["r1_after_audits"]["actions"] == {
        "foundation": 0,
        "crystal": 12,
    }
    economics = result["economics"]
    assert (
        economics["selected_serving_forwards"] < economics["baseline_serving_forwards"]
    )


def test_transfer_cli_defaults() -> None:
    arguments = build_parser().parse_args(
        ("transfer-demo", "--atlas-dir", "atlas.causal")
    )
    assert arguments.seed == 41
    assert arguments.device == "cpu"
