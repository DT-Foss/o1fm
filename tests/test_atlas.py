from __future__ import annotations

import builtins
import importlib.util
import json
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from o1fm.action_bank import ActionProfileKey
from o1fm.atlas import (
    AtlasReceipt,
    ForecastAtlas,
    ForecastAtlasIntegrityError,
    ForecastAtlasValidationError,
    SegmentStore,
)
from o1fm.causal_graph import CausalEdge
from o1fm.contracts import ForecastInputs, ForecastPrediction
from o1fm.identity import canonical_json_bytes, sha256_bytes, sha256_json
from o1fm.metrics import ForecastScore
from o1fm.runtime import RouteSelection


class FakeStore:
    def __init__(self) -> None:
        self.records: dict[str, tuple[dict[str, Any], ...]] = {}
        self.sequence = 0
        self.head = "0" * 64
        self.valid = True

    def append_segment(self, records: Iterable[Mapping[str, Any]]) -> str:
        materialized = tuple(
            json.loads(canonical_json_bytes(record)) for record in records
        )
        body = b"".join(canonical_json_bytes(record) + b"\n" for record in materialized)
        sha = sha256_bytes(body)
        if sha not in self.records:
            self.records[sha] = materialized
            self.sequence += 1
            self.head = sha256_json(
                {
                    "previous": self.head,
                    "segment": sha,
                    "sequence": self.sequence,
                }
            )
        return sha

    def revision(self) -> tuple[int, str]:
        return self.sequence, self.head

    def record(self, sha: str, index: int) -> Mapping[str, Any]:
        return self.records[sha][index]

    def verify(self) -> bool:
        return self.valid


class RecordObject:
    def __init__(self, document: Mapping[str, object], **attributes: object) -> None:
        self.document = dict(document)
        for name, value in attributes.items():
            setattr(self, name, value)

    def to_record(self) -> dict[str, object]:
        return dict(self.document)


def _ticket() -> SimpleNamespace:
    history = 8
    horizon = 3
    timeline = np.datetime64("2026-01-01") + np.arange(history + horizon).astype(
        "timedelta64[h]"
    )
    inputs = ForecastInputs(
        series_id="atlas-series",
        horizon=horizon,
        timestamps=timeline,
        past_target=np.arange(history, dtype=np.float32)[:, None],
        past_covariates=np.ones((history, 1), dtype=np.float32),
        known_covariates=np.ones((history + horizon, 1), dtype=np.float32),
    )
    point = np.full((horizon, 1), 4.0, dtype=np.float32)
    offsets = np.asarray((-1.0, 0.0, 1.0), dtype=np.float32)
    prediction = ForecastPrediction(
        series_id=inputs.series_id,
        timestamps=inputs.future_timestamps,
        point=point,
        quantiles=point[..., None] + offsets,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    edge = CausalEdge(
        source="known:0",
        target="target:0",
        lag=1,
        kind="intervention",
        observations=12,
        covariance=0.7,
        source_variance=1.0,
        target_variance=1.0,
        score=0.7,
        revision_sha256="4" * 64,
    )
    route_signature = sha256_json(
        {
            "edges": [
                {
                    "kind": edge.kind,
                    "lag": edge.lag,
                    "source": edge.source,
                    "target": edge.target,
                }
            ],
            "known": [0],
            "past": [0],
            "schema": "o1fm.routed-feature-topology/v1",
        }
    )
    profile = ActionProfileKey(
        model_identity="1" * 64,
        feature_schema="2" * 64,
        graph_revision=route_signature,
        regime="regime:7",
        horizon=horizon,
        target_count=1,
    )
    route = RouteSelection(
        past_indices=(0,),
        known_indices=(0,),
        signature_sha256=profile.graph_revision,
        edges=(edge,),
        transport_w1=0.625,
    )
    calibration_key = RecordObject(
        {
            "action": "crystal",
            "graph_revision": profile.graph_revision,
            "model_identity": profile.model_identity,
            "schema": "test.calibration-key/v1",
        }
    )
    calibration = SimpleNamespace(
        key=calibration_key,
        base_prediction=prediction,
        prediction=prediction,
        lower_alpha=0.1,
        upper_alpha=0.1,
        joint_alpha=0.2,
        prior_settlements=9,
    )
    return SimpleNamespace(
        ticket_id="5" * 64,
        inputs=inputs,
        prediction=prediction,
        base_prediction=prediction,
        calibration=calibration,
        action="crystal",
        profile=profile,
        route=route,
        regime=profile.regime,
        forward_cost=0.0,
        foundation_forwards=0,
        local_forwards=0,
        forced_foundation_refresh=False,
    )


def _settlement(ticket: SimpleNamespace) -> SimpleNamespace:
    score = ForecastScore(
        mae=0.25,
        rmse=0.3,
        pinball=0.12,
        interval_coverage=0.8,
        interval_width=2.0,
        observed_values=3,
    )
    base_score = ForecastScore(
        mae=0.35,
        rmse=0.4,
        pinball=0.16,
        interval_coverage=0.6,
        interval_width=1.5,
        observed_values=3,
    )
    o1_document = {
        "observation_sha256": "6" * 64,
        "regime_generation": 7,
        "sequence": 19,
        "surprise": 1.25,
    }
    o1 = RecordObject(
        o1_document,
        observation_sha256="6" * 64,
        sequence=19,
        regime_changed=True,
        regime_generation=7,
        surprise=1.25,
        cusum=9.5,
    )
    conformal = SimpleNamespace(
        content_sha256="7" * 64,
        observed_values=3,
        joint_hit=True,
        lower_alpha=0.08,
        upper_alpha=0.12,
        joint_alpha=0.18,
    )
    return SimpleNamespace(
        ticket_id=ticket.ticket_id,
        action=ticket.action,
        score=score,
        base_score=base_score,
        o1_update=o1,
        calibration_update=conformal,
        joint_covered=True,
        crystal_ready=True,
        forward_cost=0.0,
        foundation_forwards=0,
        local_forwards=0,
        forced_foundation_refresh=False,
        saved_foundation_forwards=1,
    )


def test_segment_store_protocol_is_runtime_checkable() -> None:
    assert isinstance(FakeStore(), SegmentStore)


def test_forecast_append_is_deterministic_idempotent_and_directly_citable() -> None:
    store = FakeStore()
    atlas = ForecastAtlas(store)
    ticket = _ticket()

    first = atlas.append_forecast(ticket)
    second = atlas.append_forecast(ticket)

    assert first == second
    assert first.revision_sequence == 1
    assert atlas.verify(first)
    record = atlas.record(first)
    assert record["kind"] == "forecast"
    assert record["schema"] == "o1fm.forecast-atlas.forecast/v1"
    assert record["trigger_key"].startswith("o1fm:forecast-origin:atlas-series:")
    assert record["outcome_key"].startswith("o1fm:prediction:")
    assert record["model_identity"] == ticket.profile.model_identity
    assert record["route"]["transport_w1"] == 0.625
    assert record["graph"]["head_sha256"] == "4" * 64
    assert record["graph"]["topology_sha256"] == ticket.route.signature_sha256
    assert record["calibration"]["identity_sha256"]
    assert record["timestamps"]["values_ns"]
    assert "past_target" not in record
    assert "point" not in record["prediction"]


def test_settlement_captures_scores_o1_conformal_and_forward_economics() -> None:
    atlas = ForecastAtlas(FakeStore())
    ticket = _ticket()
    receipt = atlas.append_settlement(ticket, _settlement(ticket))
    record = atlas.record(receipt)

    assert record["kind"] == "settlement"
    assert record["scores"]["selected"]["mae"] == 0.25
    assert record["scores"]["base"]["pinball"] == 0.16
    assert record["o1_update"]["observation_sha256"] == "6" * 64
    assert record["o1_update"]["regime_generation"] == 7
    assert record["o1_update"]["surprise"] == 1.25
    assert record["conformal_update"]["content_sha256"] == "7" * 64
    assert record["conformal_update"]["joint_hit"] is True
    assert record["conformal_update"]["lower_alpha"] == 0.08
    assert record["costs"]["saved_foundation_forwards"] == 1
    assert record["costs"]["foundation_forwards"] == 0
    assert "target" not in record


def test_transport_append_has_exact_profile_route_and_w1_identity() -> None:
    atlas = ForecastAtlas(FakeStore())
    ticket = _ticket()
    receipt = atlas.append_transport(ticket)
    record = atlas.record(receipt)

    assert receipt.kind == "transport"
    assert record["trigger_key"].startswith("o1fm:profile:")
    assert record["outcome_key"].startswith("o1fm:transport:")
    assert record["route"]["transport_w1"] == ticket.route.transport_w1
    assert record["transport_sha256"]


def test_verify_failure_is_visible_and_blocks_new_receipt() -> None:
    store = FakeStore()
    atlas = ForecastAtlas(store)
    receipt = atlas.append_forecast(_ticket())
    store.valid = False

    assert atlas.verify() is False
    assert atlas.verify(receipt) is False
    with pytest.raises(
        ForecastAtlasIntegrityError,
        match="store verification failed",
    ):
        atlas.append_forecast(_ticket())


def test_receipt_validation_and_citation_bounds_are_strict() -> None:
    with pytest.raises(ForecastAtlasValidationError):
        AtlasReceipt("bad", 0, "0" * 64, 1, "forecast")
    with pytest.raises(ForecastAtlasValidationError):
        AtlasReceipt("0" * 64, 0, "0" * 64, 0, "forecast")
    with pytest.raises(ForecastAtlasValidationError):
        AtlasReceipt("0" * 64, 0, "0" * 64, 1, "unknown")

    atlas = ForecastAtlas(FakeStore())
    receipt = atlas.append_forecast(_ticket())
    with pytest.raises(ForecastAtlasValidationError):
        atlas.record(receipt, 1)


def test_from_immer_is_a_lazy_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "immer.knowledge.livecausal":
            raise ImportError("blocked for standalone test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="requires IMMER"):
        ForecastAtlas.from_immer("unused")


@pytest.mark.skipif(
    importlib.util.find_spec("immer") is None,
    reason="IMMER is an optional sibling package",
)
def test_conditional_real_livestore_round_trip(tmp_path) -> None:
    atlas = ForecastAtlas.from_immer(tmp_path / "atlas.causal")
    receipt = atlas.append_forecast(_ticket())

    assert atlas.verify(receipt)
    assert atlas.record(receipt)["ticket_id"] == "5" * 64
