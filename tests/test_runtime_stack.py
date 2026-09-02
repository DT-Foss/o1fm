from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from o1fm.action_bank import ActionBank
from o1fm.atlas import ForecastAtlas
from o1fm.baselines import SeasonalNaiveForecaster
from o1fm.causal_graph import CausalEdge, CausalGraphConfig, OnlineCausalGraph
from o1fm.conformal import O1ConformalField
from o1fm.contracts import ForecastInputs, ForecastPrediction, ForecastTarget
from o1fm.crystals import ForecastCrystalBank
from o1fm.features import extract_features, feature_schema
from o1fm.identity import canonical_json_bytes, sha256_bytes, sha256_json
from o1fm.metrics import score_prediction
from o1fm.model import ForecastResult, O1FMConfig, O1FMForecaster
from o1fm.o1 import O1Config, O1ErrorStream
from o1fm.runtime import O1FMRuntime, RouteSelection
from o1fm.training import tensorize_inputs


def _inputs(*, known_future: float = 3.0) -> ForecastInputs:
    history = 8
    horizon = 4
    timestamps = np.datetime64("2026-01-01T00") + np.arange(history + horizon).astype(
        "timedelta64[h]"
    )
    target = np.ones((history, 1), dtype=np.float32)
    past = np.linspace(0.0, 1.0, history, dtype=np.float32)[:, None]
    known = np.ones((history + horizon, 1), dtype=np.float32)
    known[history:] = known_future
    return ForecastInputs(
        series_id="series-a",
        horizon=horizon,
        timestamps=timestamps,
        past_target=target,
        past_covariates=past,
        known_covariates=known,
    )


def _target(inputs: ForecastInputs, value: float = 3.0) -> ForecastTarget:
    return ForecastTarget(
        series_id=inputs.series_id,
        timestamps=inputs.future_timestamps,
        values=np.full((inputs.horizon, inputs.target_dim), value, np.float32),
    )


class KnownFutureModel(O1FMForecaster):
    """Tiny deterministic foundation authority for runtime integration tests."""

    def __init__(self) -> None:
        super().__init__(O1FMConfig(patch_len=4, d_model=16, num_heads=2, num_layers=1))

    def forward(self, batch):  # type: ignore[override]
        target_count = int((batch.role_ids == 0).sum())
        known_index = int(torch.nonzero(batch.role_ids == 2, as_tuple=False)[0])
        point = batch.values[:, known_index : known_index + 1, -batch.horizon :]
        point = point.expand(-1, target_count, -1)
        offsets = torch.tensor(
            (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4),
            dtype=point.dtype,
            device=point.device,
        )
        return ForecastResult(
            median=point,
            quantiles=point.unsqueeze(-1) + offsets,
        )


def test_features_and_seasonal_baseline_are_deterministic() -> None:
    inputs = _inputs()
    first = extract_features(inputs, seasonal_period=4)
    second = extract_features(inputs, seasonal_period=4)
    np.testing.assert_array_equal(first, second)
    assert len(feature_schema(inputs, seasonal_period=4)) == 64

    prediction = SeasonalNaiveForecaster(period=4).predict(inputs)
    prediction.validate_for(inputs)
    assert np.all(np.diff(prediction.quantiles, axis=2) >= 0.0)
    assert score_prediction(prediction, _target(inputs)).mae == 2.0


def test_tensor_adapter_masks_every_future_target_and_past_covariate() -> None:
    inputs = _inputs()
    batch = tensorize_inputs((inputs,))
    context = inputs.history_length
    assert not batch.observed_mask[0, 0, context:].any()
    assert not batch.observed_mask[0, 1, context:].any()
    assert batch.observed_mask[0, 2, context:].all()
    assert torch.all(batch.values[0, 0, context:] == 0.0)


def test_runtime_promotes_zero_forward_action_after_sequential_settlement(
    tmp_path: Path,
) -> None:
    model = KnownFutureModel()
    quantiles = model.config.quantiles
    runtime = O1FMRuntime(
        model,
        model_identity="a" * 64,
        o1=O1ErrorStream(
            tmp_path / "o1.json",
            stream_id="series-a",
            config=O1Config(quantiles=(0.1, 0.5, 0.9)),
        ),
        graph=OnlineCausalGraph(
            tmp_path / "graph",
            graph_id="series-a",
            config=CausalGraphConfig(min_observations=2, max_lag=2, top_k=2),
        ),
        crystals=ForecastCrystalBank(
            tmp_path / "crystals.json",
            ridge=0.01,
            min_support=4,
            min_sequential_support=2,
            quantile_levels=quantiles,
        ),
        actions=ActionBank(
            tmp_path / "actions.json",
            min_support=2,
            quality_tolerance=0.05,
            coverage_tolerance=0.05,
            forward_cost_weight=10.0,
        ),
        seasonal_period=4,
        enable_seasonal=False,
    )
    inputs = _inputs()
    target = _target(inputs)
    assert runtime.compile_teacher_crystals is False
    actions = []
    forwards = 0
    for _ in range(10):
        ticket = runtime.forecast(inputs)
        settlement = runtime.settle(ticket, target)
        actions.append(ticket.action)
        forwards += settlement.foundation_forwards

    assert actions[0] == "foundation"
    assert "crystal" in actions
    assert forwards < len(actions)
    assert runtime.o1.state.observations == len(actions)


def _edge(*, lag: int, score: float, revision: str = "b" * 64) -> CausalEdge:
    return CausalEdge(
        source="past:0",
        target="target:0",
        lag=lag,
        kind="observational",
        observations=8,
        covariance=score,
        source_variance=1.0,
        target_variance=1.0,
        score=score,
        revision_sha256=revision,
    )


def _runtime_for_routing(tmp_path: Path, *, route_stability: int = 2) -> O1FMRuntime:
    model = KnownFutureModel()
    return O1FMRuntime(
        model,
        model_identity="a" * 64,
        o1=O1ErrorStream(tmp_path / "o1-routing.json", stream_id="series-a"),
        graph=OnlineCausalGraph(tmp_path / "graph-routing", graph_id="series-a"),
        crystals=ForecastCrystalBank(
            tmp_path / "crystals-routing.json",
            quantile_levels=model.config.quantiles,
        ),
        actions=ActionBank(tmp_path / "actions-routing.json"),
        route_stability=route_stability,
    )


def test_route_identity_contains_edge_topology(tmp_path: Path) -> None:
    runtime = _runtime_for_routing(tmp_path)
    runtime.graph.route = lambda _target, top_k=None: (_edge(lag=1, score=0.7),)  # type: ignore[method-assign]
    first = runtime._route_candidate(_inputs())
    runtime.graph.route = lambda _target, top_k=None: (_edge(lag=2, score=0.7),)  # type: ignore[method-assign]
    second = runtime._route_candidate(_inputs())

    assert first.past_indices == second.past_indices == (0,)
    assert first.transport_w1 == 1.0
    assert second.transport_w1 == 2.0
    assert first.signature_sha256 != second.signature_sha256


def test_route_hysteresis_refreshes_scores_and_promotes_stable_topology(
    tmp_path: Path,
) -> None:
    runtime = _runtime_for_routing(tmp_path, route_stability=2)
    initial = RouteSelection((0,), (0,), "1" * 64, (_edge(lag=1, score=0.2),))
    refreshed = RouteSelection(
        (0,),
        (0,),
        initial.signature_sha256,
        (_edge(lag=1, score=0.9, revision="c" * 64),),
    )
    changed = RouteSelection((0,), (0,), "2" * 64, (_edge(lag=2, score=0.8),))
    candidates = iter((initial, refreshed, changed, changed))
    runtime._observe_graph_history = lambda _inputs: None  # type: ignore[method-assign]
    runtime._route_candidate = lambda _inputs: next(candidates)  # type: ignore[method-assign]

    assert runtime._route(_inputs(), regime="regime:0") is initial
    assert runtime._route(_inputs(), regime="regime:0").edges[0].score == 0.9
    assert runtime._route(_inputs(), regime="regime:0").signature_sha256 == "1" * 64
    assert runtime._route(_inputs(), regime="regime:0") is changed


def test_runtime_trains_crystal_from_partial_settlement(tmp_path: Path) -> None:
    runtime = _runtime_for_routing(tmp_path)
    inputs = _inputs()
    mask = np.zeros((inputs.horizon, inputs.target_dim), dtype=np.bool_)
    mask[0, 0] = True
    target = ForecastTarget(
        series_id=inputs.series_id,
        timestamps=inputs.future_timestamps,
        values=np.full((inputs.horizon, inputs.target_dim), 3.0, np.float32),
        mask=mask,
    )

    ticket = runtime.forecast(inputs)
    settlement = runtime.settle(ticket, target)
    stats = runtime.crystals.describe(ticket.crystal_key)

    assert settlement.score.observed_values == 1
    assert runtime.o1.metrics()["value_count"] == 1
    assert stats is not None
    assert stats.support == 1


def test_runtime_applies_and_settles_directed_conformal_field(tmp_path: Path) -> None:
    model = KnownFutureModel()
    field = O1ConformalField(tmp_path / "conformal")
    runtime = O1FMRuntime(
        model,
        model_identity="a" * 64,
        o1=O1ErrorStream(tmp_path / "o1-conformal.json", stream_id="series-a"),
        graph=OnlineCausalGraph(tmp_path / "graph-conformal", graph_id="series-a"),
        crystals=ForecastCrystalBank(
            tmp_path / "crystals-conformal.json",
            quantile_levels=model.config.quantiles,
        ),
        actions=ActionBank(tmp_path / "actions-conformal.json"),
        conformal=field,
        enable_seasonal=False,
    )
    inputs = _inputs(known_future=3.0)
    target = _target(inputs, value=5.0)

    cold = runtime.forecast(inputs)
    np.testing.assert_array_equal(
        cold.prediction.quantiles, cold.base_prediction.quantiles
    )
    first = runtime.settle(cold, target)
    assert first.calibration_update is not None
    assert first.joint_covered is False

    charged = runtime.forecast(inputs)
    assert charged.calibration is not None
    np.testing.assert_array_equal(
        charged.prediction.point, charged.base_prediction.point
    )
    assert np.all(
        charged.prediction.quantiles[:, :, -1]
        > charged.base_prediction.quantiles[:, :, -1]
    )
    second = runtime.settle(charged, target)
    assert second.score.interval_coverage == 1.0
    assert second.base_score.interval_coverage == 0.0
    assert field.metrics()["settlements"] == 2


class InjectedTeacher:
    quantile_levels = KnownFutureModel().config.quantiles

    def predict(self, inputs: ForecastInputs):
        point = np.broadcast_to(
            inputs.known_covariates[-inputs.horizon :, :1],
            (inputs.horizon, inputs.target_dim),
        ).copy()
        offsets = np.linspace(-0.4, 0.4, len(self.quantile_levels), dtype=np.float32)
        return ForecastPrediction(
            series_id=inputs.series_id,
            timestamps=inputs.future_timestamps,
            point=point,
            quantiles=point[..., None] + offsets,
            quantile_levels=self.quantile_levels,
        )


class MemorySegmentStore:
    def __init__(self) -> None:
        self.rows = {}
        self.sequence = 0
        self.head = "0" * 64

    def append_segment(self, records):
        materialized = tuple(json.loads(canonical_json_bytes(row)) for row in records)
        body = b"".join(canonical_json_bytes(row) + b"\n" for row in materialized)
        digest = sha256_bytes(body)
        if digest not in self.rows:
            self.rows[digest] = materialized
            self.sequence += 1
            self.head = sha256_json(
                {"previous": self.head, "segment": digest, "sequence": self.sequence}
            )
        return digest

    def revision(self):
        return self.sequence, self.head

    def record(self, sha, index):
        return self.rows[sha][index]

    def verify(self):
        return True


def test_runtime_accepts_external_forecast_teacher(tmp_path: Path) -> None:
    teacher = InjectedTeacher()
    runtime = O1FMRuntime(
        teacher,
        model_identity="b" * 64,
        o1=O1ErrorStream(tmp_path / "o1-teacher.json", stream_id="series-a"),
        graph=OnlineCausalGraph(tmp_path / "graph-teacher", graph_id="series-a"),
        crystals=ForecastCrystalBank(
            tmp_path / "crystals-teacher.json",
            quantile_levels=teacher.quantile_levels,
        ),
        actions=ActionBank(tmp_path / "actions-teacher.json"),
        enable_seasonal=False,
    )

    ticket = runtime.forecast(_inputs())
    settlement = runtime.settle(ticket, _target(_inputs()))

    assert runtime.teacher is teacher
    assert runtime.compile_teacher_crystals is True
    assert ticket.action == "foundation"
    assert settlement.score.mae == 0.0


def test_runtime_routes_to_verified_local_teacher_and_prices_its_forward(
    tmp_path: Path,
) -> None:
    foundation = InjectedTeacher()
    local = InjectedTeacher()
    runtime = O1FMRuntime(
        foundation,
        model_identity="c" * 64,
        o1=O1ErrorStream(tmp_path / "o1-local.json", stream_id="series-a"),
        graph=OnlineCausalGraph(tmp_path / "graph-local", graph_id="series-a"),
        crystals=ForecastCrystalBank(
            tmp_path / "crystals-local.json",
            quantile_levels=foundation.quantile_levels,
        ),
        actions=ActionBank(
            tmp_path / "actions-local.json",
            min_support=1,
            latency_cost_weight=0.0,
            forward_cost_weight=1.0,
        ),
        local_teacher=local,
        local_forward_cost=0.2,
        enable_seasonal=False,
    )
    actions = []
    settlements = []
    for _ in range(2):
        ticket = runtime.forecast(_inputs())
        actions.append(ticket.action)
        settlements.append(runtime.settle(ticket, _target(_inputs())))

    assert actions == ["foundation", "local"]
    assert settlements[0].local_forwards == 1
    assert settlements[1].local_forwards == 1
    assert settlements[1].foundation_forwards == 0
    assert settlements[1].forward_cost == 0.2


def test_runtime_forces_periodic_foundation_refresh(tmp_path: Path) -> None:
    foundation = InjectedTeacher()
    runtime = O1FMRuntime(
        foundation,
        model_identity="d" * 64,
        o1=O1ErrorStream(tmp_path / "o1-refresh.json", stream_id="series-a"),
        graph=OnlineCausalGraph(tmp_path / "graph-refresh", graph_id="series-a"),
        crystals=ForecastCrystalBank(
            tmp_path / "crystals-refresh.json",
            quantile_levels=foundation.quantile_levels,
        ),
        actions=ActionBank(
            tmp_path / "actions-refresh.json",
            min_support=1,
            latency_cost_weight=0.0,
            forward_cost_weight=1.0,
        ),
        local_teacher=InjectedTeacher(),
        local_forward_cost=0.2,
        foundation_refresh_interval=1,
        enable_seasonal=False,
    )
    tickets = []
    settlements = []
    for _ in range(3):
        ticket = runtime.forecast(_inputs())
        tickets.append(ticket)
        settlements.append(runtime.settle(ticket, _target(_inputs())))

    assert [ticket.action for ticket in tickets] == [
        "foundation",
        "local",
        "foundation",
    ]
    assert [item.forced_foundation_refresh for item in settlements] == [
        False,
        False,
        True,
    ]


def test_runtime_shadow_calibrates_local_candidate_without_pool_duplication(
    tmp_path: Path,
) -> None:
    foundation = InjectedTeacher()
    field = O1ConformalField(tmp_path / "field-shadow")
    runtime = O1FMRuntime(
        foundation,
        model_identity="e" * 64,
        o1=O1ErrorStream(tmp_path / "o1-shadow.json", stream_id="series-a"),
        graph=OnlineCausalGraph(tmp_path / "graph-shadow", graph_id="series-a"),
        crystals=ForecastCrystalBank(
            tmp_path / "crystals-shadow.json",
            quantile_levels=foundation.quantile_levels,
        ),
        actions=ActionBank(tmp_path / "actions-shadow.json", min_support=1),
        conformal=field,
        local_teacher=InjectedTeacher(),
        enable_seasonal=False,
    )
    ticket = runtime.forecast(_inputs())
    settlement = runtime.settle(ticket, _target(_inputs(), value=5.0))

    assert ticket.action == "foundation"
    assert set(settlement.shadow_calibration_updates) == {"local"}
    assert field.metrics()["exact_profile_count"] == 2
    assert field.metrics()["pooled_profile_count"] == 1


def test_runtime_emits_direct_forecast_transport_and_settlement_citations(
    tmp_path: Path,
) -> None:
    store = MemorySegmentStore()
    atlas = ForecastAtlas(store)
    foundation = InjectedTeacher()
    runtime = O1FMRuntime(
        foundation,
        model_identity="f" * 64,
        o1=O1ErrorStream(tmp_path / "o1-atlas.json", stream_id="series-a"),
        graph=OnlineCausalGraph(tmp_path / "graph-atlas", graph_id="series-a"),
        crystals=ForecastCrystalBank(
            tmp_path / "crystals-atlas.json",
            quantile_levels=foundation.quantile_levels,
        ),
        actions=ActionBank(tmp_path / "actions-atlas.json"),
        atlas=atlas,
        enable_seasonal=False,
    )
    ticket = runtime.forecast(_inputs())
    settlement = runtime.settle(ticket, _target(_inputs()))

    assert ticket.atlas_forecast_receipt is not None
    assert ticket.atlas_transport_receipt is not None
    assert settlement.atlas_settlement_receipt is not None
    assert store.sequence == 3
    assert atlas.record(ticket.atlas_forecast_receipt)["kind"] == "forecast"
    assert atlas.record(ticket.atlas_transport_receipt)["kind"] == "transport"
    assert atlas.record(settlement.atlas_settlement_receipt)["kind"] == "settlement"
