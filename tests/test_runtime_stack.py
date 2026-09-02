from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from o1fm.action_bank import ActionBank
from o1fm.baselines import SeasonalNaiveForecaster
from o1fm.causal_graph import CausalEdge, CausalGraphConfig, OnlineCausalGraph
from o1fm.contracts import ForecastInputs, ForecastTarget
from o1fm.crystals import ForecastCrystalBank
from o1fm.features import extract_features, feature_schema
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
