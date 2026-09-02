from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from o1fm.branches import InterventionPlan, InterventionSpec
from o1fm.cartography import ForecastCartographer, ForecastProbeTarget
from o1fm.causal_graph import CausalEdge
from o1fm.contracts import ForecastInputs
from o1fm.response_features import (
    OPERATOR_CELL_FEATURES,
    InterventionResponseFeatureEncoder,
    ResponseFeatureError,
    ResponseFeatureSchema,
    ResponseRouteContext,
    encode_intervention_response_features,
)


def _inputs(
    *, future_shift: float = 0.0, start_offset_hours: int = 0
) -> ForecastInputs:
    history = 12
    horizon = 6
    timestamps = (
        np.datetime64("2026-01-01T00")
        + np.timedelta64(start_offset_hours, "h")
        + np.arange(history + horizon).astype("timedelta64[h]")
    )
    first = np.arange(history, dtype=np.float32)
    second = np.asarray(
        (0.0, 1.0, -1.0, 2.0, -2.0, 1.0, 0.0, 3.0, -1.0, 1.0, 2.0, 0.0),
        dtype=np.float32,
    )
    target = np.stack((first, second), axis=1)
    target_mask = np.ones_like(target, dtype=np.bool_)
    target_mask[2, 0] = False
    target_mask[-2, 1] = False
    known = np.stack(
        (
            np.linspace(-0.5, 0.5, history + horizon, dtype=np.float32),
            np.ones(history + horizon, dtype=np.float32),
            np.linspace(0.0, 1.0, history + horizon, dtype=np.float32),
        ),
        axis=1,
    )
    known[history:] += np.float32(future_shift)
    return ForecastInputs(
        series_id="series-a",
        horizon=horizon,
        timestamps=timestamps,
        past_target=target,
        past_target_mask=target_mask,
        past_covariates=np.zeros((history, 1), dtype=np.float32),
        known_covariates=known,
    )


def _schema(
    *,
    topology: str = "a",
    target_names: tuple[str, ...] = ("target:0", "target:1"),
    known_names: tuple[str, ...] = ("known:0", "known:1", "known:2"),
    projection_scope: str | None = None,
) -> ResponseFeatureSchema:
    return ResponseFeatureSchema(
        target_names=target_names,
        known_covariate_names=known_names,
        horizon=6,
        graph_topology_sha256=topology * 64,
        seasonal_periods=(3, 6),
        state_window=10,
        mask_tail=4,
        design_dimension=64,
        projection_scope_sha256=projection_scope,
    )


def _edge(
    *,
    source: str = "known:0",
    target: str = "target:0",
    lag: int = 0,
    score: float = 0.7,
    covariance: float | None = None,
    kind: str = "intervention",
) -> CausalEdge:
    return CausalEdge(
        source=source,
        target=target,
        lag=lag,
        kind=kind,  # type: ignore[arg-type]
        observations=12,
        covariance=score if covariance is None else covariance,
        source_variance=1.0,
        target_variance=1.0,
        score=score,
        revision_sha256="b" * 64,
    )


def _route(*edges: CausalEdge, topology: str = "a") -> ResponseRouteContext:
    return ResponseRouteContext(
        graph_topology_sha256=topology * 64,
        edges=edges,
    )


def _plan(
    inputs: ForecastInputs,
    *,
    name: str = "shift",
    operation: str = "shift",
    channel: int = 0,
    rows: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    values: tuple[float, ...] | None = None,
    edit_mask: object | None = None,
) -> InterventionPlan:
    selected = inputs.future_timestamps[list(rows)]
    operands = values or tuple(0.5 for _ in rows)
    return InterventionPlan(
        name,
        (
            InterventionSpec(
                name=f"{name}-cells",
                operation=operation,
                timestamps=selected,
                covariate_indices=(channel,),
                values=np.asarray(operands, dtype=np.float32)[:, None],
                edit_mask=edit_mask,
                lower_bounds=(-100.0,),
                upper_bounds=(100.0,),
            ),
        ),
    )


def _cell(features, channel: int, row: int, name: str) -> float:
    return float(features.operator[channel, row, OPERATOR_CELL_FEATURES.index(name)])


def test_schema_binds_order_horizon_seasonal_basis_topology_and_layout() -> None:
    schema = _schema()
    assert schema.feature_dimension == (
        1 + schema.state_dimension + schema.route_dimension + schema.operator_dimension
    )
    assert len(schema.identity_sha256) == 64
    assert len(schema.target_schema_sha256) == 64
    assert len(schema.known_covariate_schema_sha256) == 64
    assert (
        schema.identity_sha256
        != _schema(target_names=("target:1", "target:0")).identity_sha256
    )
    assert (
        schema.identity_sha256
        != _schema(known_names=("known:1", "known:0", "known:2")).identity_sha256
    )
    assert schema.identity_sha256 != _schema(topology="c").identity_sha256
    assert schema.layout_identity_sha256 == _schema(topology="c").layout_identity_sha256
    assert schema.resolved_projection_scope_sha256 == schema.identity_sha256
    changed_period = ResponseFeatureSchema(
        target_names=schema.target_names,
        known_covariate_names=schema.known_covariate_names,
        horizon=6,
        graph_topology_sha256="a" * 64,
        seasonal_periods=(2, 6),
        state_window=10,
        mask_tail=4,
        design_dimension=64,
    )
    assert schema.identity_sha256 != changed_period.identity_sha256
    schema.validate_inputs(_inputs())
    with pytest.raises(ResponseFeatureError, match="inputs differ"):
        _schema(known_names=("known:0", "known:1")).validate_inputs(_inputs())


def test_encoder_is_leakage_free_deterministic_and_immutable() -> None:
    inputs = _inputs()
    schema = _schema()
    encoder = InterventionResponseFeatureEncoder(schema)
    plan = _plan(inputs)
    route = _route(_edge())

    first = encoder.encode(inputs, plan, route)
    second = encode_intervention_response_features(schema, inputs, plan, route)

    assert tuple(inspect.signature(encoder.encode).parameters) == (
        "inputs",
        "plan",
        "route",
    )
    assert "future_target" not in inputs.model_inputs()
    assert first.identity_sha256 == second.identity_sha256
    np.testing.assert_array_equal(first.vector, second.vector)
    np.testing.assert_array_equal(first.design_vector, second.design_vector)
    assert first.vector.shape == (schema.feature_dimension,)
    assert first.design_vector.shape == (64,)
    assert np.max(np.abs(first.vector)) <= 1.0
    assert np.max(np.abs(first.design_vector)) <= 1.0
    assert not first.state.flags.writeable
    assert not first.route.flags.writeable
    assert not first.operator.flags.writeable
    assert not first.vector.flags.writeable
    assert not first.design_vector.flags.writeable
    with pytest.raises(ValueError):
        first.operator[0, 0, 0] = 0.0


def test_projection_scope_can_share_parent_space_without_relaxing_topology() -> None:
    inputs = _inputs()
    plan = _plan(inputs)
    default_a = InterventionResponseFeatureEncoder(_schema(topology="a")).encode(
        inputs,
        plan,
        _route(topology="a"),
    )
    default_c = InterventionResponseFeatureEncoder(_schema(topology="c")).encode(
        inputs,
        plan,
        _route(topology="c"),
    )
    np.testing.assert_array_equal(default_a.vector, default_c.vector)
    assert not np.array_equal(default_a.design_vector, default_c.design_vector)

    parent_scope = "d" * 64
    shared_a_schema = _schema(topology="a", projection_scope=parent_scope)
    shared_c_schema = _schema(topology="c", projection_scope=parent_scope)
    shared_a = InterventionResponseFeatureEncoder(shared_a_schema).encode(
        inputs,
        plan,
        _route(topology="a"),
    )
    shared_c = InterventionResponseFeatureEncoder(shared_c_schema).encode(
        inputs,
        plan,
        _route(topology="c"),
    )
    assert shared_a_schema.identity_sha256 != shared_c_schema.identity_sha256
    assert (
        shared_a_schema.layout_identity_sha256 == shared_c_schema.layout_identity_sha256
    )
    assert shared_a_schema.resolved_projection_scope_sha256 == parent_scope
    np.testing.assert_array_equal(shared_a.vector, shared_c.vector)
    np.testing.assert_array_equal(shared_a.design_vector, shared_c.design_vector)
    with pytest.raises(ResponseFeatureError, match="another graph topology"):
        InterventionResponseFeatureEncoder(shared_a_schema).encode(
            inputs,
            plan,
            _route(topology="c"),
        )


def test_factual_state_encodes_level_slope_volatility_seasonality_and_masks() -> None:
    features = InterventionResponseFeatureEncoder(_schema()).encode(
        _inputs(),
        _plan(_inputs()),
        _route(),
    )
    per_target = features.schema.state_features_per_target
    first = features.state[:per_target]
    second = features.state[per_target:]
    assert first[0] > 0.0  # last level
    assert first[2] > 0.0  # slope
    assert first[3] == 0.0  # constant first differences
    assert second[3] > 0.0  # volatile target
    assert first[4] < 1.0  # observed fraction carries the missing cell
    np.testing.assert_array_equal(first[7:11], (1.0, 1.0, 1.0, 1.0))
    np.testing.assert_array_equal(second[7:11], (1.0, 1.0, 0.0, 1.0))
    assert first[13] == 1.0  # seasonal lag availability for period 3
    shifted_inputs = _inputs(start_offset_hours=1)
    shifted = InterventionResponseFeatureEncoder(_schema()).encode(
        shifted_inputs,
        _plan(shifted_inputs),
        _route(),
    )
    assert not np.array_equal(features.state, shifted.state)


def test_operator_geometry_distinguishes_operations_channels_timing_and_sign() -> None:
    inputs = _inputs()
    encoder = InterventionResponseFeatureEncoder(_schema())
    route = _route()
    set_features = encoder.encode(
        inputs,
        _plan(inputs, name="set", operation="set", rows=(2,), values=(1.0,)),
        route,
    )
    shift_features = encoder.encode(
        inputs,
        _plan(inputs, name="shift", operation="shift", rows=(2,), values=(1.0,)),
        route,
    )
    scale_features = encoder.encode(
        inputs,
        _plan(inputs, name="scale", operation="scale", rows=(2,), values=(1.0,)),
        route,
    )
    assert _cell(set_features, 0, 2, "operation_set") == 1.0
    assert _cell(shift_features, 0, 2, "operation_shift") == 1.0
    assert _cell(scale_features, 0, 2, "operation_scale") == 1.0
    assert (
        len(
            {
                set_features.identity_sha256,
                shift_features.identity_sha256,
                scale_features.identity_sha256,
            }
        )
        == 3
    )

    pulse = encoder.encode(
        inputs,
        _plan(
            inputs,
            name="negative-pulse",
            channel=1,
            rows=(4,),
            values=(-0.5,),
        ),
        route,
    )
    assert _cell(pulse, 1, 4, "pulse") == 1.0
    assert _cell(pulse, 1, 4, "run_start") == 1.0
    assert _cell(pulse, 1, 4, "run_end") == 1.0
    assert _cell(pulse, 1, 4, "sign_negative") == 1.0
    assert _cell(pulse, 0, 4, "active") == 0.0

    ramp = encoder.encode(
        inputs,
        _plan(
            inputs,
            name="ramp",
            rows=(1, 2, 3),
            values=(-0.5, 0.0, 0.5),
        ),
        route,
    )
    assert all(_cell(ramp, 0, row, "ramp") == 1.0 for row in (1, 2, 3))
    assert _cell(ramp, 0, 1, "run_start") == 1.0
    assert _cell(ramp, 0, 3, "run_end") == 1.0
    assert _cell(ramp, 0, 2, "run_duration") == pytest.approx(0.5)
    assert _cell(ramp, 0, 2, "operand_slope") > 0.0


def test_interval_masks_and_multispec_plan_are_order_independent() -> None:
    inputs = _inputs()
    first = InterventionSpec(
        name="masked-interval",
        operation="shift",
        timestamps=inputs.future_timestamps,
        covariate_indices=(0,),
        values=np.ones((6, 1), dtype=np.float32),
        edit_mask=np.asarray(((True,), (True,), (False,), (True,), (True,), (False,))),
    )
    second = InterventionSpec(
        name="revealed-set",
        operation="set",
        timestamps=inputs.future_timestamps[2:4],
        covariate_indices=(2,),
        values=np.asarray(((0.25,), (0.75,)), dtype=np.float32),
        observed_mask=np.ones((2, 1), dtype=np.bool_),
    )
    left = InterventionPlan("multi", (first, second))
    right = InterventionPlan("multi", (second, first))
    encoder = InterventionResponseFeatureEncoder(_schema())
    left_features = encoder.encode(inputs, left, _route())
    right_features = encoder.encode(inputs, right, _route())
    assert left.identity == right.identity
    np.testing.assert_array_equal(left_features.operator, right_features.operator)
    assert _cell(left_features, 0, 0, "run_start") == 1.0
    assert _cell(left_features, 0, 1, "run_end") == 1.0
    assert _cell(left_features, 0, 2, "active") == 0.0
    assert _cell(left_features, 0, 3, "run_start") == 1.0
    assert _cell(left_features, 2, 2, "revealed") == 1.0


def test_route_geometry_binds_topology_w1_edges_channels_and_targets() -> None:
    inputs = _inputs()
    plan = _plan(inputs)
    encoder = InterventionResponseFeatureEncoder(_schema())
    empty = encoder.encode(inputs, plan, _route())
    routed = encoder.encode(
        inputs,
        plan,
        _route(
            _edge(),
            _edge(
                source="known:1",
                target="target:1",
                lag=2,
                score=0.3,
                covariance=-0.3,
                kind="observational",
            ),
        ),
    )
    assert not np.array_equal(empty.route, routed.route)
    assert empty.route_edge_set_sha256 != routed.route_edge_set_sha256
    supplied = encoder.encode(
        inputs,
        plan,
        ResponseRouteContext(
            graph_topology_sha256="a" * 64,
            edges=(),
            per_target_w1={"target:0": 1.0, "target:1": 3.0},
        ),
    )
    assert supplied.route[0] > 0.0
    with pytest.raises(ResponseFeatureError, match="another graph topology"):
        encoder.encode(inputs, plan, _route(topology="f"))
    with pytest.raises(ResponseFeatureError, match="cover.*exactly"):
        encoder.encode(
            inputs,
            plan,
            ResponseRouteContext(
                graph_topology_sha256="a" * 64,
                per_target_w1={"target:0": 1.0},
            ),
        )


def _probe(seed: int) -> ForecastProbeTarget:
    return ForecastProbeTarget(
        series_id="series-a",
        dataset_identity_sha256="1" * 64,
        model_identity_sha256="2" * 64,
        graph_topology_sha256="a" * 64,
        regime="stable",
        horizon=6,
        target_indices=(0, 1),
        action_inventory=("foundation", "crystal"),
        seed=seed,
        expected_work=1.0,
        expected_bytes=10,
    )


def test_pre_outcome_design_vectors_feed_d_optimal_cartography(tmp_path: Path) -> None:
    inputs = _inputs()
    encoder = InterventionResponseFeatureEncoder(_schema())
    plans = (
        _plan(inputs, name="early", channel=0, rows=(0,), values=(0.5,)),
        _plan(inputs, name="late", channel=0, rows=(5,), values=(0.5,)),
        _plan(inputs, name="other-channel", channel=2, rows=(2,), values=(-0.5,)),
    )
    probes = tuple(_probe(index) for index in range(3))
    vectors = {
        probe: encoder.design_vector(inputs, plan, _route())
        for probe, plan in zip(probes, plans, strict=True)
    }
    assert all(vector.shape == (64,) for vector in vectors.values())
    assert len({vector.tobytes() for vector in vectors.values()}) == 3
    cartographer = ForecastCartographer(tmp_path / "map")
    cartographer.extend_targets(probes)
    selected = cartographer.d_optimal_batch(vectors, batch_size=2)
    reversed_selected = cartographer.d_optimal_batch(
        dict(reversed(tuple(vectors.items()))),
        batch_size=2,
    )
    assert len(selected) == 2
    assert selected == reversed_selected


def test_context_plan_and_route_changes_have_distinct_canonical_identities() -> None:
    encoder = InterventionResponseFeatureEncoder(_schema())
    inputs = _inputs()
    base = encoder.encode(inputs, _plan(inputs), _route())
    changed_context = encoder.encode(
        _inputs(future_shift=0.1),
        _plan(_inputs(future_shift=0.1)),
        _route(),
    )
    changed_plan = encoder.encode(
        inputs,
        _plan(inputs, values=tuple(-0.5 for _ in range(6))),
        _route(),
    )
    changed_route = encoder.encode(inputs, _plan(inputs), _route(_edge()))
    assert (
        len(
            {
                base.identity_sha256,
                changed_context.identity_sha256,
                changed_plan.identity_sha256,
                changed_route.identity_sha256,
            }
        )
        == 4
    )
    assert base.plan_identity_sha256 != changed_plan.plan_identity_sha256
    assert base.input_context_sha256 != changed_context.input_context_sha256


def test_strict_schema_and_value_bounds_reject_invalid_contracts() -> None:
    with pytest.raises(ResponseFeatureError):
        ResponseFeatureSchema(
            target_names=("same",),
            known_covariate_names=("same",),
            horizon=6,
            graph_topology_sha256="a" * 64,
        )
    with pytest.raises(ResponseFeatureError):
        ResponseFeatureSchema(
            target_names=("target",),
            known_covariate_names=("known",),
            horizon=6,
            graph_topology_sha256="a" * 64,
            design_dimension=1025,
        )
    with pytest.raises(ResponseFeatureError):
        ResponseFeatureSchema(
            target_names=("target",),
            known_covariate_names=("known",),
            horizon=6,
            graph_topology_sha256="a" * 64,
            seasonal_periods=(6, 3),
        )
    with pytest.raises(ResponseFeatureError):
        ResponseRouteContext(
            graph_topology_sha256="a" * 64,
            per_target_w1={"target:0": -1.0},
        )
