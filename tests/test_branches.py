from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from o1fm.atlas import AtlasReceipt
from o1fm.branches import (
    BranchExecutionResult,
    BranchExecutor,
    BranchSet,
    ForecastBranch,
    IdentificationStatus,
    InterventionOperation,
    InterventionPlan,
    InterventionSpec,
    JointEnvelope,
    apply_intervention,
)
from o1fm.causal_graph import CausalEdge
from o1fm.contracts import ForecastInputs, ForecastPrediction


def _inputs() -> ForecastInputs:
    history = 4
    horizon = 3
    timestamps = np.datetime64("2026-01-01T00") + np.arange(history + horizon).astype(
        "timedelta64[h]"
    )
    known = np.stack(
        (
            np.arange(history + horizon, dtype=np.float32),
            np.full(history + horizon, 2.0, dtype=np.float32),
        ),
        axis=1,
    )
    known_mask = np.ones_like(known, dtype=np.bool_)
    known_mask[history + 1, 1] = False
    return ForecastInputs(
        series_id="retail-a",
        horizon=horizon,
        timestamps=timestamps,
        past_target=np.arange(history, dtype=np.float32)[:, None],
        past_covariates=np.full((history, 1), 7.0, dtype=np.float32),
        known_covariates=known,
        known_covariate_mask=known_mask,
    )


def _spec(
    inputs: ForecastInputs,
    *,
    name: str = "price-cut",
    operation: InterventionOperation | str = "set",
    covariate: int = 0,
    values: tuple[float, ...] = (10.0, 11.0, 12.0),
    observed_mask: object | None = None,
) -> InterventionSpec:
    return InterventionSpec(
        name=name,
        operation=operation,
        timestamps=inputs.future_timestamps,
        covariate_indices=(covariate,),
        values=np.asarray(values, dtype=np.float32)[:, None],
        observed_mask=observed_mask,
        lower_bounds=(-100.0,),
        upper_bounds=(100.0,),
    )


def _receipt(seed: str = "a") -> AtlasReceipt:
    return AtlasReceipt(
        segment_sha256=seed * 64,
        revision_sequence=1,
        revision_sha256="f" * 64,
        record_count=1,
        kind="forecast",
    )


def _edge(*, source: str = "known:0", kind: str = "intervention") -> CausalEdge:
    return CausalEdge(
        source=source,
        target="target:0",
        lag=0,
        kind=kind,  # type: ignore[arg-type]
        observations=12,
        covariance=0.8,
        source_variance=1.0,
        target_variance=1.0,
        score=0.8,
        revision_sha256="e" * 64,
    )


def _prediction(inputs: ForecastInputs) -> ForecastPrediction:
    point = inputs.known_covariates[inputs.history_length :, :1]
    offsets = np.asarray((-1.0, 0.0, 1.0), dtype=np.float32)
    return ForecastPrediction(
        series_id=inputs.series_id,
        timestamps=inputs.future_timestamps,
        point=point,
        quantiles=point[:, :, None] + offsets,
        quantile_levels=(0.1, 0.5, 0.9),
    )


def _execution(
    inputs: ForecastInputs,
    *,
    edges: tuple[CausalEdge, ...] = (),
) -> BranchExecutionResult:
    prediction = _prediction(inputs)
    envelope = JointEnvelope(
        series_id=prediction.series_id,
        timestamps=prediction.timestamps,
        lower=prediction.quantiles[:, :, 0] - 0.25,
        upper=prediction.quantiles[:, :, -1] + 0.25,
    )
    return BranchExecutionResult(
        prediction=prediction,
        action_identity="toto-22m",
        model_identity="model-v1",
        profile_identity="profile-v1",
        route_identity="route-v1",
        graph_identity="graph-v1",
        route_w1=0.25,
        atlas_citations=(_receipt(),),
        joint_envelope=envelope,
        route_edges=edges,
        assumptions=("Known covariate schedule is exact",),
    )


def test_interventions_are_future_only_and_preserve_every_other_input_channel() -> None:
    inputs = _inputs()
    plan = InterventionPlan("scenario", (_spec(inputs),))
    changed = apply_intervention(inputs, plan)

    np.testing.assert_array_equal(changed.past_target, inputs.past_target)
    np.testing.assert_array_equal(changed.past_target_mask, inputs.past_target_mask)
    np.testing.assert_array_equal(changed.past_covariates, inputs.past_covariates)
    np.testing.assert_array_equal(
        changed.past_covariate_mask, inputs.past_covariate_mask
    )
    np.testing.assert_array_equal(changed.timestamps, inputs.timestamps)
    np.testing.assert_array_equal(
        changed.known_covariates[: inputs.history_length],
        inputs.known_covariates[: inputs.history_length],
    )
    np.testing.assert_array_equal(
        changed.known_covariates[inputs.history_length :, 0],
        (10.0, 11.0, 12.0),
    )
    np.testing.assert_array_equal(
        changed.known_covariate_mask, inputs.known_covariate_mask
    )
    assert "target" not in plan.to_record()

    past_edit = InterventionSpec(
        name="illegal-past-edit",
        operation="set",
        timestamps=inputs.past_timestamps[-1:],
        covariate_indices=(0,),
        values=((999.0,),),
    )
    with pytest.raises(ValueError, match="exact forecast horizon"):
        apply_intervention(inputs, InterventionPlan("illegal", (past_edit,)))


def test_set_shift_scale_masks_revelation_and_declared_bounds() -> None:
    inputs = _inputs()
    set_result = apply_intervention(
        inputs,
        InterventionPlan("set", (_spec(inputs, values=(3.0, 4.0, 5.0)),)),
    )
    shift_result = apply_intervention(
        inputs,
        InterventionPlan(
            "shift",
            (_spec(inputs, operation="shift", values=(1.0, 2.0, 3.0)),),
        ),
    )
    scale_result = apply_intervention(
        inputs,
        InterventionPlan(
            "scale",
            (_spec(inputs, operation="scale", values=(2.0, 3.0, 4.0)),),
        ),
    )
    np.testing.assert_array_equal(
        set_result.known_covariates[inputs.history_length :, 0], (3.0, 4.0, 5.0)
    )
    np.testing.assert_array_equal(
        shift_result.known_covariates[inputs.history_length :, 0], (5.0, 7.0, 9.0)
    )
    np.testing.assert_array_equal(
        scale_result.known_covariates[inputs.history_length :, 0], (8.0, 15.0, 24.0)
    )

    reveal = _spec(
        inputs,
        name="revealed-promotion",
        covariate=1,
        values=(2.0, 9.0, 2.0),
        observed_mask=np.asarray(((False,), (True,), (False,)), dtype=np.bool_),
    )
    revealed = apply_intervention(inputs, InterventionPlan("reveal", (reveal,)))
    assert revealed.known_covariate_mask[inputs.history_length + 1, 1]

    bounded = InterventionSpec(
        name="bounded",
        operation="shift",
        timestamps=inputs.future_timestamps[:1],
        covariate_indices=(0,),
        values=((50.0,),),
        lower_bounds=(-10.0,),
        upper_bounds=(10.0,),
    )
    with pytest.raises(ValueError, match="declared bounds"):
        apply_intervention(inputs, InterventionPlan("bounded", (bounded,)))


def test_intervention_identity_is_canonical_order_independent_and_immutable() -> None:
    inputs = _inputs()
    first = _spec(inputs, name="a")
    same = InterventionSpec(
        name="a",
        operation=InterventionOperation.SET,
        timestamps=list(inputs.future_timestamps),
        covariate_indices=[0],
        values=[[10], [11], [12]],
        lower_bounds=[-100],
        upper_bounds=[100],
    )
    second = _spec(inputs, name="b", covariate=1, values=(4.0, 4.0, 4.0))
    assert first.identity == same.identity
    assert first.canonical_identity == first.identity
    assert (
        InterventionPlan("plan", (first, second)).identity
        == InterventionPlan("plan", (second, first)).identity
    )
    with pytest.raises(ValueError):
        first.values[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        first.name = "mutated"  # type: ignore[misc]


def test_executor_separates_graph_support_observed_policy_and_hypothesis() -> None:
    inputs = _inputs()
    supported_plan = InterventionPlan("supported", (_spec(inputs),))
    observed_plan = InterventionPlan(
        "observed",
        (_spec(inputs, name="observed-edit", values=(8.0, 8.0, 8.0)),),
        observed_policy=True,
    )
    supported = BranchExecutor(
        lambda value: _execution(value, edges=(_edge(),))
    ).execute(inputs, (supported_plan, observed_plan))
    statuses = {branch.identification for branch in supported.branches.values()}
    assert statuses == {
        IdentificationStatus.FACTUAL,
        IdentificationStatus.GRAPH_SUPPORTED_INTERVENTION,
        IdentificationStatus.OBSERVED_POLICY_INTERVENTION,
    }
    causal = next(
        branch
        for branch in supported.branches.values()
        if branch.identification is IdentificationStatus.GRAPH_SUPPORTED_INTERVENTION
    )
    assert causal.supporting_edges == (_edge(),)

    hypothetical = BranchExecutor(lambda value: _execution(value)).execute(
        inputs, (supported_plan,)
    )
    branch = next(
        value for value in hypothetical.branches.values() if not value.is_factual
    )
    assert branch.identification is IdentificationStatus.COUNTERFACTUAL_HYPOTHESIS
    assert "No complete intervention-edge support" in branch.assumptions[-1]
    with pytest.raises(ValueError, match="requires intervention CausalEdges"):
        ForecastBranch(
            prediction=_prediction(inputs),
            identification=IdentificationStatus.GRAPH_SUPPORTED_INTERVENTION,
            action_identity="a",
            model_identity="m",
            profile_identity="p",
            route_identity="r",
            graph_identity="g",
            route_w1=0.0,
            atlas_citations=(_receipt(),),
            parent_identity=hypothetical.factual.branch_identity,
            factual_identity=hypothetical.factual.branch_identity,
            intervention_identity=supported_plan.identity,
        )


def test_branch_set_alignment_crossing_deltas_envelopes_and_extremes() -> None:
    inputs = _inputs()
    plan = InterventionPlan(
        "raise",
        (_spec(inputs, values=(6.0, 7.0, 8.0)),),
    )
    branches = BranchExecutor(lambda value: _execution(value)).execute(inputs, (plan,))
    scenario = next(
        branch for branch in branches.branches.values() if not branch.is_factual
    )
    np.testing.assert_array_equal(branches.delta(scenario), np.full((3, 1), 2.0))
    envelope = branches.effect_envelope(scenario)
    np.testing.assert_array_equal(envelope.lower, np.full((3, 1), -0.5))
    np.testing.assert_array_equal(envelope.upper, np.full((3, 1), 4.5))
    extremes = branches.best_worst_target_trajectories(0)
    assert extremes.best.branch_identity == scenario.branch_identity
    assert extremes.worst.branch_identity == branches.factual.branch_identity
    assert extremes.best.quantiles is not None
    assert extremes.best.quantiles.shape == (3, 3)
    assert extremes.best.joint_lower.shape == (3,)

    with pytest.raises(ValueError, match="non-decreasing"):
        ForecastPrediction(
            series_id=inputs.series_id,
            timestamps=inputs.future_timestamps,
            point=np.zeros((3, 1), dtype=np.float32),
            quantiles=np.asarray([[[1.0, 0.0]]] * 3, dtype=np.float32),
            quantile_levels=(0.1, 0.9),
        )

    shifted_inputs = ForecastInputs(
        series_id=inputs.series_id,
        horizon=inputs.horizon,
        timestamps=inputs.timestamps + np.timedelta64(1, "D"),
        past_target=inputs.past_target,
        past_covariates=inputs.past_covariates,
        known_covariates=inputs.known_covariates,
        known_covariate_mask=inputs.known_covariate_mask,
    )
    unaligned_execution = _execution(shifted_inputs)
    unaligned = ForecastBranch(
        prediction=unaligned_execution.prediction,
        identification=IdentificationStatus.COUNTERFACTUAL_HYPOTHESIS,
        action_identity=unaligned_execution.action_identity,
        model_identity=unaligned_execution.model_identity,
        profile_identity=unaligned_execution.profile_identity,
        route_identity=unaligned_execution.route_identity,
        graph_identity=unaligned_execution.graph_identity,
        route_w1=unaligned_execution.route_w1,
        atlas_citations=unaligned_execution.atlas_citations,
        parent_identity=branches.factual.branch_identity,
        factual_identity=branches.factual.branch_identity,
        intervention_identity=plan.identity,
        assumptions=("timestamp hypothesis",),
    )
    with pytest.raises(ValueError, match="not aligned"):
        BranchSet((branches.factual, unaligned))


def test_branch_set_and_executor_are_deterministic_and_keep_direct_citations() -> None:
    inputs = _inputs()
    plans = (
        InterventionPlan("z", (_spec(inputs, name="z", values=(8.0, 8.0, 8.0)),)),
        InterventionPlan("a", (_spec(inputs, name="a", values=(7.0, 7.0, 7.0)),)),
    )
    executor = BranchExecutor(lambda value: _execution(value, edges=(_edge(),)))
    first = executor.execute(inputs, plans)
    second = executor.execute(inputs, tuple(reversed(plans)))
    assert tuple(first.branches) == tuple(second.branches)
    assert first.factual.branch_identity == second.factual.branch_identity
    assert all(
        branch.atlas_citations == (_receipt(),) for branch in first.branches.values()
    )
    with pytest.raises(TypeError):
        first.branches["new"] = first.factual  # type: ignore[index]
    with pytest.raises(ValueError):
        first.delta(next(iter(first.branches)))[0, 0] = 2.0
