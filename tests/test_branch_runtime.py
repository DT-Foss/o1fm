from __future__ import annotations

import numpy as np
import pytest
import torch

from o1fm.atlas import AtlasReceipt
from o1fm.branch_runtime import TeacherBranchProvenance, TeacherBranchRuntime
from o1fm.branches import (
    IdentificationStatus,
    InterventionPlan,
    InterventionSpec,
    JointEnvelope,
)
from o1fm.causal_graph import CausalEdge
from o1fm.contracts import ForecastInputs, ForecastPrediction
from o1fm.model import DEFAULT_QUANTILES
from o1fm.teachers import (
    Chronos2Teacher,
    ForecastTeacherCapabilities,
    NativeO1FMTeacher,
    Toto2Teacher,
)


def _inputs() -> ForecastInputs:
    history = 4
    horizon = 3
    timestamps = np.datetime64("2026-01-01") + np.arange(history + horizon).astype(
        "timedelta64[D]"
    )
    known = np.stack(
        (
            np.arange(history + horizon, dtype=np.float32),
            np.full(history + horizon, 2.0, dtype=np.float32),
        ),
        axis=1,
    )
    return ForecastInputs(
        series_id="grid-a",
        horizon=horizon,
        timestamps=timestamps,
        past_target=np.arange(history, dtype=np.float32)[:, None],
        past_covariates=np.full((history, 1), 5.0, dtype=np.float32),
        known_covariates=known,
    )


def _receipt(seed: str = "a", sequence: int = 1) -> AtlasReceipt:
    return AtlasReceipt(
        segment_sha256=seed * 64,
        revision_sequence=sequence,
        revision_sha256="f" * 64,
        record_count=1,
        kind="forecast",
    )


def _edge(source: str = "price") -> CausalEdge:
    return CausalEdge(
        source=source,
        target="demand",
        lag=0,
        kind="intervention",
        observations=32,
        covariance=0.7,
        source_variance=1.0,
        target_variance=1.0,
        score=0.7,
        revision_sha256="e" * 64,
    )


def _plan(
    inputs: ForecastInputs,
    *,
    name: str,
    covariate: int,
    value: float,
    observed: bool = False,
) -> InterventionPlan:
    return InterventionPlan(
        name,
        (
            InterventionSpec(
                name=f"{name}-edit",
                operation="set",
                timestamps=inputs.future_timestamps,
                covariate_indices=(covariate,),
                values=np.full((inputs.horizon, 1), value, dtype=np.float32),
                lower_bounds=(-100.0,),
                upper_bounds=(100.0,),
            ),
        ),
        observed_policy=observed,
        assumptions=(f"{name} schedule remains fixed",),
    )


class _DeterministicTeacher:
    quantile_levels = (0.1, 0.5, 0.9)
    identity = "teacher-model-v1"
    capabilities = ForecastTeacherCapabilities(True, True, True, True)

    def __init__(self, *, mutate_private_copy: bool = False) -> None:
        self.calls: list[ForecastInputs] = []
        self.mutate_private_copy = mutate_private_copy

    def predict(self, inputs: ForecastInputs) -> ForecastPrediction:
        self.calls.append(inputs)
        future = inputs.known_covariates[inputs.history_length :]
        point = (future[:, :1] - 0.5 * future[:, 1:2]).astype(np.float32)
        if self.mutate_private_copy:
            inputs.known_covariates.setflags(write=True)
            inputs.known_covariates[0, 0] = np.float32(999.0)
        offsets = np.asarray((-1.0, 0.0, 1.0), dtype=np.float32)
        return ForecastPrediction(
            series_id=inputs.series_id,
            timestamps=inputs.future_timestamps,
            point=point,
            quantiles=point[:, :, None] + offsets,
            quantile_levels=self.quantile_levels,
        )


def test_runtime_executes_real_teacher_branches_with_explicit_identification() -> None:
    inputs = _inputs()
    teacher = _DeterministicTeacher()
    supported = _plan(inputs, name="supported", covariate=0, value=10.0)
    observed = _plan(
        inputs,
        name="observed",
        covariate=0,
        value=8.0,
        observed=True,
    )
    hypothesis = _plan(inputs, name="hypothesis", covariate=1, value=5.0)

    def provenance(
        branch_inputs: ForecastInputs,
        prediction: ForecastPrediction,
    ) -> TeacherBranchProvenance:
        changed = branch_inputs.known_covariates[branch_inputs.history_length :, 0]
        revision = int(round(float(changed[0]))) + 1
        return TeacherBranchProvenance(
            action_identity="foundation",
            profile_identity="profile:grid-a",
            route_identity=f"route:{revision}",
            graph_identity="graph:42",
            route_w1=0.25,
            atlas_citations=(_receipt(format(revision % 16, "x"), revision),),
            route_edges=(_edge(),),
            joint_envelope=JointEnvelope(
                prediction.series_id,
                prediction.timestamps,
                prediction.quantiles[:, :, 0] - 0.25,
                prediction.quantiles[:, :, -1] + 0.25,
            ),
            assumptions=("Teacher checkpoint is fixed",),
        )

    runtime = TeacherBranchRuntime(
        teacher,
        provenance,
        known_covariate_names=("price", "promotion"),
        target_names=("demand",),
    )
    result = runtime.execute(inputs, (hypothesis, observed, supported))

    assert len(teacher.calls) == 4
    assert runtime.model_identity == teacher.identity
    statuses = {branch.identification for branch in result.branches.values()}
    assert statuses == {
        IdentificationStatus.FACTUAL,
        IdentificationStatus.GRAPH_SUPPORTED_INTERVENTION,
        IdentificationStatus.OBSERVED_POLICY_INTERVENTION,
        IdentificationStatus.COUNTERFACTUAL_HYPOTHESIS,
    }
    supported_branch = next(
        branch
        for branch in result.branches.values()
        if branch.identification is IdentificationStatus.GRAPH_SUPPORTED_INTERVENTION
    )
    assert supported_branch.model_identity == teacher.identity
    assert supported_branch.supporting_edges == (_edge(),)
    assert supported_branch.atlas_citations[0].revision_sequence == 11
    hypothetical_branch = next(
        branch
        for branch in result.branches.values()
        if branch.identification is IdentificationStatus.COUNTERFACTUAL_HYPOTHESIS
    )
    assert (
        "No complete intervention-edge support" in hypothetical_branch.assumptions[-1]
    )
    np.testing.assert_array_equal(
        result.delta(supported_branch),
        np.asarray([[6.0], [5.0], [4.0]], dtype=np.float32),
    )


def test_runtime_never_exposes_callers_arrays_to_the_teacher() -> None:
    inputs = _inputs()
    original = {
        name: np.array(getattr(inputs, name), copy=True)
        for name in (
            "timestamps",
            "past_target",
            "past_target_mask",
            "past_covariates",
            "past_covariate_mask",
            "known_covariates",
            "known_covariate_mask",
        )
    }
    teacher = _DeterministicTeacher(mutate_private_copy=True)
    provenance = TeacherBranchProvenance(
        action_identity="foundation",
        model_identity="exact-weight-sha256",
        profile_identity="profile-v1",
        route_identity="route-v1",
        graph_identity="graph-v1",
        atlas_citations=(_receipt(),),
    )
    runtime = TeacherBranchRuntime(teacher, provenance)

    result = runtime.execute(
        inputs, (_plan(inputs, name="raise", covariate=0, value=9),)
    )

    assert len(result.branches) == 2
    assert all(call is not inputs for call in teacher.calls)
    assert all(
        not np.shares_memory(call.known_covariates, inputs.known_covariates)
        for call in teacher.calls
    )
    for name, expected in original.items():
        np.testing.assert_array_equal(getattr(inputs, name), expected)
    assert all(
        branch.model_identity == "exact-weight-sha256"
        for branch in result.branches.values()
    )


def test_runtime_rejects_past_edits_bad_protocols_and_bad_provenance() -> None:
    inputs = _inputs()
    provenance = TeacherBranchProvenance(
        action_identity="foundation",
        profile_identity="profile-v1",
        route_identity="route-v1",
        graph_identity="graph-v1",
        atlas_citations=(_receipt(),),
    )
    runtime = TeacherBranchRuntime(_DeterministicTeacher(), provenance)
    past_edit = InterventionSpec(
        name="past-edit",
        operation="set",
        timestamps=inputs.past_timestamps[-1:],
        covariate_indices=(0,),
        values=((100.0,),),
    )
    with pytest.raises(ValueError, match="exact forecast horizon"):
        runtime.execute(inputs, (InterventionPlan("illegal", (past_edit,)),))

    with pytest.raises(TypeError, match="ForecastTeacher"):
        TeacherBranchRuntime(object(), provenance)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="direct AtlasReceipt"):
        TeacherBranchProvenance(
            action_identity="foundation",
            profile_identity="profile-v1",
            route_identity="route-v1",
            graph_identity="graph-v1",
            atlas_citations=(),
        )
    bad_provider = TeacherBranchRuntime(
        _DeterministicTeacher(),
        lambda inputs, prediction: object(),  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(TypeError, match="must return TeacherBranchProvenance"):
        bad_provider.forecast(inputs)


def test_runtime_fallback_model_identity_is_deterministic() -> None:
    class TeacherWithoutIdentity(_DeterministicTeacher):
        identity = None
        adapter = "deterministic/v1"
        checkpoint = "weights-a"

    provenance = TeacherBranchProvenance(
        action_identity="foundation",
        profile_identity="profile-v1",
        route_identity="route-v1",
        graph_identity="graph-v1",
        atlas_citations=(_receipt(),),
    )
    first = TeacherBranchRuntime(TeacherWithoutIdentity(), provenance)
    second = TeacherBranchRuntime(TeacherWithoutIdentity(), provenance)
    assert first.model_identity == second.model_identity
    assert len(first.model_identity) == 64


def test_toto_capability_rejects_intervention_before_any_forward() -> None:
    class FakeToto:
        def __init__(self) -> None:
            self.calls = 0

        def forecast(
            self,
            payload: dict[str, torch.Tensor],
            horizon: int,
            **_: object,
        ) -> torch.Tensor:
            self.calls += 1
            variates = payload["target"].shape[1]
            result = torch.empty(
                (len(DEFAULT_QUANTILES), 1, variates, horizon),
                dtype=torch.float32,
            )
            for index in range(len(DEFAULT_QUANTILES)):
                result[index].fill_(float(index))
            return result

    inputs = _inputs()
    model = FakeToto()
    provenance = TeacherBranchProvenance(
        action_identity="foundation",
        profile_identity="profile-v1",
        route_identity="route-v1",
        graph_identity="graph-v1",
        atlas_citations=(_receipt(),),
    )
    runtime = TeacherBranchRuntime(Toto2Teacher(model), provenance)
    plan = _plan(inputs, name="unsupported", covariate=0, value=9.0)
    with pytest.raises(ValueError, match="does not support exogenous known-future"):
        runtime.execute(inputs, (plan,))
    assert model.calls == 0

    factual = runtime.execute(inputs)
    assert len(factual.branches) == 1
    assert factual.factual.is_factual
    assert model.calls == 1


@pytest.mark.parametrize(
    "capabilities",
    (NativeO1FMTeacher.capabilities, Chronos2Teacher.capabilities),
)
def test_known_future_capable_teachers_execute_intervention_branches(
    capabilities: ForecastTeacherCapabilities,
) -> None:
    class CapableTeacher(_DeterministicTeacher):
        pass

    CapableTeacher.capabilities = capabilities
    inputs = _inputs()
    teacher = CapableTeacher()
    provenance = TeacherBranchProvenance(
        action_identity="foundation",
        profile_identity="profile-v1",
        route_identity="route-v1",
        graph_identity="graph-v1",
        atlas_citations=(_receipt(),),
        route_edges=(_edge(),),
    )
    runtime = TeacherBranchRuntime(
        teacher,
        provenance,
        known_covariate_names=("price", "promotion"),
        target_names=("demand",),
    )
    branches = runtime.execute(
        inputs,
        (_plan(inputs, name="supported", covariate=0, value=10.0),),
    )
    assert len(branches.branches) == 2
    assert len(teacher.calls) == 2


def test_branch_runtime_requires_explicit_capabilities() -> None:
    class InferenceOnly:
        quantile_levels = (0.1, 0.5, 0.9)

        def predict(self, inputs: ForecastInputs) -> ForecastPrediction:
            return _DeterministicTeacher().predict(inputs)

    provenance = TeacherBranchProvenance(
        action_identity="foundation",
        profile_identity="profile-v1",
        route_identity="route-v1",
        graph_identity="graph-v1",
        atlas_citations=(_receipt(),),
    )
    with pytest.raises(TypeError, match="teacher.capabilities"):
        TeacherBranchRuntime(InferenceOnly(), provenance)
