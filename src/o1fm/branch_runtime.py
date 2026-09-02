"""Execute intervention branches through any O1FM forecast teacher.

The adapter keeps model inference and branch provenance separate.  A teacher
only receives a private copy of :class:`~o1fm.contracts.ForecastInputs`, while
``TeacherBranchProvenance`` supplies the graph, route, and Atlas evidence that
turns the prediction into a :class:`~o1fm.branches.BranchExecutionResult`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .atlas import AtlasReceipt
from .branches import (
    BranchExecutionResult,
    BranchExecutor,
    BranchSet,
    InterventionPlan,
    JointEnvelope,
)
from .causal_graph import CausalEdge
from .contracts import ForecastInputs, ForecastPrediction
from .identity import canonical_json_bytes, sha256_json
from .teachers import ForecastTeacher

_MAX_TEXT_BYTES = 1024


def _text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > _MAX_TEXT_BYTES
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    canonical_json_bytes(value)
    return value


def _texts(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return tuple(_text(child, field=field) for child in value)


def _teacher_model_identity(teacher: ForecastTeacher) -> str:
    """Return the teacher's declared identity or a deterministic fallback."""

    declared = getattr(teacher, "identity", None)
    if isinstance(declared, str):
        return _text(declared, field="teacher identity")
    adapter = getattr(teacher, "adapter", None)
    checkpoint = getattr(teacher, "checkpoint", None)
    return sha256_json(
        {
            "adapter": (
                adapter
                if isinstance(adapter, str) and adapter
                else f"{type(teacher).__module__}.{type(teacher).__qualname__}"
            ),
            "checkpoint": checkpoint if isinstance(checkpoint, str) else None,
            "quantile_levels": [float(level) for level in teacher.quantile_levels],
            "schema": "o1fm.teacher-branch-model/v1",
        }
    )


def _copy_inputs(inputs: ForecastInputs) -> ForecastInputs:
    """Create the model-visible copy used for one isolated teacher call."""

    return ForecastInputs(
        series_id=inputs.series_id,
        horizon=inputs.horizon,
        timestamps=np.array(inputs.timestamps, copy=True),
        past_target=np.array(inputs.past_target, copy=True),
        past_target_mask=np.array(inputs.past_target_mask, copy=True),
        past_covariates=np.array(inputs.past_covariates, copy=True),
        past_covariate_mask=np.array(inputs.past_covariate_mask, copy=True),
        known_covariates=np.array(inputs.known_covariates, copy=True),
        known_covariate_mask=np.array(inputs.known_covariate_mask, copy=True),
    )


@dataclass(frozen=True, slots=True)
class TeacherBranchProvenance:
    """Execution evidence attached to exactly one teacher prediction.

    A callable provenance provider can return different graph revisions,
    routes, edges, or Atlas citations for the factual input and every edited
    input.  ``model_identity`` may be omitted to use the identity declared by
    the teacher adapter.
    """

    action_identity: str
    profile_identity: str
    route_identity: str
    graph_identity: str
    atlas_citations: Sequence[AtlasReceipt]
    route_w1: float = 0.0
    model_identity: str | None = None
    route_edges: Sequence[CausalEdge] = ()
    joint_envelope: JointEnvelope | None = None
    assumptions: Sequence[str] = ()

    def __post_init__(self) -> None:
        for field in (
            "action_identity",
            "profile_identity",
            "route_identity",
            "graph_identity",
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field=field))
        if self.model_identity is not None:
            object.__setattr__(
                self,
                "model_identity",
                _text(self.model_identity, field="model_identity"),
            )
        if (
            isinstance(self.route_w1, bool)
            or not isinstance(self.route_w1, (int, float, np.number))
            or not math.isfinite(float(self.route_w1))
            or float(self.route_w1) < 0.0
        ):
            raise ValueError("route_w1 must be finite and non-negative")
        route_w1 = float(self.route_w1)
        citations = tuple(self.atlas_citations)
        if not citations or not all(
            isinstance(receipt, AtlasReceipt) for receipt in citations
        ):
            raise TypeError("atlas_citations must contain direct AtlasReceipt evidence")
        citation_ids = {
            (receipt.segment_sha256, receipt.revision_sequence, receipt.kind)
            for receipt in citations
        }
        if len(citation_ids) != len(citations):
            raise ValueError("atlas_citations must be unique")
        edges = tuple(self.route_edges)
        if not all(isinstance(edge, CausalEdge) for edge in edges):
            raise TypeError("route_edges must contain CausalEdge instances")
        if self.joint_envelope is not None and not isinstance(
            self.joint_envelope, JointEnvelope
        ):
            raise TypeError("joint_envelope must be JointEnvelope or None")
        object.__setattr__(self, "route_w1", 0.0 if route_w1 == 0.0 else route_w1)
        object.__setattr__(self, "atlas_citations", citations)
        object.__setattr__(self, "route_edges", edges)
        object.__setattr__(
            self,
            "assumptions",
            _texts(self.assumptions, field="assumption"),
        )


@runtime_checkable
class TeacherBranchProvenanceProvider(Protocol):
    """Resolve branch-specific evidence after a teacher prediction exists."""

    def __call__(
        self,
        inputs: ForecastInputs,
        prediction: ForecastPrediction,
    ) -> TeacherBranchProvenance: ...


class TeacherBranchRuntime:
    """Run factual and future-known intervention branches through a teacher."""

    def __init__(
        self,
        teacher: ForecastTeacher,
        provenance: TeacherBranchProvenance
        | Callable[[ForecastInputs, ForecastPrediction], TeacherBranchProvenance],
        *,
        known_covariate_names: Sequence[str] | None = None,
        target_names: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(teacher, ForecastTeacher):
            raise TypeError("teacher must implement ForecastTeacher")
        if not isinstance(provenance, TeacherBranchProvenance) and not callable(
            provenance
        ):
            raise TypeError(
                "provenance must be TeacherBranchProvenance or a callable provider"
            )
        self.teacher = teacher
        self._provenance = provenance
        self.model_identity = _teacher_model_identity(teacher)
        self._executor = BranchExecutor(
            self.forecast,
            known_covariate_names=known_covariate_names,
            target_names=target_names,
        )

    def _evidence(
        self,
        inputs: ForecastInputs,
        prediction: ForecastPrediction,
    ) -> TeacherBranchProvenance:
        if isinstance(self._provenance, TeacherBranchProvenance):
            return self._provenance
        result = self._provenance(inputs, prediction)
        if not isinstance(result, TeacherBranchProvenance):
            raise TypeError("provenance provider must return TeacherBranchProvenance")
        return result

    def forecast(self, inputs: ForecastInputs) -> BranchExecutionResult:
        """Execute one isolated teacher call and attach its exact evidence."""

        if not isinstance(inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        model_inputs = _copy_inputs(inputs)
        prediction = self.teacher.predict(model_inputs)
        if not isinstance(prediction, ForecastPrediction):
            raise TypeError("teacher.predict must return ForecastPrediction")
        prediction.validate_for(inputs)
        evidence = self._evidence(_copy_inputs(inputs), prediction)
        return BranchExecutionResult(
            prediction=prediction,
            action_identity=evidence.action_identity,
            model_identity=evidence.model_identity or self.model_identity,
            profile_identity=evidence.profile_identity,
            route_identity=evidence.route_identity,
            graph_identity=evidence.graph_identity,
            route_w1=evidence.route_w1,
            atlas_citations=evidence.atlas_citations,
            joint_envelope=evidence.joint_envelope,
            route_edges=evidence.route_edges,
            assumptions=evidence.assumptions,
        )

    def execute(
        self,
        inputs: ForecastInputs,
        plans: Sequence[InterventionPlan] = (),
    ) -> BranchSet:
        """Execute the factual path plus each canonical intervention plan."""

        return self._executor.execute(inputs, plans)


__all__ = [
    "TeacherBranchProvenance",
    "TeacherBranchProvenanceProvider",
    "TeacherBranchRuntime",
]
