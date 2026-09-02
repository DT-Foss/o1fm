"""Real Toto branch-cartography proof with paired counterfactual outcomes."""

from __future__ import annotations

import hashlib
import platform
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .action_bank import ActionBank
from .atlas import AtlasReceipt, ForecastAtlas
from .branch_runtime import TeacherBranchProvenance, TeacherBranchRuntime
from .branches import (
    BranchSet,
    InterventionPlan,
    InterventionSpec,
    JointEnvelope,
    apply_intervention,
)
from .cartography import (
    ZERO_SHA256,
    CartographerBudgets,
    ForecastCartographer,
    ForecastProbeTarget,
)
from .cartography_runtime import CartographyExecutionBridge, ProbeWindow
from .causal_graph import CausalGraphConfig, OnlineCausalGraph
from .contracts import ForecastInputs, ForecastPrediction, ForecastTarget
from .crystals import ForecastCrystalBank
from .identity import sha256_bytes, sha256_json
from .metrics import ForecastScore, score_prediction
from .o1 import O1Config, O1ErrorStream
from .runtime import ForecastTicket, O1FMRuntime
from .synthetic import SyntheticConfig, SyntheticDataset, generate_synthetic_dataset
from .teachers import Chronos2Teacher, ForecastTeacher, Toto2Teacher


def _module_sha256(module: object) -> str | None:
    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        return None
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_teacher(
    teacher_name: str,
    *,
    checkpoint: str | None,
    device: str,
) -> ForecastTeacher:
    if teacher_name == "toto2":
        return Toto2Teacher.from_pretrained(
            "Datadog/Toto-2.0-22m" if checkpoint is None else checkpoint,
            device=device,
        )
    if teacher_name == "chronos2":
        return Chronos2Teacher.from_pretrained(
            "amazon/chronos-2" if checkpoint is None else checkpoint,
            device=device,
        )
    raise ValueError("teacher_name must be toto2 or chronos2")


def _array_sha256(value: object) -> str:
    array = np.asarray(value)
    canonical = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return sha256_bytes(canonical.tobytes(order="C"))


def _window_identity(inputs: ForecastInputs) -> str:
    return sha256_json(
        {
            "known_covariates_sha256": _array_sha256(inputs.known_covariates),
            "known_mask_sha256": _array_sha256(inputs.known_covariate_mask),
            "past_covariates_sha256": _array_sha256(inputs.past_covariates),
            "past_covariate_mask_sha256": _array_sha256(inputs.past_covariate_mask),
            "past_target_mask_sha256": _array_sha256(inputs.past_target_mask),
            "past_target_sha256": _array_sha256(inputs.past_target),
            "schema": "o1fm.cartography-demo-window/v1",
            "series_id": inputs.series_id,
            "timestamps_sha256": _array_sha256(inputs.timestamps.astype(np.int64)),
        }
    )


def _plans(inputs: ForecastInputs) -> tuple[InterventionPlan, InterventionPlan]:
    supported = InterventionPlan(
        "seasonal-driver-shift",
        (
            InterventionSpec(
                name="shift-known-seasonal-driver",
                operation="shift",
                timestamps=inputs.future_timestamps,
                covariate_indices=(0,),
                values=np.full((inputs.horizon, 1), 0.5, dtype=np.float32),
                lower_bounds=(-2.0,),
                upper_bounds=(2.0,),
            ),
        ),
        assumptions=("Paired structural replay preserves the exogenous noise path",),
    )
    intervention_off = InterventionPlan(
        "remove-scheduled-intervention",
        (
            InterventionSpec(
                name="set-intervention-channel-off",
                operation="set",
                timestamps=inputs.future_timestamps,
                covariate_indices=(2,),
                values=np.zeros((inputs.horizon, 1), dtype=np.float32),
                lower_bounds=(0.0,),
                upper_bounds=(1.0,),
            ),
        ),
        assumptions=("Paired structural replay preserves the exogenous noise path",),
    )
    return supported, intervention_off


def _joint(prediction: ForecastPrediction) -> JointEnvelope:
    if prediction.quantiles is None:
        lower = prediction.point
        upper = prediction.point
    else:
        lower = prediction.quantiles[..., 0]
        upper = prediction.quantiles[..., -1]
    return JointEnvelope(
        series_id=prediction.series_id,
        timestamps=prediction.timestamps,
        lower=lower,
        upper=upper,
    )


def _citations(ticket: ForecastTicket, settlement: object) -> tuple[AtlasReceipt, ...]:
    values = (
        ticket.atlas_forecast_receipt,
        ticket.atlas_transport_receipt,
        getattr(settlement, "atlas_settlement_receipt", None),
    )
    result = tuple(value for value in values if isinstance(value, AtlasReceipt))
    if len(result) != 3:
        raise RuntimeError("branch cartography requires three direct Atlas citations")
    return result


def _score_record(score: ForecastScore) -> dict[str, float | int | None]:
    return {
        "coverage": score.interval_coverage,
        "mae": score.mae,
        "mean_width": score.interval_width,
        "observed_values": score.observed_values,
        "pinball": score.pinball,
    }


def _effect_mae(
    branch_point: np.ndarray,
    factual_point: np.ndarray,
    branch_target: ForecastTarget,
    factual_target: ForecastTarget,
) -> float:
    observed = branch_target.mask & factual_target.mask
    predicted = np.asarray(branch_point) - np.asarray(factual_point)
    realized = np.asarray(branch_target.values) - np.asarray(factual_target.values)
    return float(np.mean(np.abs(predicted[observed] - realized[observed])))


def _branch_records(
    branches: BranchSet,
    dataset: SyntheticDataset,
    inputs: ForecastInputs,
    plans: tuple[InterventionPlan, ...],
) -> dict[str, object]:
    plans_by_identity = {plan.identity: plan for plan in plans}
    factual_target = dataset.counterfactual_target(inputs)
    factual = branches.factual
    records: dict[str, object] = {}
    for branch in branches.branches.values():
        if branch.is_factual:
            target = factual_target
            name = "factual"
            delta = np.zeros_like(branch.prediction.point)
            effect_mae = 0.0
            mean_absolute_oracle_effect = 0.0
            effect_recovery = 1.0
            directional_cosine = 1.0
            envelope = None
        else:
            assert branch.intervention_identity is not None
            plan = plans_by_identity[branch.intervention_identity]
            target = dataset.counterfactual_target(apply_intervention(inputs, plan))
            name = plan.name
            delta = branches.delta(branch)
            effect_mae = _effect_mae(
                branch.prediction.point,
                factual.prediction.point,
                target,
                factual_target,
            )
            observed = target.mask & factual_target.mask
            oracle_delta = np.asarray(target.values) - np.asarray(factual_target.values)
            mean_absolute_oracle_effect = float(np.mean(np.abs(oracle_delta[observed])))
            mean_absolute_predicted = float(np.mean(np.abs(delta[observed])))
            effect_recovery = (
                0.0
                if mean_absolute_oracle_effect == 0.0
                else mean_absolute_predicted / mean_absolute_oracle_effect
            )
            predicted_vector = np.asarray(delta[observed], dtype=np.float64)
            oracle_vector = np.asarray(oracle_delta[observed], dtype=np.float64)
            denominator = float(
                np.linalg.norm(predicted_vector) * np.linalg.norm(oracle_vector)
            )
            directional_cosine = (
                0.0
                if denominator == 0.0
                else float(predicted_vector @ oracle_vector) / denominator
            )
            effect = branches.effect_envelope(branch)
            envelope = {
                "lower_mean": float(np.mean(effect.lower)),
                "upper_mean": float(np.mean(effect.upper)),
            }
        records[name] = {
            "assumptions": list(branch.assumptions),
            "branch_sha256": branch.branch_identity,
            "effect_envelope": envelope,
            "effect_mae": effect_mae,
            "effect_recovery": effect_recovery,
            "effect_direction_cosine": directional_cosine,
            "identification": branch.identification.value,
            "intervention_sha256": branch.intervention_identity,
            "max_absolute_predicted_delta": float(np.max(np.abs(delta))),
            "mean_absolute_oracle_effect": mean_absolute_oracle_effect,
            "mean_absolute_predicted_delta": float(np.mean(np.abs(delta))),
            "mean_predicted_delta": float(np.mean(delta)),
            "oracle_score": _score_record(score_prediction(branch.prediction, target)),
            "supporting_edges": [edge.to_record() for edge in branch.supporting_edges],
        }
    return records


def _candidate_features(
    candidates: tuple[ForecastProbeTarget, ...],
    intervention_order: tuple[str, ...],
    target_dim: int,
    max_horizon: int,
) -> dict[ForecastProbeTarget, np.ndarray]:
    result: dict[ForecastProbeTarget, np.ndarray] = {}
    for candidate in candidates:
        horizon = candidate.horizon / max_horizon
        target_vector = np.zeros(target_dim, dtype=np.float64)
        target_vector[list(candidate.target_indices)] = 1.0
        intervention_vector = np.zeros(len(intervention_order), dtype=np.float64)
        intervention_vector[
            intervention_order.index(candidate.intervention_identity_sha256)
        ] = 4.0
        result[candidate] = np.concatenate(
            (
                np.asarray(
                    (
                        1.0,
                        horizon,
                        horizon * horizon,
                        len(candidate.target_indices) / target_dim,
                    ),
                    dtype=np.float64,
                ),
                target_vector,
                intervention_vector,
            )
        )
    return result


def _cartography_targets(
    *,
    inputs: ForecastInputs,
    model_identity: str,
    graph_topology: str,
    regime: str,
    intervention_order: tuple[str, ...],
    seed: int,
) -> tuple[ForecastProbeTarget, ...]:
    window_identity = _window_identity(inputs)
    horizons = tuple(
        sorted({min(4, inputs.horizon), min(8, inputs.horizon), inputs.horizon})
    )
    groups = tuple((index,) for index in range(inputs.target_dim)) + (
        tuple(range(inputs.target_dim)),
    )
    return tuple(
        ForecastProbeTarget(
            series_id=inputs.series_id,
            dataset_identity_sha256=window_identity,
            model_identity_sha256=model_identity,
            graph_topology_sha256=graph_topology,
            regime=regime,
            horizon=horizon,
            target_indices=group,
            action_inventory=("foundation",),
            intervention_identity_sha256=intervention,
            seed=seed,
            expected_work=1.0,
            expected_bytes=200_000,
        )
        for intervention in intervention_order
        for horizon in horizons
        for group in groups
    )


def run_cartography_demo(
    *,
    teacher_name: str = "chronos2",
    checkpoint: str | None = None,
    atlas_dir: str | Path,
    device: str = "cpu",
    seed: int = 23,
    state_dir: str | Path | None = None,
    selected_jobs: int = 12,
) -> dict[str, object]:
    """Run factual and paired intervention cartography through a real teacher."""

    if selected_jobs <= 0:
        raise ValueError("selected_jobs must be positive")
    if teacher_name == "toto2":
        raise ValueError(
            "toto2 does not support known-future intervention branches; "
            "use chronos2 or native O1FM"
        )
    config = SyntheticConfig(
        seed=seed,
        total_steps=512,
        train_fraction=0.25,
        missing_rate=0.03,
        history_length=64,
        horizon=16,
        train_stride=16,
        online_stride=16,
    )
    dataset = generate_synthetic_dataset(config)
    series_id = dataset.series[0].series_id
    example = next(row for row in dataset.online if row.series_id == series_id)
    inputs = example.inputs
    factual_target = dataset.counterfactual_target(inputs)
    plans = _plans(inputs)

    load_started = time.perf_counter()
    teacher = _load_teacher(teacher_name, checkpoint=checkpoint, device=device)
    load_seconds = time.perf_counter() - load_started
    if not teacher.capabilities.known_future_covariates:
        raise ValueError(
            f"{teacher_name} cannot execute known-future intervention branches; "
            "use chronos2 or native O1FM"
        )
    model = getattr(teacher, "model", None)
    if model is None:
        model = getattr(getattr(teacher, "pipeline", None), "model", None)
    weights_sha256 = _module_sha256(model) or teacher.identity
    parameter_count = (
        None
        if model is None or not callable(getattr(model, "parameters", None))
        else sum(parameter.numel() for parameter in model.parameters())
    )

    temporary = tempfile.TemporaryDirectory() if state_dir is None else None
    context = temporary if temporary is not None else nullcontext()
    with context:
        root = (
            Path(temporary.name)
            if temporary is not None
            else Path(state_dir).expanduser().absolute()
        )
        root.mkdir(parents=True, exist_ok=True)
        atlas = ForecastAtlas.from_immer(atlas_dir)
        runtime = O1FMRuntime(
            teacher,
            model_identity=weights_sha256,
            o1=O1ErrorStream(
                root / "authority" / "o1.json",
                stream_id=inputs.series_id,
                config=O1Config(quantiles=(0.1, 0.5, 0.9), cusum_threshold=4.0),
            ),
            graph=OnlineCausalGraph(
                root / "authority" / "graph",
                graph_id=inputs.series_id,
                config=CausalGraphConfig(
                    min_observations=4,
                    max_lag=4,
                    top_k=32,
                    min_score=0.01,
                ),
            ),
            crystals=ForecastCrystalBank(
                root / "authority" / "crystals.json",
                quantile_levels=teacher.quantile_levels,
                min_support=4,
                min_sequential_support=2,
            ),
            actions=ActionBank(root / "authority" / "actions.json"),
            atlas=atlas,
            route_top_k=32,
            enable_seasonal=False,
        )
        authority_ticket = runtime.forecast(inputs)
        authority_settlement = runtime.settle(authority_ticket, factual_target)
        citations = _citations(authority_ticket, authority_settlement)
        profile_identity = sha256_json(authority_ticket.profile.to_document())

        def provenance(
            _inputs: ForecastInputs,
            prediction: ForecastPrediction,
        ) -> TeacherBranchProvenance:
            return TeacherBranchProvenance(
                action_identity="foundation",
                model_identity=weights_sha256,
                profile_identity=profile_identity,
                route_identity=authority_ticket.route.signature_sha256,
                graph_identity=authority_ticket.profile.graph_revision,
                route_w1=authority_ticket.route.transport_w1,
                atlas_citations=citations,
                joint_envelope=_joint(prediction),
                route_edges=authority_ticket.route.edges,
                assumptions=(
                    "Known-future schedule is explicit",
                    "Counterfactual quality is scored on paired structural replay",
                ),
            )

        branch_runtime = TeacherBranchRuntime(
            teacher,
            provenance,
            known_covariate_names=tuple(
                f"known:{index}" for index in range(inputs.known_covariate_dim)
            ),
            target_names=tuple(f"target:{index}" for index in range(inputs.target_dim)),
        )
        branch_started = time.perf_counter()
        branches = branch_runtime.execute(inputs, plans)
        branch_seconds = time.perf_counter() - branch_started
        branch_records = _branch_records(branches, dataset, inputs, plans)

        intervention_order = (ZERO_SHA256, *(plan.identity for plan in plans))
        candidates = _cartography_targets(
            inputs=inputs,
            model_identity=weights_sha256,
            graph_topology=authority_ticket.route.signature_sha256,
            regime=authority_ticket.regime,
            intervention_order=intervention_order,
            seed=seed,
        )
        selector = ForecastCartographer(root / "selector")
        selector.extend_targets(candidates)
        feature_vectors = _candidate_features(
            candidates,
            intervention_order,
            inputs.target_dim,
            inputs.horizon,
        )
        selected = selector.d_optimal_batch(
            feature_vectors,
            batch_size=min(selected_jobs, len(candidates)),
        )
        selected_scenarios = {
            target.intervention_identity_sha256 for target in selected
        }
        if selected_scenarios != set(intervention_order):
            raise RuntimeError("D-optimal acquisition omitted an intervention family")

        plans_by_identity = {plan.identity: plan for plan in plans}

        def resolve_window(target: ForecastProbeTarget) -> ProbeWindow:
            plan = plans_by_identity.get(target.intervention_identity_sha256)
            realized_inputs = (
                inputs if plan is None else apply_intervention(inputs, plan)
            )
            realized = dataset.counterfactual_target(realized_inputs)
            return ProbeWindow(
                inputs=inputs,
                target=realized,
                interventions=plans,
                realized_intervention_identity_sha256=(
                    ZERO_SHA256 if plan is None else plan.identity
                ),
            )

        bridge = CartographyExecutionBridge(
            resolve_window,
            branch_runtime.forecast,
            nominal_coverage=0.8,
            default_work_cost=1.0,
            strict_byte_reservations=True,
        )
        bridge.register(selected)
        cartographer = ForecastCartographer(
            root / "cartographer",
            budgets=CartographerBudgets(
                work=float(len(selected)),
                bytes=200_000 * len(selected),
            ),
        )
        cartographer.extend_targets(selected)
        cartography_started = time.perf_counter()
        receipts = cartographer.run(bridge, max_jobs=len(selected))
        cartography_seconds = time.perf_counter() - cartography_started
        bridge_metrics = dict(bridge.metrics())
        cartography_metrics = cartographer.metrics()
        cartographer.close()
        if len(receipts) != len(selected):
            raise RuntimeError("cartographer did not settle every acquired job")
        physical_forwards = int(bridge_metrics["forward_count"])
        if physical_forwards >= len(receipts):
            raise RuntimeError("cartography did not coalesce logical probe cells")

        receipt_records = [
            {
                "atlas_sha256": receipt.atlas_sha256,
                "coverage_gap": receipt.coverage_gap,
                "information_gain": receipt.information_gain,
                "intervention_sha256": next(
                    target.intervention_identity_sha256
                    for target in selected
                    if target.sha256 == receipt.target_sha256
                ),
                "mae": receipt.mae,
                "pinball": receipt.pinball,
                "receipt_sha256": receipt.sha256,
                "result_sha256": receipt.result_sha256,
                "route_w1": receipt.route_w1,
                "target_sha256": receipt.target_sha256,
                "work_cost": receipt.work_cost,
            }
            for receipt in receipts
        ]
        atlas_receipts = [
            {
                "kind": receipt.kind,
                "revision_sequence": receipt.revision_sequence,
                "revision_sha256": receipt.revision_sha256,
                "segment_sha256": receipt.segment_sha256,
            }
            for receipt in citations
        ]
        remounted_atlas = ForecastAtlas.from_immer(atlas_dir)
        result: dict[str, object] = {
            "atlas": {
                "receipts": atlas_receipts,
                "revision": list(atlas.store.revision()),
                "remount_revision": list(remounted_atlas.store.revision()),
                "remount_verified": remounted_atlas.verify(),
                "verified": atlas.verify(),
            },
            "authority": {
                "action": authority_ticket.action,
                "fragment_sha256": authority_settlement.fragment_sha256,
                "route_edges": len(authority_ticket.route.edges),
                "route_topology_sha256": authority_ticket.route.signature_sha256,
                "route_w1": authority_ticket.route.transport_w1,
                "score": _score_record(authority_settlement.score),
            },
            "branches": branch_records,
            "cartography": {
                "candidate_cells": len(candidates),
                "cartographer": cartography_metrics,
                "coalescing_reduction": 1.0 - physical_forwards / len(receipts),
                "execution": bridge_metrics,
                "logical_cells": len(receipts),
                "physical_forwards": physical_forwards,
                "receipts": receipt_records,
                "selected_intervention_families": len(selected_scenarios),
            },
            "data": {
                "factual_target_sha256": _array_sha256(factual_target.values),
                "history": inputs.history_length,
                "horizon": inputs.horizon,
                "known_covariates": inputs.known_covariate_dim,
                "series_id": inputs.series_id,
                "target_count": inputs.target_dim,
                "window_sha256": _window_identity(inputs),
            },
            "environment": {
                "device": device,
                "machine": platform.machine(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
            },
            "schema": "o1fm.counterfactual-cartography-demo/v1",
            "teacher": {
                "adapter_identity_sha256": teacher.identity,
                "capabilities": teacher.capabilities.to_record(),
                "checkpoint": teacher.checkpoint,
                "load_seconds": load_seconds,
                "parameter_count": parameter_count,
                "weights_sha256": weights_sha256,
            },
            "timings": {
                "branch_seconds": branch_seconds,
                "cartography_seconds": cartography_seconds,
            },
        }
        atlas_result = result["atlas"]
        assert isinstance(atlas_result, dict)
        if not atlas_result["verified"] or not atlas_result["remount_verified"]:
            raise RuntimeError("counterfactual cartography Atlas failed verification")
        return result


__all__ = ["run_cartography_demo"]
