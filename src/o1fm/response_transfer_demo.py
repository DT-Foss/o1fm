"""State/operator/topology transfer proof for intervention response Crystals."""

from __future__ import annotations

import math
import platform
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .action_bank import ActionBank, ActionProfileKey, ActionReceipt
from .atlas import ForecastAtlas
from .branches import InterventionPlan, InterventionSpec, apply_intervention
from .cartography import (
    CartographerBudgets,
    ForecastCartographer,
    ForecastProbeTarget,
    ProbeEvidence,
)
from .cartography_demo import (
    _array_sha256,
    _load_teacher,
    _module_sha256,
    _window_identity,
)
from .causal_graph import CausalGraphConfig, OnlineCausalGraph
from .conformal import ConformalFieldConfig, O1ConformalField
from .contracts import (
    ForecastExample,
    ForecastInputs,
    ForecastPrediction,
    ForecastTarget,
)
from .crystals import ForecastCrystalBank
from .identity import sha256_json
from .intervention_compiler_demo import (
    _conformal_key,
    _effect_metrics,
    _mean,
    _settle_conformal,
)
from .intervention_crystals import (
    InterventionResponseCrystal,
    InterventionResponseKey,
)
from .metrics import ForecastScore, score_prediction
from .o1 import O1Config, O1ErrorStream
from .response_features import (
    InterventionResponseFeatureEncoder,
    ResponseFeatureSchema,
    ResponseRouteContext,
)
from .runtime import ForecastTicket, O1FMRuntime
from .synthetic import SyntheticConfig, SyntheticDataset, generate_synthetic_dataset
from .teachers import ForecastTeacher


@dataclass(frozen=True, slots=True)
class _Operator:
    family: str
    shape: str
    channel: int
    operation: str
    values: tuple[float, ...]
    mask: tuple[bool, ...]

    def plan(self, inputs: ForecastInputs, *, name: str) -> InterventionPlan:
        if len(self.values) != inputs.horizon or len(self.mask) != inputs.horizon:
            raise ValueError("operator differs from the forecast horizon")
        return InterventionPlan(
            name,
            (
                InterventionSpec(
                    name=f"{name}-cells",
                    operation=self.operation,
                    timestamps=inputs.future_timestamps,
                    covariate_indices=(self.channel,),
                    values=np.asarray(self.values, dtype=np.float32)[:, None],
                    edit_mask=np.asarray(self.mask, dtype=np.bool_)[:, None],
                    lower_bounds=(-3.0,),
                    upper_bounds=(3.0,),
                ),
            ),
            assumptions=("Same-noise structural response",),
        )


@dataclass(frozen=True, slots=True)
class _Probe:
    example: ForecastExample
    operator: _Operator
    plan: InterventionPlan
    features: np.ndarray
    target: ForecastProbeTarget


def _family(channel: int) -> str:
    return {0: "seasonal-shift", 2: "intervention-set", 3: "trend-placebo"}[channel]


def _training_operators(horizon: int) -> tuple[_Operator, ...]:
    operators: list[_Operator] = []
    for channel, operation in ((0, "shift"), (2, "set"), (3, "shift")):
        for index in range(horizon):
            mask = tuple(row == index for row in range(horizon))
            operand = (
                0.25
                if operation == "set" and index % 2
                else 0.75
                if operation == "set"
                else 0.75
                if index % 2 == 0
                else -0.75
            )
            values = tuple(operand if selected else 0.0 for selected in mask)
            operators.append(
                _Operator(
                    family=_family(channel),
                    shape="pulse",
                    channel=channel,
                    operation=operation,
                    values=values,
                    mask=mask,
                )
            )
        constants = (
            (0.0, 0.25, 0.5, 0.75) if operation == "set" else (-0.8, -0.3, 0.3, 0.8)
        )
        for value in constants:
            operators.append(
                _Operator(
                    family=_family(channel),
                    shape="constant",
                    channel=channel,
                    operation=operation,
                    values=(value,) * horizon,
                    mask=(True,) * horizon,
                )
            )
    return tuple(operators)


def _evaluation_operators(horizon: int, ordinal: int) -> tuple[_Operator, ...]:
    time_axis = np.linspace(0.0, 1.0, horizon, dtype=np.float64)
    seasonal_values = tuple(
        float(value)
        for value in ((-0.6 + 0.15 * ordinal) + (0.9 - 0.1 * ordinal) * time_axis)
    )
    start = 2 + ordinal
    end = min(horizon, start + 5 + ordinal % 2)
    set_mask = tuple(start <= index < end for index in range(horizon))
    set_values = tuple(
        (0.2 + 0.15 * ordinal) if selected else 0.0 for selected in set_mask
    )
    placebo_values = tuple(
        float(0.45 if index in {3 + ordinal, 11 - ordinal} else 0.0)
        for index in range(horizon)
    )
    placebo_mask = tuple(value != 0.0 for value in placebo_values)
    return (
        _Operator(
            "seasonal-shift",
            "ramp",
            0,
            "shift",
            seasonal_values,
            (True,) * horizon,
        ),
        _Operator(
            "intervention-set",
            "masked-interval",
            2,
            "set",
            set_values,
            set_mask,
        ),
        _Operator(
            "trend-placebo",
            "double-pulse",
            3,
            "shift",
            placebo_values,
            placebo_mask,
        ),
    )


def _regime_examples(
    dataset: SyntheticDataset, regime: int
) -> tuple[ForecastExample, ...]:
    series = dataset.series[0]
    ticks = series.timestamps.astype(np.int64, copy=False)
    selected: list[ForecastExample] = []
    for example in dataset.online:
        if example.series_id != series.series_id:
            continue
        origin = int(
            np.searchsorted(
                ticks,
                example.inputs.future_timestamps[0].astype(np.int64),
            )
        )
        regimes = series.regime_ids[origin : origin + example.inputs.horizon]
        if regimes.size == example.inputs.horizon and bool((regimes == regime).all()):
            selected.append(example)
    return tuple(selected)


def _authority(
    teacher: ForecastTeacher,
    example: ForecastExample,
    *,
    model_identity: str,
    root: Path,
    atlas: ForecastAtlas | None,
    name: str,
) -> tuple[ForecastTicket, ForecastTarget]:
    target = example.require_target()
    runtime = O1FMRuntime(
        teacher,
        model_identity=model_identity,
        o1=O1ErrorStream(
            root / name / "o1.json", stream_id=f"{example.series_id}:{name}"
        ),
        graph=OnlineCausalGraph(
            root / name / "graph",
            graph_id=f"{example.series_id}:{name}",
            config=CausalGraphConfig(
                decay=0.98,
                max_lag=4,
                top_k=8,
                min_observations=8,
                min_score=0.01,
            ),
        ),
        crystals=ForecastCrystalBank(
            root / name / "crystals.json",
            quantile_levels=teacher.quantile_levels,
        ),
        actions=ActionBank(root / name / "actions.json"),
        atlas=atlas,
        route_top_k=8,
        enable_seasonal=False,
    )
    ticket = runtime.forecast(example.inputs)
    runtime.settle(ticket, target)
    return ticket, target


def _schema(
    inputs: ForecastInputs,
    ticket: ForecastTicket,
    projection_scope: str,
) -> tuple[ResponseFeatureSchema, ResponseRouteContext]:
    target_names = tuple(f"target:{index}" for index in range(inputs.target_dim))
    known_names = tuple(f"known:{index}" for index in range(inputs.known_covariate_dim))
    schema = ResponseFeatureSchema(
        target_names=target_names,
        known_covariate_names=known_names,
        horizon=inputs.horizon,
        graph_topology_sha256=ticket.route.signature_sha256,
        seasonal_periods=(24,),
        state_window=64,
        mask_tail=8,
        design_dimension=128,
        projection_scope_sha256=projection_scope,
    )
    return schema, ResponseRouteContext(
        graph_topology_sha256=ticket.route.signature_sha256,
        edges=ticket.route.edges,
    )


def _response_key(
    schema: ResponseFeatureSchema,
    *,
    model_identity: str,
    regime: int,
    o1_generation: int,
    fit_scope: str,
) -> InterventionResponseKey:
    return InterventionResponseKey(
        model_identity=model_identity,
        feature_schema=sha256_json(
            {
                "control_tail": "structural-regime-plus-o1-generation/v1",
                "layout_sha256": schema.layout_identity_sha256,
                "projection_scope_sha256": schema.resolved_projection_scope_sha256,
                "schema": "o1fm.transfer-response-feature-schema/v1",
            }
        ),
        intervention_schema=sha256_json(
            {
                "known_schema_sha256": schema.known_covariate_schema_sha256,
                "operator_abi": "o1fm.intervention-response-features/v1",
            }
        ),
        target_schema=schema.target_schema_sha256,
        graph_revision=schema.graph_topology_sha256,
        regime=f"structural-regime:{regime}:o1-generation:{o1_generation}",
        horizon=schema.horizon,
        target_count=schema.target_count,
        fit_scope_sha256=fit_scope,
        fit_scope_regime="shared-response-parent",
    )


def _features(
    encoder: InterventionResponseFeatureEncoder,
    inputs: ForecastInputs,
    plan: InterventionPlan,
    route: ResponseRouteContext,
    regime: int,
    o1_generation: int,
) -> np.ndarray:
    encoded = encoder.encode(inputs, plan, route)
    result = np.concatenate(
        (
            encoded.design_vector,
            np.asarray((regime / 2.0, o1_generation / 8.0), dtype=np.float64),
        )
    )
    result.setflags(write=False)
    return result


def _profile(key: InterventionResponseKey, family: str) -> ActionProfileKey:
    return ActionProfileKey(
        model_identity=key.model_identity,
        feature_schema=key.feature_schema,
        graph_revision=key.graph_revision,
        regime=f"{key.regime}:family:{family}",
        horizon=key.horizon,
        target_count=key.target_count,
    )


def _probe_target(
    *,
    model_identity: str,
    key: InterventionResponseKey,
    example: ForecastExample,
    plan: InterventionPlan,
    ordinal: int,
) -> ForecastProbeTarget:
    return ForecastProbeTarget(
        series_id=example.series_id,
        dataset_identity_sha256=_window_identity(example.inputs),
        model_identity_sha256=model_identity,
        graph_topology_sha256=key.graph_revision,
        regime=key.regime,
        horizon=example.inputs.horizon,
        target_indices=tuple(range(example.inputs.target_dim)),
        action_inventory=("crystal", "foundation"),
        intervention_identity_sha256=plan.identity,
        seed=ordinal,
        expected_work=1.0,
        expected_bytes=100_000,
    )


def _score_summary(scores: list[ForecastScore]) -> dict[str, float | None]:
    return {
        "coverage": _mean([score.interval_coverage or 0.0 for score in scores]),
        "mae": _mean([score.mae for score in scores]),
        "pinball": _mean([score.pinball or score.mae for score in scores]),
        "width": _mean([score.interval_width or 0.0 for score in scores]),
    }


def _response_stats(
    response: InterventionResponseCrystal, key: InterventionResponseKey
) -> dict[str, object]:
    stats = response.describe(key)
    return {
        "coverage": stats.empirical_coverage,
        "direction_cosine": stats.effect_direction_cosine,
        "feature_dim": stats.feature_dim,
        "mean_effect_error": stats.mean_effect_error,
        "observed_values": stats.observed_values,
        "ready": stats.ready,
        "sequential_support": stats.sequential_support,
        "support": stats.support,
    }


def _evaluation_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    def aggregate(selected_rows: list[dict[str, object]]) -> dict[str, object]:
        raw_scores = [row["raw_score"] for row in selected_rows]
        compiled_scores = [
            row["compiled_score"]
            for row in selected_rows
            if row["compiled_score"] is not None
        ]
        selected_scores = [row["selected_score"] for row in selected_rows]
        return {
            "actions": {
                action: sum(row["action"] == action for row in selected_rows)
                for action in ("foundation", "crystal")
            },
            "compiled": _score_summary(compiled_scores),  # type: ignore[arg-type]
            "compiled_direction": _mean(
                [
                    float(row["compiled_effect"]["direction_cosine"])
                    for row in selected_rows
                    if row["compiled_effect"] is not None
                ]  # type: ignore[index]
            ),
            "compiled_recovery": _mean(
                [
                    float(row["compiled_effect"]["recovery"])
                    for row in selected_rows
                    if row["compiled_effect"] is not None
                ]  # type: ignore[index]
            ),
            "raw": _score_summary(raw_scores),  # type: ignore[arg-type]
            "raw_direction": _mean(
                [float(row["raw_effect"]["direction_cosine"]) for row in selected_rows]  # type: ignore[index]
            ),
            "raw_recovery": _mean(
                [float(row["raw_effect"]["recovery"]) for row in selected_rows]  # type: ignore[index]
            ),
            "selected": _score_summary(selected_scores),  # type: ignore[arg-type]
        }

    result = aggregate(rows)
    result["families"] = {
        family: aggregate([row for row in rows if row["family"] == family])
        for family in sorted({str(row["family"]) for row in rows})
    }
    return result


def run_response_transfer_demo(
    *,
    checkpoint: str | None = None,
    atlas_dir: str | Path | None = None,
    device: str = "cpu",
    seed: int = 41,
    state_dir: str | Path | None = None,
    teacher: ForecastTeacher | None = None,
) -> dict[str, object]:
    """Fit operator bases in r0, then cold-audit and release them in r1."""

    dataset = generate_synthetic_dataset(
        SyntheticConfig(
            seed=seed,
            n_series=1,
            total_steps=1536,
            train_fraction=0.125,
            history_length=64,
            horizon=16,
            train_stride=16,
            online_stride=16,
            missing_rate=0.0,
        )
    )
    r0 = _regime_examples(dataset, 0)
    r1 = _regime_examples(dataset, 1)
    if len(r0) < 12 or len(r1) < 12:
        raise RuntimeError("transfer world lacks complete regime windows")
    load_started = time.perf_counter()
    selected_teacher = teacher or _load_teacher(
        "chronos2", checkpoint=checkpoint, device=device
    )
    enforce_o1_transition = teacher is None
    r1_o1_generation = 1 if enforce_o1_transition else 0
    load_seconds = time.perf_counter() - load_started
    if not selected_teacher.capabilities.known_future_covariates:
        raise ValueError("transfer demo requires a known-future-capable teacher")
    model = getattr(getattr(selected_teacher, "pipeline", None), "model", None)
    model_identity = _module_sha256(model) or selected_teacher.identity  # type: ignore[attr-defined]

    temporary = tempfile.TemporaryDirectory() if state_dir is None else None
    context = temporary if temporary is not None else nullcontext()
    with context:
        root = (
            Path(temporary.name)
            if temporary is not None
            else Path(state_dir).expanduser().absolute()
        )  # type: ignore[arg-type]
        root.mkdir(parents=True, exist_ok=True)
        atlas = None if atlas_dir is None else ForecastAtlas.from_immer(atlas_dir)
        r0_ticket, _ = _authority(
            selected_teacher,
            r0[0],
            model_identity=model_identity,
            root=root,
            atlas=atlas,
            name="route-r0",
        )
        r1_ticket, _ = _authority(
            selected_teacher,
            r1[4],
            model_identity=model_identity,
            root=root,
            atlas=atlas,
            name="route-r1",
        )
        authority_citations = tuple(
            receipt
            for ticket in (r0_ticket, r1_ticket)
            for receipt in (
                ticket.atlas_forecast_receipt,
                ticket.atlas_transport_receipt,
            )
            if receipt is not None
        )
        atlas_evidence_sha256 = sha256_json(
            {
                "receipts": [
                    {
                        "kind": receipt.kind,
                        "revision_sha256": receipt.revision_sha256,
                        "segment_sha256": receipt.segment_sha256,
                    }
                    for receipt in authority_citations
                ],
                "schema": "o1fm.transfer-authority-citations/v1",
            }
        )
        projection_scope = sha256_json(
            {
                "horizon": 16,
                "known": 4,
                "schema": "o1fm.response-transfer-parent/v1",
                "targets": 3,
            }
        )
        fit_scope = sha256_json(
            {
                "model_identity": model_identity,
                "projection_scope": projection_scope,
                "schema": "o1fm.response-transfer-fit-scope/v1",
            }
        )
        schema0, route0 = _schema(r0[0].inputs, r0_ticket, projection_scope)
        schema1, route1 = _schema(r1[4].inputs, r1_ticket, projection_scope)
        if schema0.layout_identity_sha256 != schema1.layout_identity_sha256:
            raise RuntimeError("transfer schemas do not share one layout")
        if schema0.graph_topology_sha256 == schema1.graph_topology_sha256:
            raise RuntimeError("transfer world did not produce a topology shift")
        encoder0 = InterventionResponseFeatureEncoder(schema0)
        encoder1 = InterventionResponseFeatureEncoder(schema1)
        key0 = _response_key(
            schema0,
            model_identity=model_identity,
            regime=0,
            o1_generation=0,
            fit_scope=fit_scope,
        )
        key1 = _response_key(
            schema1,
            model_identity=model_identity,
            regime=1,
            o1_generation=r1_o1_generation,
            fit_scope=fit_scope,
        )
        if key0.crystal_key() != key1.crystal_key():
            raise RuntimeError("exact transfer domains do not share parent parameters")

        responses = InterventionResponseCrystal(
            root / "responses",
            ridge=0.03,
            min_support=8,
            min_sequential_support=3,
            max_mean_effect_error=0.35,
            min_direction_cosine=0.4,
            min_coverage=0.4,
            quantile_levels=selected_teacher.quantile_levels,
        )
        actions = ActionBank(
            root / "actions.json",
            min_support=3,
            quality_tolerance=0.03,
            coverage_tolerance=0.12,
            latency_cost_weight=0.0,
            forward_cost_weight=10.0,
        )
        conformal = O1ConformalField(
            root / "conformal",
            config=ConformalFieldConfig(target_coverage=0.8),
        )
        response_o1 = O1ErrorStream(
            root / "response-o1.json",
            stream_id="response-transfer",
            config=O1Config(quantiles=(0.1, 0.5, 0.9)),
        )
        authority_o1 = O1ErrorStream(
            root / "authority-regime-o1.json",
            stream_id="chronos-authority-transfer",
            config=O1Config(
                quantiles=(0.1, 0.5, 0.9),
                cusum_decay=0.9,
                cusum_drift=0.0,
                cusum_threshold=1.0,
            ),
        )

        factual_cache: dict[str, ForecastPrediction] = {
            _window_identity(r0[0].inputs): r0_ticket.prediction,
            _window_identity(r1[4].inputs): r1_ticket.prediction,
        }

        def factual(example: ForecastExample) -> ForecastPrediction:
            identity = _window_identity(example.inputs)
            if identity not in factual_cache:
                factual_cache[identity] = selected_teacher.predict(example.inputs)
            return factual_cache[identity]

        series_ticks = dataset.series[0].timestamps.astype(np.int64, copy=False)

        def observe_authority_o1(
            example: ForecastExample,
            *,
            observation_id: str,
        ) -> dict[str, object]:
            prediction = factual(example)
            target = dataset.counterfactual_target(example.inputs)
            assert prediction.quantiles is not None
            assert prediction.quantile_levels is not None
            quantile_map = {}
            for index, raw_level in enumerate(prediction.quantile_levels):
                level = float(raw_level)
                for configured in authority_o1.config.quantiles:
                    if math.isclose(level, configured, abs_tol=1e-7):
                        quantile_map[configured] = prediction.quantiles[..., index]
            update = authority_o1.observe(
                target.values,
                prediction.point,
                quantile_forecasts=quantile_map,
                mask=target.mask,
                observation_id=observation_id,
            )
            origin = int(
                np.searchsorted(
                    series_ticks,
                    example.inputs.future_timestamps[0].astype(np.int64),
                )
            )
            return {
                "cusum": update.cusum,
                "generation": update.regime_generation,
                "origin": origin,
                "regime_changed": update.regime_changed,
                "structural_regime": int(dataset.series[0].regime_ids[origin]),
                "surprise": update.surprise,
            }

        operators = _training_operators(16)
        candidates: list[_Probe] = []
        for ordinal, operator in enumerate(operators):
            example = r0[ordinal % 4]
            plan = operator.plan(example.inputs, name=f"r0-train:{ordinal}")
            feature = _features(encoder0, example.inputs, plan, route0, 0, 0)
            candidates.append(
                _Probe(
                    example=example,
                    operator=operator,
                    plan=plan,
                    features=feature,
                    target=_probe_target(
                        model_identity=model_identity,
                        key=key0,
                        example=example,
                        plan=plan,
                        ordinal=ordinal,
                    ),
                )
            )
        selector = ForecastCartographer(
            root / "cartographer-r0",
            budgets=CartographerBudgets(work=100.0, bytes=10_000_000),
        )
        selector.extend_targets([probe.target for probe in candidates])
        selected_targets = selector.d_optimal_batch(
            {probe.target: probe.features for probe in candidates},
            batch_size=48,
        )
        selected_ids = {target.sha256 for target in selected_targets}
        selected_probes = tuple(
            probe for probe in candidates if probe.target.sha256 in selected_ids
        )
        family_counts = {
            family: sum(probe.operator.family == family for probe in selected_probes)
            for family in {_family(0), _family(2), _family(3)}
        }
        if min(family_counts.values()) < 8:
            raise RuntimeError("D-optimal acquisition omitted an operator family")
        design_rank = int(
            np.linalg.matrix_rank(
                np.stack([probe.features for probe in selected_probes], axis=0)
            )
        )

        training_forwards = 0
        grouped: dict[str, list[_Probe]] = defaultdict(list)
        for probe in selected_probes:
            grouped[_window_identity(probe.example.inputs)].append(probe)
        for group in grouped.values():
            example = group[0].example
            factual_prediction = factual(example)
            factual_target = dataset.counterfactual_target(example.inputs)
            prepared = []
            for probe in group:
                edited = apply_intervention(example.inputs, probe.plan)
                raw_started = time.perf_counter()
                raw = selected_teacher.predict(edited)
                raw_latency = time.perf_counter() - raw_started
                training_forwards += 1
                compiled = responses.predict(key0, probe.features, factual_prediction)
                profile = _profile(key0, probe.operator.family)
                application = (
                    None
                    if compiled is None
                    else conformal.apply(
                        _conformal_key(profile, compiled.corrected), compiled.corrected
                    )
                )
                prepared.append(
                    (probe, edited, raw, raw_latency, compiled, profile, application)
                )
            for (
                probe,
                edited,
                raw,
                raw_latency,
                compiled,
                profile,
                application,
            ) in prepared:
                target = dataset.counterfactual_target(edited)
                raw_score = score_prediction(raw, target)
                actions.settle(
                    ActionReceipt(
                        profile,
                        "foundation",
                        raw_score.quality,
                        raw_score.interval_coverage or 0.0,
                        raw_latency,
                        1.0,
                    ),
                    settlement_id=f"r0:{probe.target.sha256}:foundation",
                )
                if compiled is not None:
                    assert application is not None
                    compiled_score = score_prediction(application.prediction, target)
                    actions.settle(
                        ActionReceipt(
                            profile,
                            "crystal",
                            compiled_score.quality,
                            compiled_score.interval_coverage or 0.0,
                            0.0,
                            0.0,
                        ),
                        settlement_id=f"r0:{probe.target.sha256}:crystal",
                    )
                    _settle_conformal(
                        conformal,
                        response_o1,
                        application.key,
                        application,
                        target,
                        observation_id=f"r0:{probe.target.sha256}:conformal",
                    )
                responses.observe(
                    key0,
                    probe.features,
                    factual_prediction,
                    factual_target,
                    target,
                    observation_id=f"r0:{probe.target.sha256}:response",
                )
                selector.settle(
                    ProbeEvidence(
                        target_sha256=probe.target.sha256,
                        coverage_gap=(raw_score.interval_coverage or 0.0) - 0.8,
                        pinball=raw_score.pinball or raw_score.mae,
                        mae=raw_score.mae,
                        joint_miss=0.0,
                        o1_surprise=0.0,
                        learning_progress=0.0,
                        route_w1=r0_ticket.route.transport_w1,
                        observed_values=raw_score.observed_values,
                        work_cost=1.0,
                        result_sha256=_array_sha256(raw.point),
                        atlas_sha256=atlas_evidence_sha256,
                        started_at=0.0,
                        completed_at=0.0,
                    )
                )

        def evaluate(
            examples: tuple[ForecastExample, ...],
            *,
            key: InterventionResponseKey,
            encoder: InterventionResponseFeatureEncoder,
            route: ResponseRouteContext,
            regime: int,
            o1_generation: int,
            stage: str,
            settle: bool,
        ) -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []
            for origin_ordinal, example in enumerate(examples):
                factual_prediction = factual(example)
                factual_target = dataset.counterfactual_target(example.inputs)
                prepared = []
                for operator in _evaluation_operators(
                    example.inputs.horizon, origin_ordinal
                ):
                    plan = operator.plan(
                        example.inputs,
                        name=f"{stage}:{origin_ordinal}:{operator.family}",
                    )
                    edited = apply_intervention(example.inputs, plan)
                    feature = _features(
                        encoder,
                        example.inputs,
                        plan,
                        route,
                        regime,
                        o1_generation,
                    )
                    compiled = responses.predict(key, feature, factual_prediction)
                    profile = _profile(key, operator.family)
                    application = (
                        None
                        if compiled is None
                        else conformal.apply(
                            _conformal_key(profile, compiled.corrected),
                            compiled.corrected,
                        )
                    )
                    recommendation = actions.recommend(profile)
                    action = (
                        "crystal"
                        if recommendation.action == "crystal"
                        and compiled is not None
                        and compiled.stats.ready
                        else "foundation"
                    )
                    raw_started = time.perf_counter()
                    raw = selected_teacher.predict(edited)
                    raw_latency = time.perf_counter() - raw_started
                    prepared.append(
                        (
                            operator,
                            plan,
                            edited,
                            feature,
                            compiled,
                            profile,
                            application,
                            action,
                            raw,
                            raw_latency,
                        )
                    )
                for (
                    operator,
                    _plan,
                    edited,
                    feature,
                    _compiled,
                    profile,
                    application,
                    action,
                    raw,
                    raw_latency,
                ) in prepared:
                    target = dataset.counterfactual_target(edited)
                    raw_score = score_prediction(raw, target)
                    compiled_prediction = (
                        None if application is None else application.prediction
                    )
                    compiled_score = (
                        None
                        if compiled_prediction is None
                        else score_prediction(compiled_prediction, target)
                    )
                    selected_prediction = (
                        compiled_prediction
                        if action == "crystal" and compiled_prediction is not None
                        else raw
                    )
                    selected_score = score_prediction(selected_prediction, target)
                    row = {
                        "action": action,
                        "family": operator.family,
                        "shape": operator.shape,
                        "raw_score": raw_score,
                        "compiled_score": compiled_score,
                        "selected_score": selected_score,
                        "raw_effect": _effect_metrics(
                            raw, factual_prediction, target, factual_target
                        ),
                        "compiled_effect": None
                        if compiled_prediction is None
                        else _effect_metrics(
                            compiled_prediction,
                            factual_prediction,
                            target,
                            factual_target,
                        ),
                    }
                    rows.append(row)
                    if settle:
                        actions.settle(
                            ActionReceipt(
                                profile,
                                "foundation",
                                raw_score.quality,
                                raw_score.interval_coverage or 0.0,
                                raw_latency,
                                1.0,
                            ),
                            settlement_id=f"{stage}:{origin_ordinal}:{operator.family}:foundation",
                        )
                        if compiled_score is not None and application is not None:
                            actions.settle(
                                ActionReceipt(
                                    profile,
                                    "crystal",
                                    compiled_score.quality,
                                    compiled_score.interval_coverage or 0.0,
                                    0.0,
                                    0.0,
                                ),
                                settlement_id=f"{stage}:{origin_ordinal}:{operator.family}:crystal",
                            )
                            _settle_conformal(
                                conformal,
                                response_o1,
                                application.key,
                                application,
                                target,
                                observation_id=f"{stage}:{origin_ordinal}:{operator.family}:conformal",
                            )
                        responses.observe(
                            key,
                            feature,
                            factual_prediction,
                            factual_target,
                            target,
                            observation_id=f"{stage}:{origin_ordinal}:{operator.family}:response",
                        )
            return rows

        r0_holdout = evaluate(
            r0[-4:],
            key=key0,
            encoder=encoder0,
            route=route0,
            regime=0,
            o1_generation=0,
            stage="r0-holdout",
            settle=True,
        )
        o1_transition_trace = [
            observe_authority_o1(
                example,
                observation_id=f"authority-transition:{phase}:{index}",
            )
            for phase, examples in (("r0", r0[-6:]), ("r1", r1[:1]))
            for index, example in enumerate(examples)
        ]
        if enforce_o1_transition and any(
            bool(row["regime_changed"])
            for row in o1_transition_trace
            if row["structural_regime"] == 0
        ):
            raise RuntimeError("authority O1 changed generation before the world shift")
        transition = o1_transition_trace[-1]
        if enforce_o1_transition and (
            transition["structural_regime"] != 1
            or transition["generation"] != 1
            or not transition["regime_changed"]
        ):
            raise RuntimeError(
                "authority O1 did not detect the structural regime shift"
            )
        r1_audits = evaluate(
            r1[5:8],
            key=key1,
            encoder=encoder1,
            route=route1,
            regime=1,
            o1_generation=r1_o1_generation,
            stage="r1-audit",
            settle=True,
        )
        r1_local = evaluate(
            r1[8:12],
            key=key1,
            encoder=encoder1,
            route=route1,
            regime=1,
            o1_generation=r1_o1_generation,
            stage="r1-local",
            settle=True,
        )
        r0_summary = _evaluation_summary(r0_holdout)
        audit_summary = _evaluation_summary(r1_audits)
        r1_summary = _evaluation_summary(r1_local)
        baseline_forwards = (
            (len(r0_holdout) + len(r1_audits) + len(r1_local)) + 4 + 3 + 4
        )
        selected_branch_forwards = sum(
            row["action"] == "foundation"
            for row in (*r0_holdout, *r1_audits, *r1_local)
        )
        origins = 4 + 3 + 4
        selected_forwards = origins + selected_branch_forwards

        def stage_economics(
            rows: list[dict[str, object]], stage_origins: int
        ) -> dict[str, float | int]:
            baseline = stage_origins * 4
            selected = stage_origins + sum(
                row["action"] == "foundation" for row in rows
            )
            return {
                "baseline_forwards": baseline,
                "selected_forwards": selected,
                "serving_forward_reduction": 1.0 - selected / baseline,
            }

        atlas_result = None
        if atlas is not None:
            remounted = ForecastAtlas.from_immer(atlas_dir)  # type: ignore[arg-type]
            atlas_result = {
                "authority_citations": len(authority_citations),
                "revision": list(atlas.store.revision()),
                "verified": atlas.verify(),
                "remount_verified": remounted.verify(),
            }
        return {
            "atlas": atlas_result,
            "cartographer": {
                "candidate_count": len(candidates),
                "design_rank": design_rank,
                "family_counts": family_counts,
                "metrics": selector.metrics(),
                "selected_count": len(selected_probes),
            },
            "control": {
                "authority_o1": authority_o1.metrics(),
                "authority_o1_transition_trace": o1_transition_trace,
                "conformal": conformal.metrics(),
                "response_o1": response_o1.metrics(),
            },
            "domains": {
                "r0_holdout": r0_summary,
                "r1_cold_audits": audit_summary,
                "r1_after_audits": r1_summary,
            },
            "economics": {
                "actual_measurement_branch_forwards": len(r0_holdout)
                + len(r1_audits)
                + len(r1_local),
                "baseline_serving_forwards": baseline_forwards,
                "selected_serving_forwards": selected_forwards,
                "serving_forward_reduction": 1.0
                - selected_forwards / baseline_forwards,
                "stages": {
                    "r0_holdout": stage_economics(r0_holdout, 4),
                    "r1_after_audits": stage_economics(r1_local, 4),
                    "r1_cold_audits": stage_economics(r1_audits, 3),
                },
                "training_branch_forwards": training_forwards,
            },
            "environment": {
                "device": device,
                "machine": platform.machine(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
            },
            "identities": {
                "fit_scope_sha256": fit_scope,
                "layout_sha256": schema0.layout_identity_sha256,
                "projection_scope_sha256": projection_scope,
                "r0_exact_key_sha256": key0.identity_sha256,
                "r0_topology_sha256": schema0.graph_topology_sha256,
                "r1_exact_key_sha256": key1.identity_sha256,
                "r1_o1_generation": r1_o1_generation,
                "r1_topology_sha256": schema1.graph_topology_sha256,
                "shared_inner_crystal": key0.crystal_key() == key1.crystal_key(),
            },
            "response": {
                "r0": _response_stats(responses, key0),
                "r1": _response_stats(responses, key1),
            },
            "schema": "o1fm.response-transfer-demo/v1",
            "teacher": {
                "checkpoint": getattr(selected_teacher, "checkpoint", "injected"),
                "identity_sha256": selected_teacher.identity,  # type: ignore[attr-defined]
                "load_seconds": load_seconds,
                "weights_sha256": model_identity,
            },
        }


__all__ = ["run_response_transfer_demo"]
