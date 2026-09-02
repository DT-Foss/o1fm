"""Sequential Chronos-plus-response-Crystal intervention compiler proof."""

from __future__ import annotations

import math
import platform
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .action_bank import ActionBank, ActionProfileKey, ActionReceipt
from .atlas import ForecastAtlas
from .branches import InterventionPlan, InterventionSpec, apply_intervention
from .cartography_demo import _array_sha256, _load_teacher, _module_sha256
from .causal_graph import CausalGraphConfig, OnlineCausalGraph
from .conformal import (
    ConformalApplication,
    ConformalFieldConfig,
    ConformalFieldKey,
    O1ConformalField,
)
from .contracts import ForecastInputs, ForecastPrediction, ForecastTarget
from .crystals import ForecastCrystalBank
from .identity import sha256_json
from .intervention_crystals import (
    InterventionResponseCrystal,
    InterventionResponseKey,
    InterventionResponsePrediction,
)
from .metrics import ForecastScore, score_prediction
from .o1 import O1Config, O1ErrorStream
from .runtime import O1FMRuntime
from .synthetic import SyntheticConfig, SyntheticDataset, generate_synthetic_dataset


@dataclass(frozen=True, slots=True)
class _Family:
    name: str
    covariate: int
    operation: str

    @property
    def identity_sha256(self) -> str:
        return sha256_json(
            {
                "covariate": self.covariate,
                "name": self.name,
                "operation": self.operation,
                "schema": "o1fm.intervention-family/v1",
            }
        )

    def plan(self, inputs: ForecastInputs, value: float) -> InterventionPlan:
        values = np.full(
            (inputs.horizon, 1),
            value,
            dtype=np.float32,
        )
        return InterventionPlan(
            self.name,
            (
                InterventionSpec(
                    name=f"{self.name}-cells",
                    operation=self.operation,
                    timestamps=inputs.future_timestamps,
                    covariate_indices=(self.covariate,),
                    values=values,
                    lower_bounds=(-2.0,),
                    upper_bounds=(2.0,),
                ),
            ),
            assumptions=("Paired structural replay preserves exogenous noise",),
        )


@dataclass(frozen=True, slots=True)
class _PreparedBranch:
    family: _Family
    target: ForecastTarget
    features: np.ndarray
    profile: ActionProfileKey
    raw: ForecastPrediction
    raw_latency: float
    compiled: InterventionResponsePrediction | None
    application: ConformalApplication | None
    action: str | None = None


_FAMILIES = (
    _Family("seasonal-driver-shift", 0, "shift"),
    _Family("scheduled-intervention-level", 2, "set"),
    _Family("unused-trend-placebo", 3, "shift"),
)

_TRAIN_VALUES = {
    "seasonal-driver-shift": (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0),
    "scheduled-intervention-level": (0.0, 1.0, 0.25, 0.75, 0.5, 0.125, 0.875, 0.375),
    "unused-trend-placebo": (-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8),
}

_EVALUATION_VALUES = {
    "seasonal-driver-shift": (-0.6, -0.1, 0.35, 0.9),
    "scheduled-intervention-level": (0.2, 0.4, 0.6, 0.8),
    "unused-trend-placebo": (-0.7, -0.1, 0.3, 0.7),
}


def _feature_schema(inputs: ForecastInputs) -> str:
    return sha256_json(
        {
            "abi": "o1fm.intervention-delta-schedule/v1",
            "horizon": inputs.horizon,
            "known_covariates": inputs.known_covariate_dim,
            "route_w1": True,
            "target_count": inputs.target_dim,
        }
    )


def _response_features(
    inputs: ForecastInputs,
    edited: ForecastInputs,
    route_w1: float,
) -> np.ndarray:
    history = inputs.history_length
    delta = np.asarray(
        edited.known_covariates[history:], dtype=np.float64
    ) - np.asarray(inputs.known_covariates[history:], dtype=np.float64)
    mask = inputs.known_covariate_mask[history:] & edited.known_covariate_mask[history:]
    delta[~mask] = 0.0
    result = np.concatenate(
        (
            np.asarray((1.0, route_w1), dtype=np.float64),
            delta.reshape(-1),
        )
    )
    result.setflags(write=False)
    return result


def _profile(
    key: InterventionResponseKey,
    family: _Family,
) -> ActionProfileKey:
    return ActionProfileKey(
        model_identity=key.model_identity,
        feature_schema=sha256_json(
            {
                "family_sha256": family.identity_sha256,
                "response_key_sha256": key.identity_sha256,
                "schema": "o1fm.intervention-action-profile/v1",
            }
        ),
        graph_revision=key.graph_revision,
        regime=key.regime,
        horizon=key.horizon,
        target_count=key.target_count,
    )


def _conformal_key(
    profile: ActionProfileKey,
    prediction: ForecastPrediction,
) -> ConformalFieldKey:
    assert prediction.quantile_levels is not None
    levels = tuple(float(value) for value in prediction.quantile_levels)
    return ConformalFieldKey(
        model_identity=profile.model_identity,
        feature_schema=profile.feature_schema,
        graph_revision=profile.graph_revision,
        regime=profile.regime,
        action="crystal",
        horizon=profile.horizon,
        target_count=profile.target_count,
        lower_quantile=levels[0],
        upper_quantile=levels[-1],
    )


def _settle_conformal(
    field: O1ConformalField,
    o1: O1ErrorStream,
    key: ConformalFieldKey,
    application: ConformalApplication,
    target: ForecastTarget,
    *,
    observation_id: str,
) -> None:
    prediction = application.prediction
    assert prediction.quantiles is not None
    assert prediction.quantile_levels is not None
    quantile_map = {}
    for index, raw_level in enumerate(prediction.quantile_levels):
        level = float(raw_level)
        for configured in o1.config.quantiles:
            if math.isclose(level, configured, abs_tol=1e-7):
                quantile_map[configured] = prediction.quantiles[..., index]
    update = o1.observe(
        target.values,
        prediction.point,
        quantile_forecasts=quantile_map,
        mask=target.mask,
        observation_id=observation_id,
    )
    field.observe(
        key,
        application,
        target,
        surprise=update.surprise,
        cusum=update.cusum,
        observation_id=observation_id,
    )


def _score_record(score: ForecastScore) -> dict[str, float | int | None]:
    return {
        "coverage": score.interval_coverage,
        "mae": score.mae,
        "observed_values": score.observed_values,
        "pinball": score.pinball,
        "width": score.interval_width,
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _effect_metrics(
    prediction: ForecastPrediction,
    factual_prediction: ForecastPrediction,
    intervention_target: ForecastTarget,
    factual_target: ForecastTarget,
) -> dict[str, float]:
    mask = intervention_target.mask & factual_target.mask
    predicted = np.asarray(prediction.point) - np.asarray(factual_prediction.point)
    actual = np.asarray(intervention_target.values) - np.asarray(factual_target.values)
    predicted_values = np.asarray(predicted[mask], dtype=np.float64)
    actual_values = np.asarray(actual[mask], dtype=np.float64)
    predicted_magnitude = float(np.mean(np.abs(predicted_values)))
    actual_magnitude = float(np.mean(np.abs(actual_values)))
    denominator = float(
        np.linalg.norm(predicted_values) * np.linalg.norm(actual_values)
    )
    cosine = (
        1.0
        if denominator == 0.0
        and not np.any(predicted_values)
        and not np.any(actual_values)
        else 0.0
        if denominator == 0.0
        else float(predicted_values @ actual_values) / denominator
    )
    return {
        "direction_cosine": cosine,
        "effect_mae": float(np.mean(np.abs(predicted_values - actual_values))),
        "magnitude": predicted_magnitude,
        "recovery": (
            1.0
            if actual_magnitude == 0.0 and predicted_magnitude == 0.0
            else 0.0
            if actual_magnitude == 0.0
            else predicted_magnitude / actual_magnitude
        ),
        "true_magnitude": actual_magnitude,
    }


def _regime_zero_examples(dataset: SyntheticDataset) -> tuple[object, ...]:
    series = dataset.series[0]
    ticks = series.timestamps.astype(np.int64, copy=False)
    selected = []
    for example in dataset.online:
        if example.series_id != series.series_id:
            continue
        origin = int(
            np.searchsorted(
                ticks,
                example.inputs.future_timestamps[0].astype(np.int64),
            )
        )
        if bool(
            (series.regime_ids[origin : origin + example.inputs.horizon] == 0).all()
        ):
            selected.append(example)
    return tuple(selected)


def run_intervention_compiler_demo(
    *,
    checkpoint: str | None = None,
    atlas_dir: str | Path,
    device: str = "cpu",
    seed: int = 31,
    state_dir: str | Path | None = None,
    training_windows: int = 8,
    evaluation_windows: int = 4,
) -> dict[str, object]:
    """Compile early paired effects and release later branches locally."""

    if training_windows < 4 or evaluation_windows < 1:
        raise ValueError(
            "compiler demo requires at least four train and one eval window"
        )
    if training_windows > 8 or evaluation_windows > 4:
        raise ValueError(
            "compiler demo supports at most eight train and four eval windows"
        )
    config = SyntheticConfig(
        seed=seed,
        n_series=1,
        total_steps=1152,
        train_fraction=0.15,
        history_length=64,
        horizon=16,
        train_stride=16,
        online_stride=16,
        missing_rate=0.0,
    )
    dataset = generate_synthetic_dataset(config)
    available = _regime_zero_examples(dataset)
    required = training_windows + evaluation_windows
    if len(available) < required:
        raise RuntimeError("synthetic world has too few same-regime online windows")
    train = available[:training_windows]
    evaluation = available[training_windows:required]

    load_started = time.perf_counter()
    teacher = _load_teacher("chronos2", checkpoint=checkpoint, device=device)
    load_seconds = time.perf_counter() - load_started
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
        first = train[0]
        atlas = ForecastAtlas.from_immer(atlas_dir)
        authority = O1FMRuntime(
            teacher,
            model_identity=weights_sha256,
            o1=O1ErrorStream(
                root / "authority" / "o1.json",
                stream_id=first.series_id,
                config=O1Config(quantiles=(0.1, 0.5, 0.9)),
            ),
            graph=OnlineCausalGraph(
                root / "authority" / "graph",
                graph_id=first.series_id,
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
            ),
            actions=ActionBank(root / "authority" / "actions.json"),
            atlas=atlas,
            route_top_k=32,
            enable_seasonal=False,
        )
        first_ticket = authority.forecast(first.inputs)
        first_target = dataset.counterfactual_target(first.inputs)
        first_settlement = authority.settle(first_ticket, first_target)
        route = first_ticket.route
        response_key = InterventionResponseKey(
            model_identity=weights_sha256,
            feature_schema=sha256_json(
                {
                    "abi": "o1fm.response-bias-route-w1/v1",
                    "route_topology": route.signature_sha256,
                }
            ),
            intervention_schema=_feature_schema(first.inputs),
            target_schema=sha256_json(
                {
                    "dtype": "float32",
                    "schema": "o1fm.synthetic-target-order/v1",
                    "targets": [
                        f"target:{index}" for index in range(first.inputs.target_dim)
                    ],
                }
            ),
            graph_revision=route.signature_sha256,
            regime="synthetic-regime:0",
            horizon=first.inputs.horizon,
            target_count=first.inputs.target_dim,
        )
        responses = InterventionResponseCrystal(
            root / "responses",
            ridge=0.01,
            min_support=4,
            min_sequential_support=3,
            max_mean_effect_error=0.25,
            min_direction_cosine=0.5,
            min_coverage=0.5,
            quantile_levels=teacher.quantile_levels,
        )
        branch_actions = ActionBank(
            root / "branch-actions.json",
            min_support=3,
            quality_tolerance=0.02,
            coverage_tolerance=0.10,
            latency_cost_weight=0.0,
            forward_cost_weight=10.0,
        )
        response_field = O1ConformalField(
            root / "response-conformal",
            config=ConformalFieldConfig(target_coverage=0.8),
        )
        response_o1 = O1ErrorStream(
            root / "response-o1.json",
            stream_id=f"{first.series_id}:intervention-response",
            config=O1Config(quantiles=(0.1, 0.5, 0.9)),
        )

        training_forwards = 1
        training_crystal_receipts = 0
        training_started = time.perf_counter()
        for window_index, example in enumerate(train):
            factual_target = dataset.counterfactual_target(example.inputs)
            factual_prediction = (
                first_ticket.prediction
                if window_index == 0
                else teacher.predict(example.inputs)
            )
            if window_index:
                training_forwards += 1
            prepared: list[_PreparedBranch] = []
            for family in _FAMILIES:
                plan = family.plan(
                    example.inputs,
                    _TRAIN_VALUES[family.name][window_index],
                )
                edited = apply_intervention(example.inputs, plan)
                intervention_target = dataset.counterfactual_target(edited)
                features = _response_features(
                    example.inputs, edited, route.transport_w1
                )
                profile = _profile(response_key, family)
                raw_started = time.perf_counter()
                raw = teacher.predict(edited)
                raw_latency = time.perf_counter() - raw_started
                training_forwards += 1
                compiled = responses.predict(response_key, features, factual_prediction)
                application = (
                    None
                    if compiled is None
                    else response_field.apply(
                        _conformal_key(profile, compiled.corrected),
                        compiled.corrected,
                    )
                )
                prepared.append(
                    _PreparedBranch(
                        family=family,
                        target=intervention_target,
                        features=features,
                        profile=profile,
                        raw=raw,
                        raw_latency=raw_latency,
                        compiled=compiled,
                        application=application,
                    )
                )

            # All simultaneous branches above see the same pre-settlement
            # compiler, O1, conformal, and ActionBank state.
            for branch in prepared:
                family = branch.family
                intervention_target = branch.target
                profile = branch.profile
                raw = branch.raw
                raw_latency = branch.raw_latency
                raw_score = score_prediction(raw, intervention_target)
                branch_actions.settle(
                    ActionReceipt(
                        profile=profile,
                        action="foundation",
                        quality=raw_score.quality,
                        coverage=raw_score.interval_coverage or 0.0,
                        latency_seconds=raw_latency,
                        forward_cost=1.0,
                    ),
                    settlement_id=f"train:{window_index}:{family.name}:foundation",
                )
                if branch.compiled is not None:
                    assert branch.application is not None
                    conformal_key = branch.application.key
                    compiled_score = score_prediction(
                        branch.application.prediction,
                        intervention_target,
                    )
                    branch_actions.settle(
                        ActionReceipt(
                            profile=profile,
                            action="crystal",
                            quality=compiled_score.quality,
                            coverage=compiled_score.interval_coverage or 0.0,
                            latency_seconds=0.0,
                            forward_cost=0.0,
                        ),
                        settlement_id=f"train:{window_index}:{family.name}:crystal",
                    )
                    training_crystal_receipts += 1
                    _settle_conformal(
                        response_field,
                        response_o1,
                        conformal_key,
                        branch.application,
                        intervention_target,
                        observation_id=(f"response:train:{window_index}:{family.name}"),
                    )
                responses.observe(
                    response_key,
                    branch.features,
                    factual_prediction,
                    factual_target,
                    intervention_target,
                    observation_id=f"train:{window_index}:{family.name}",
                )
        training_seconds = time.perf_counter() - training_started

        family_rows: dict[str, dict[str, object]] = {}
        raw_scores: list[ForecastScore] = []
        compiled_scores: list[ForecastScore] = []
        raw_effects: list[dict[str, float]] = []
        compiled_effects: list[dict[str, float]] = []
        action_trace: list[str] = []
        selected_branch_forwards = 0
        benchmark_branch_forwards = 0
        evaluation_started = time.perf_counter()
        for window_index, example in enumerate(evaluation):
            factual_prediction = teacher.predict(example.inputs)
            factual_target = dataset.counterfactual_target(example.inputs)
            prepared: list[_PreparedBranch] = []
            for family in _FAMILIES:
                plan = family.plan(
                    example.inputs,
                    _EVALUATION_VALUES[family.name][window_index],
                )
                edited = apply_intervention(example.inputs, plan)
                intervention_target = dataset.counterfactual_target(edited)
                features = _response_features(
                    example.inputs, edited, route.transport_w1
                )
                profile = _profile(response_key, family)
                compiled = responses.predict(response_key, features, factual_prediction)
                application = (
                    None
                    if compiled is None
                    else response_field.apply(
                        _conformal_key(profile, compiled.corrected),
                        compiled.corrected,
                    )
                )
                recommendation = branch_actions.recommend(profile)
                action = (
                    "crystal"
                    if recommendation.action == "crystal"
                    and compiled is not None
                    and compiled.stats.ready
                    else "foundation"
                )
                action_trace.append(action)
                raw_started = time.perf_counter()
                raw = teacher.predict(edited)
                raw_latency = time.perf_counter() - raw_started
                benchmark_branch_forwards += 1
                prepared.append(
                    _PreparedBranch(
                        family=family,
                        target=intervention_target,
                        features=features,
                        profile=profile,
                        raw=raw,
                        raw_latency=raw_latency,
                        compiled=compiled,
                        application=application,
                        action=action,
                    )
                )

            # Predict and decide the whole simultaneous branch family before
            # any realized alternative world is allowed to update state.
            for branch in prepared:
                family = branch.family
                intervention_target = branch.target
                profile = branch.profile
                raw = branch.raw
                raw_latency = branch.raw_latency
                compiled = branch.compiled
                action = branch.action
                assert action is not None
                raw_score = score_prediction(raw, intervention_target)
                raw_scores.append(raw_score)
                raw_effect = _effect_metrics(
                    raw,
                    factual_prediction,
                    intervention_target,
                    factual_target,
                )
                raw_effects.append(raw_effect)
                if compiled is None:
                    selected = raw
                    selected_branch_forwards += 1
                else:
                    assert branch.application is not None
                    compiled_prediction = branch.application.prediction
                    compiled_score = score_prediction(
                        compiled_prediction,
                        intervention_target,
                    )
                    compiled_scores.append(compiled_score)
                    compiled_effect = _effect_metrics(
                        compiled_prediction,
                        factual_prediction,
                        intervention_target,
                        factual_target,
                    )
                    compiled_effects.append(compiled_effect)
                    selected = compiled_prediction if action == "crystal" else raw
                    selected_branch_forwards += int(action == "foundation")
                selected_score = score_prediction(selected, intervention_target)
                branch_actions.settle(
                    ActionReceipt(
                        profile=profile,
                        action="foundation",
                        quality=raw_score.quality,
                        coverage=raw_score.interval_coverage or 0.0,
                        latency_seconds=raw_latency,
                        forward_cost=1.0,
                    ),
                    settlement_id=f"eval:{window_index}:{family.name}:foundation",
                )
                if compiled is not None:
                    assert branch.application is not None
                    branch_actions.settle(
                        ActionReceipt(
                            profile=profile,
                            action="crystal",
                            quality=compiled_score.quality,
                            coverage=compiled_score.interval_coverage or 0.0,
                            latency_seconds=0.0,
                            forward_cost=0.0,
                        ),
                        settlement_id=f"eval:{window_index}:{family.name}:crystal",
                    )
                    _settle_conformal(
                        response_field,
                        response_o1,
                        branch.application.key,
                        branch.application,
                        intervention_target,
                        observation_id=f"response:eval:{window_index}:{family.name}",
                    )
                responses.observe(
                    response_key,
                    branch.features,
                    factual_prediction,
                    factual_target,
                    intervention_target,
                    observation_id=f"eval:{window_index}:{family.name}",
                )
                row = family_rows.setdefault(
                    family.name,
                    {
                        "actions": [],
                        "compiled_coverage": [],
                        "compiled_mae": [],
                        "compiled_pinball": [],
                        "compiled_recovery": [],
                        "compiled_direction": [],
                        "raw_mae": [],
                        "raw_pinball": [],
                        "raw_recovery": [],
                        "raw_direction": [],
                        "raw_coverage": [],
                        "selected_mae": [],
                    },
                )
                row["actions"].append(action)  # type: ignore[union-attr]
                row["raw_mae"].append(raw_score.mae)  # type: ignore[union-attr]
                row["raw_coverage"].append(  # type: ignore[union-attr]
                    raw_score.interval_coverage or 0.0
                )
                row["raw_pinball"].append(raw_score.pinball or raw_score.mae)  # type: ignore[union-attr]
                row["raw_recovery"].append(raw_effect["recovery"])  # type: ignore[union-attr]
                row["raw_direction"].append(raw_effect["direction_cosine"])  # type: ignore[union-attr]
                row["selected_mae"].append(selected_score.mae)  # type: ignore[union-attr]
                if compiled is not None:
                    row["compiled_coverage"].append(  # type: ignore[union-attr]
                        compiled_score.interval_coverage or 0.0
                    )
                    row["compiled_mae"].append(compiled_score.mae)  # type: ignore[union-attr]
                    row["compiled_pinball"].append(  # type: ignore[union-attr]
                        compiled_score.pinball or compiled_score.mae
                    )
                    row["compiled_recovery"].append(  # type: ignore[union-attr]
                        compiled_effect["recovery"]
                    )
                    row["compiled_direction"].append(  # type: ignore[union-attr]
                        compiled_effect["direction_cosine"]
                    )
        evaluation_seconds = time.perf_counter() - evaluation_started

        summarized_families = {
            name: {
                "action_counts": {
                    action: row["actions"].count(action)  # type: ignore[union-attr]
                    for action in ("foundation", "crystal")
                },
                "compiled_coverage": _mean(row["compiled_coverage"]),  # type: ignore[arg-type]
                "compiled_direction_cosine": _mean(row["compiled_direction"]),  # type: ignore[arg-type]
                "compiled_mae": _mean(row["compiled_mae"]),  # type: ignore[arg-type]
                "compiled_pinball": _mean(row["compiled_pinball"]),  # type: ignore[arg-type]
                "compiled_recovery": _mean(row["compiled_recovery"]),  # type: ignore[arg-type]
                "raw_direction_cosine": _mean(row["raw_direction"]),  # type: ignore[arg-type]
                "raw_coverage": _mean(row["raw_coverage"]),  # type: ignore[arg-type]
                "raw_mae": _mean(row["raw_mae"]),  # type: ignore[arg-type]
                "raw_pinball": _mean(row["raw_pinball"]),  # type: ignore[arg-type]
                "raw_recovery": _mean(row["raw_recovery"]),  # type: ignore[arg-type]
                "selected_mae": _mean(row["selected_mae"]),  # type: ignore[arg-type]
            }
            for name, row in sorted(family_rows.items())
        }
        response_stats = responses.describe(response_key)
        baseline_forwards = evaluation_windows * (1 + len(_FAMILIES))
        selected_forwards = evaluation_windows + selected_branch_forwards
        remounted = ForecastAtlas.from_immer(atlas_dir)
        return {
            "atlas": {
                "remount_verified": remounted.verify(),
                "revision": list(atlas.store.revision()),
                "verified": atlas.verify(),
            },
            "compiler": {
                "action_trace": action_trace,
                "key_sha256": response_key.identity_sha256,
                "stats": {
                    "coverage": response_stats.empirical_coverage,
                    "direction_cosine": response_stats.effect_direction_cosine,
                    "feature_dim": response_stats.feature_dim,
                    "mean_effect_error": response_stats.mean_effect_error,
                    "observed_values": response_stats.observed_values,
                    "ready": response_stats.ready,
                    "sequential_support": response_stats.sequential_support,
                    "support": response_stats.support,
                },
                "training_crystal_receipts": training_crystal_receipts,
            },
            "conformal": {
                "field": response_field.metrics(),
                "o1": response_o1.metrics(),
            },
            "data": {
                "evaluation_windows": evaluation_windows,
                "families": [family.name for family in _FAMILIES],
                "evaluation_values": {
                    key: list(value) for key, value in _EVALUATION_VALUES.items()
                },
                "horizon": first.inputs.horizon,
                "target_count": first.inputs.target_dim,
                "training_windows": training_windows,
                "training_values": {
                    key: list(value) for key, value in _TRAIN_VALUES.items()
                },
                "world_target_sha256": _array_sha256(
                    dataset.series[0].structural_state.latent_target
                ),
            },
            "economics": {
                "actual_measured_eval_forwards": evaluation_windows
                + benchmark_branch_forwards,
                "baseline_eval_forwards": baseline_forwards,
                "benchmark_only_branch_forwards": benchmark_branch_forwards,
                "selected_eval_forwards": selected_forwards,
                "serving_policy_forward_reduction": 1.0
                - selected_forwards / baseline_forwards,
                "training_forwards": training_forwards,
            },
            "environment": {
                "device": device,
                "machine": platform.machine(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
            },
            "evaluation": {
                "compiled_mean_coverage": _mean(
                    [score.interval_coverage or 0.0 for score in compiled_scores]
                ),
                "compiled_mean_mae": _mean([score.mae for score in compiled_scores]),
                "compiled_mean_pinball": _mean(
                    [score.pinball or score.mae for score in compiled_scores]
                ),
                "compiled_mean_recovery": _mean(
                    [row["recovery"] for row in compiled_effects]
                ),
                "compiled_mean_direction_cosine": _mean(
                    [row["direction_cosine"] for row in compiled_effects]
                ),
                "families": summarized_families,
                "raw_mean_mae": _mean([score.mae for score in raw_scores]),
                "raw_mean_pinball": _mean(
                    [score.pinball or score.mae for score in raw_scores]
                ),
                "raw_mean_recovery": _mean([row["recovery"] for row in raw_effects]),
                "raw_mean_direction_cosine": _mean(
                    [row["direction_cosine"] for row in raw_effects]
                ),
                "raw_mean_coverage": _mean(
                    [score.interval_coverage or 0.0 for score in raw_scores]
                ),
            },
            "schema": "o1fm.intervention-compiler-demo/v1",
            "teacher": {
                "capabilities": teacher.capabilities.to_record(),
                "checkpoint": teacher.checkpoint,
                "identity_sha256": teacher.identity,
                "load_seconds": load_seconds,
                "parameter_count": parameter_count,
                "weights_sha256": weights_sha256,
            },
            "timings": {
                "evaluation_seconds": evaluation_seconds,
                "training_seconds": training_seconds,
            },
            "authority_fragment_sha256": first_settlement.fragment_sha256,
        }


__all__ = ["run_intervention_compiler_demo"]
