"""Forecast-native O1 runtime joining model, graph, Crystals and economics."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .action_bank import ActionBank, ActionProfileKey, ActionReceipt
from .baselines import SeasonalNaiveForecaster
from .causal_graph import CausalEdge, OnlineCausalGraph
from .conformal import (
    ConformalApplication,
    ConformalFieldKey,
    ConformalUpdate,
    O1ConformalField,
)
from .contracts import ForecastInputs, ForecastPrediction, ForecastTarget
from .crystals import ForecastCrystalBank, ForecastCrystalKey, ForecastCrystalPrediction
from .features import extract_features, feature_schema
from .identity import sha256_json
from .metrics import ForecastScore, score_prediction, verify_prediction
from .model import O1FMForecaster
from .o1 import O1ErrorStream, O1Update
from .teachers import ForecastTeacher, NativeO1FMTeacher


@dataclass(frozen=True, slots=True)
class RouteSelection:
    past_indices: tuple[int, ...]
    known_indices: tuple[int, ...]
    signature_sha256: str
    edges: tuple[CausalEdge, ...]


@dataclass(slots=True)
class _RouteState:
    active: RouteSelection
    pending: RouteSelection | None = None
    pending_support: int = 0


@dataclass(frozen=True, slots=True)
class ForecastTicket:
    ticket_id: str
    inputs: ForecastInputs
    prediction: ForecastPrediction
    base_prediction: ForecastPrediction
    calibration: ConformalApplication | None
    action: str
    profile: ActionProfileKey
    crystal_key: ForecastCrystalKey
    features: np.ndarray
    latency_seconds: float
    foundation_forwards: int
    route: RouteSelection
    regime: str
    candidates: Mapping[str, tuple[ForecastPrediction, float]]


@dataclass(frozen=True, slots=True)
class ForecastSettlement:
    ticket_id: str
    action: str
    score: ForecastScore
    base_score: ForecastScore
    o1_update: O1Update
    calibration_update: ConformalUpdate | None
    joint_covered: bool | None
    crystal_ready: bool
    foundation_forwards: int
    saved_foundation_forwards: int


def _prediction_from_crystal(
    inputs: ForecastInputs,
    prediction: ForecastCrystalPrediction,
) -> ForecastPrediction:
    levels = tuple(sorted(prediction.quantiles))
    quantiles = np.stack([prediction.quantiles[level] for level in levels], axis=2)
    return ForecastPrediction(
        series_id=inputs.series_id,
        timestamps=inputs.future_timestamps,
        point=prediction.point,
        quantiles=quantiles,
        quantile_levels=levels,
    )


class O1FMRuntime:
    """Use expensive forecasting for novelty and settle cheaper local actions."""

    def __init__(
        self,
        model: O1FMForecaster | ForecastTeacher,
        *,
        model_identity: str,
        o1: O1ErrorStream,
        graph: OnlineCausalGraph,
        crystals: ForecastCrystalBank,
        actions: ActionBank,
        conformal: O1ConformalField | None = None,
        compile_teacher_crystals: bool | None = None,
        seasonal_period: int = 24,
        route_top_k: int = 8,
        route_stability: int = 3,
        enable_seasonal: bool = True,
        device: str = "cpu",
    ) -> None:
        teacher: ForecastTeacher
        if isinstance(model, O1FMForecaster):
            teacher = NativeO1FMTeacher(model, device=device)
        elif isinstance(model, ForecastTeacher):
            teacher = model
        else:
            raise TypeError("model must be O1FMForecaster or ForecastTeacher")
        if not isinstance(model_identity, str) or len(model_identity) != 64:
            raise ValueError("model_identity must be a SHA-256 digest")
        if (
            seasonal_period <= 0
            or route_top_k <= 0
            or isinstance(route_stability, bool)
            or not isinstance(route_stability, int)
            or route_stability <= 0
        ):
            raise ValueError("runtime routing controls must be positive")
        if not isinstance(enable_seasonal, bool):
            raise TypeError("enable_seasonal must be boolean")
        if tuple(crystals.quantile_levels) != tuple(teacher.quantile_levels):
            raise ValueError("Crystal and foundation quantile inventories must match")
        if conformal is not None and not isinstance(conformal, O1ConformalField):
            raise TypeError("conformal must be O1ConformalField or None")
        if compile_teacher_crystals is not None and not isinstance(
            compile_teacher_crystals, bool
        ):
            raise TypeError("compile_teacher_crystals must be boolean or None")
        self.model = model
        self.teacher = teacher
        self.model_identity = model_identity
        self.o1 = o1
        self.graph = graph
        self.crystals = crystals
        self.actions = actions
        self.conformal = conformal
        self.compile_teacher_crystals = (
            not isinstance(teacher, NativeO1FMTeacher)
            if compile_teacher_crystals is None
            else compile_teacher_crystals
        )
        self.seasonal = SeasonalNaiveForecaster(
            period=seasonal_period,
            quantile_levels=crystals.quantile_levels,
        )
        self.seasonal_period = seasonal_period
        self.route_top_k = route_top_k
        self.route_stability = route_stability
        self.enable_seasonal = enable_seasonal
        self.device = device
        self._sequence = 0
        self._routes_by_regime: dict[str, _RouteState] = {}

    @staticmethod
    def _target_name(index: int) -> str:
        return f"target:{index}"

    @staticmethod
    def _past_name(index: int) -> str:
        return f"past:{index}"

    @staticmethod
    def _known_name(index: int) -> str:
        return f"known:{index}"

    def _observe_graph_history(self, inputs: ForecastInputs) -> None:
        observations: list[dict[str, object]] = []
        for row, timestamp in enumerate(inputs.past_timestamps):
            targets = {
                self._target_name(index): float(inputs.past_target[row, index])
                for index in range(inputs.target_dim)
                if inputs.past_target_mask[row, index]
            }
            if not targets:
                continue
            covariates = {
                self._past_name(index): float(inputs.past_covariates[row, index])
                for index in range(inputs.past_covariate_dim)
                if inputs.past_covariate_mask[row, index]
            }
            known = {
                self._known_name(index): float(inputs.known_covariates[row, index])
                for index in range(inputs.known_covariate_dim)
                if inputs.known_covariate_mask[row, index]
            }
            observations.append(
                {
                    "timestamp_ns": int(
                        timestamp.astype("datetime64[ns]").astype(np.int64)
                    ),
                    "targets": targets,
                    "covariates": covariates,
                    "known_future": known,
                    "observation_id": (
                        f"obs:{inputs.series_id}:{int(timestamp.astype(np.int64))}"
                    ),
                }
            )
        self.graph.observe_many(observations)

    def _route_candidate(self, inputs: ForecastInputs) -> RouteSelection:
        edges: list[CausalEdge] = []
        for target in range(inputs.target_dim):
            edges.extend(
                self.graph.route(self._target_name(target), top_k=self.route_top_k)
            )
        unique = {
            (edge.source, edge.target, edge.lag, edge.kind): edge for edge in edges
        }
        ordered = tuple(unique[key] for key in sorted(unique))
        past = tuple(
            sorted(
                {
                    int(edge.source.split(":", 1)[1])
                    for edge in ordered
                    if edge.source.startswith("past:")
                }
            )
        )
        known = tuple(
            sorted(
                {
                    int(edge.source.split(":", 1)[1])
                    for edge in ordered
                    if edge.source.startswith("known:")
                }
            )
        )
        if not past:
            past = tuple(range(inputs.past_covariate_dim))
        if not known:
            known = tuple(range(inputs.known_covariate_dim))
        signature = sha256_json(
            {
                "edges": [
                    {
                        "kind": edge.kind,
                        "lag": edge.lag,
                        "source": edge.source,
                        "target": edge.target,
                    }
                    for edge in ordered
                ],
                "known": list(known),
                "past": list(past),
                "schema": "o1fm.routed-feature-topology/v1",
            }
        )
        return RouteSelection(past, known, signature, ordered)

    def _route(self, inputs: ForecastInputs, *, regime: str) -> RouteSelection:
        self._observe_graph_history(inputs)
        candidate = self._route_candidate(inputs)
        state = self._routes_by_regime.get(regime)
        if state is None:
            self._routes_by_regime[regime] = _RouteState(active=candidate)
            return candidate

        if candidate.signature_sha256 == state.active.signature_sha256:
            state.active = candidate
            state.pending = None
            state.pending_support = 0
            return candidate

        if (
            state.pending is None
            or state.pending.signature_sha256 != candidate.signature_sha256
        ):
            state.pending = candidate
            state.pending_support = 1
        else:
            state.pending = candidate
            state.pending_support += 1

        if state.pending_support >= self.route_stability:
            state.active = candidate
            state.pending = None
            state.pending_support = 0
        return state.active

    def _calibrate(
        self,
        profile: ActionProfileKey,
        action: str,
        prediction: ForecastPrediction,
    ) -> ConformalApplication | None:
        if self.conformal is None:
            return None
        if prediction.quantile_levels is None:
            raise ValueError("conformal runtime requires probabilistic predictions")
        levels = tuple(prediction.quantile_levels)
        key = ConformalFieldKey(
            model_identity=profile.model_identity,
            feature_schema=profile.feature_schema,
            graph_revision=profile.graph_revision,
            regime=profile.regime,
            action=action,
            horizon=profile.horizon,
            target_count=profile.target_count,
            lower_quantile=levels[0],
            upper_quantile=levels[-1],
        )
        return self.conformal.apply(key, prediction)

    def forecast(self, inputs: ForecastInputs) -> ForecastTicket:
        if not isinstance(inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        regime = f"regime:{self.o1.state.regime_generation}"
        route = self._route(inputs, regime=regime)
        schema = feature_schema(
            inputs,
            selected_past=route.past_indices,
            selected_known=route.known_indices,
            seasonal_period=self.seasonal_period,
        )
        features = extract_features(
            inputs,
            selected_past=route.past_indices,
            selected_known=route.known_indices,
            seasonal_period=self.seasonal_period,
        )
        crystal_key = ForecastCrystalKey(
            model_identity=self.model_identity,
            feature_schema=schema,
            graph_revision=route.signature_sha256,
            regime=regime,
            horizon=inputs.horizon,
            target_count=inputs.target_dim,
        )
        profile = ActionProfileKey(**crystal_key.to_document())
        candidates: dict[str, tuple[ForecastPrediction, float]] = {}
        candidate_calibrations: dict[str, ConformalApplication] = {}

        if self.enable_seasonal:
            started = time.perf_counter()
            seasonal_base = self.seasonal.predict(inputs)
            seasonal_calibration = self._calibrate(
                profile,
                "seasonal",
                seasonal_base,
            )
            seasonal = (
                seasonal_base
                if seasonal_calibration is None
                else seasonal_calibration.prediction
            )
            candidates["seasonal"] = (seasonal, time.perf_counter() - started)
            if seasonal_calibration is not None:
                candidate_calibrations["seasonal"] = seasonal_calibration

        started = time.perf_counter()
        crystal_raw = self.crystals.predict(crystal_key, features, regime=regime)
        if crystal_raw is not None:
            crystal_base = _prediction_from_crystal(inputs, crystal_raw)
            crystal_calibration = self._calibrate(
                profile,
                "crystal",
                crystal_base,
            )
            candidates["crystal"] = (
                (
                    crystal_base
                    if crystal_calibration is None
                    else crystal_calibration.prediction
                ),
                time.perf_counter() - started,
            )
            if crystal_calibration is not None:
                candidate_calibrations["crystal"] = crystal_calibration

        recommendation = self.actions.recommend(profile)
        action = recommendation.action
        if action in candidates:
            prediction, latency = candidates[action]
            calibration = candidate_calibrations.get(action)
            base_prediction = (
                prediction if calibration is None else calibration.base_prediction
            )
            forwards = 0
        else:
            action = "foundation"
            started = time.perf_counter()
            base_prediction = self.teacher.predict(inputs)
            calibration = self._calibrate(
                profile,
                action,
                base_prediction,
            )
            prediction = (
                base_prediction if calibration is None else calibration.prediction
            )
            latency = time.perf_counter() - started
            forwards = 1
        prediction.validate_for(inputs)
        verify_prediction(prediction)
        self._sequence += 1
        ticket_id = sha256_json(
            {
                "action": action,
                "forecast_origin": int(inputs.past_timestamps[-1].astype(np.int64)),
                "profile": profile.to_document(),
                "sequence": self._sequence,
                "series_id": inputs.series_id,
            }
        )
        return ForecastTicket(
            ticket_id=ticket_id,
            inputs=inputs,
            prediction=prediction,
            base_prediction=base_prediction,
            calibration=calibration,
            action=action,
            profile=profile,
            crystal_key=crystal_key,
            features=features,
            latency_seconds=latency,
            foundation_forwards=forwards,
            route=route,
            regime=regime,
            candidates=candidates,
        )

    def _settle_graph(self, ticket: ForecastTicket, target: ForecastTarget) -> None:
        history = ticket.inputs.history_length
        for row, timestamp in enumerate(target.timestamps):
            targets = {
                self._target_name(index): float(target.values[row, index])
                for index in range(target.target_dim)
                if target.mask[row, index]
            }
            if not targets:
                continue
            known = {
                self._known_name(index): float(
                    ticket.inputs.known_covariates[history + row, index]
                )
                for index in range(ticket.inputs.known_covariate_dim)
                if ticket.inputs.known_covariate_mask[history + row, index]
            }
            self.graph.observe(
                timestamp_ns=int(timestamp.astype("datetime64[ns]").astype(np.int64)),
                targets=targets,
                known_future=known,
                observation_id=f"settle:{target.series_id}:{int(timestamp.astype(np.int64))}",
            )

    def settle(
        self,
        ticket: ForecastTicket,
        target: ForecastTarget,
    ) -> ForecastSettlement:
        if not isinstance(ticket, ForecastTicket):
            raise TypeError("ticket must be ForecastTicket")
        score = score_prediction(ticket.prediction, target)
        base_score = score_prediction(ticket.base_prediction, target)
        quantile_map = {}
        if ticket.prediction.quantiles is not None:
            assert ticket.prediction.quantile_levels is not None
            for index, raw_level in enumerate(ticket.prediction.quantile_levels):
                matches = tuple(
                    level
                    for level in self.o1.config.quantiles
                    if math.isclose(float(raw_level), level, abs_tol=1e-7)
                )
                if len(matches) == 1:
                    quantile_map[matches[0]] = ticket.prediction.quantiles[..., index]
        update = self.o1.observe(
            target.values,
            ticket.prediction.point,
            quantile_forecasts=quantile_map,
            mask=target.mask,
            observation_id=f"ticket:{ticket.ticket_id}",
        )
        calibration_update = None
        if self.conformal is not None and ticket.calibration is not None:
            calibration_update = self.conformal.observe(
                ticket.calibration.key,
                ticket.calibration,
                target,
                surprise=update.surprise,
                cusum=update.cusum,
                observation_id=f"ticket:{ticket.ticket_id}",
            )
        crystal_stats = self.crystals.observe(
            ticket.crystal_key,
            ticket.features,
            target.values,
            future_target_mask=target.mask,
            fit_target=(
                ticket.base_prediction.point if self.compile_teacher_crystals else None
            ),
            update_fit=(
                not self.compile_teacher_crystals
                or ticket.foundation_forwards == 1
                or ticket.action == "crystal"
            ),
            regime=ticket.regime,
        )

        settled_actions: set[str] = set()

        def settle_action(
            action: str,
            prediction: ForecastPrediction,
            latency: float,
            forwards: int,
        ) -> None:
            candidate_score = score_prediction(prediction, target)
            candidate_coverage = (
                0.0
                if candidate_score.interval_coverage is None
                else candidate_score.interval_coverage
            )
            self.actions.settle(
                ActionReceipt(
                    profile=ticket.profile,
                    action=action,
                    quality=candidate_score.quality,
                    coverage=candidate_coverage,
                    latency_seconds=latency,
                    forward_cost=float(forwards),
                )
            )
            settled_actions.add(action)

        settle_action(
            ticket.action,
            ticket.prediction,
            ticket.latency_seconds,
            ticket.foundation_forwards,
        )
        for action, (prediction, latency) in ticket.candidates.items():
            if action not in settled_actions:
                settle_action(action, prediction, latency, 0)
        return ForecastSettlement(
            ticket_id=ticket.ticket_id,
            action=ticket.action,
            score=score,
            base_score=base_score,
            o1_update=update,
            calibration_update=calibration_update,
            joint_covered=(
                None if calibration_update is None else calibration_update.joint_hit
            ),
            crystal_ready=False if crystal_stats is None else crystal_stats.ready,
            foundation_forwards=ticket.foundation_forwards,
            saved_foundation_forwards=int(ticket.action != "foundation"),
        )


__all__ = [
    "ForecastSettlement",
    "ForecastTicket",
    "O1FMRuntime",
    "RouteSelection",
]
