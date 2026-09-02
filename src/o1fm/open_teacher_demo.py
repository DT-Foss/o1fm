"""Real open-weight teacher proof through the complete O1FM control plane."""

from __future__ import annotations

import hashlib
import platform
import sys
import tempfile
import time
from contextlib import nullcontext
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from .action_bank import ActionBank
from .atlas import ForecastAtlas
from .causal_graph import CausalGraphConfig, OnlineCausalGraph
from .conformal import ConformalFieldConfig, O1ConformalField
from .crystals import ForecastCrystalBank
from .metrics import ForecastScore, score_prediction
from .o1 import O1Config, O1ErrorStream
from .runtime import O1FMRuntime
from .synthetic import SyntheticConfig, generate_synthetic_dataset
from .teachers import Chronos2Teacher, ForecastTeacher, Toto2Teacher


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _module_sha256(module: object) -> str | None:
    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        return None
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        value = tensor.detach().cpu().contiguous()
        descriptor = f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode()
        digest.update(descriptor)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _load_teacher(
    teacher: str,
    *,
    checkpoint: str | None,
    device: str,
) -> ForecastTeacher:
    if teacher == "toto2":
        return Toto2Teacher.from_pretrained(
            "Datadog/Toto-2.0-22m" if checkpoint is None else checkpoint,
            device=device,
        )
    if teacher == "chronos2":
        return Chronos2Teacher.from_pretrained(
            "amazon/chronos-2" if checkpoint is None else checkpoint,
            device=device,
        )
    raise ValueError("teacher must be toto2 or chronos2")


def _joint_hit(score_prediction_source, target) -> float:
    assert score_prediction_source.quantiles is not None
    observed = np.asarray(target.mask, dtype=np.bool_)
    lower = score_prediction_source.quantiles[..., 0]
    upper = score_prediction_source.quantiles[..., -1]
    return float(
        (
            (target.values[observed] >= lower[observed])
            & (target.values[observed] <= upper[observed])
        ).all()
    )


def _score_summary(scores: list[ForecastScore]) -> dict[str, float]:
    return {
        "coverage": _mean(
            [
                0.0 if score.interval_coverage is None else score.interval_coverage
                for score in scores
            ]
        ),
        "mae": _mean([score.mae for score in scores]),
        "mean_width": _mean(
            [
                0.0 if score.interval_width is None else score.interval_width
                for score in scores
            ]
        ),
        "pinball": _mean(
            [score.mae if score.pinball is None else score.pinball for score in scores]
        ),
    }


def _run_policy(
    teacher: ForecastTeacher,
    online,
    *,
    root: Path,
    model_identity: str,
    seasonal_period: int,
    local_actions: bool,
    local_teacher: ForecastTeacher | None = None,
    local_forward_cost: float = 0.0,
    quality_tolerance: float = 0.005,
    coverage_tolerance: float = 0.05,
    atlas: ForecastAtlas | None = None,
) -> dict[str, object]:
    field = O1ConformalField(
        root / "conformal",
        config=ConformalFieldConfig(target_coverage=0.8),
    )
    runtime = O1FMRuntime(
        teacher,
        model_identity=model_identity,
        o1=O1ErrorStream(
            root / "o1.json",
            stream_id=online[0].series_id,
            config=O1Config(quantiles=(0.1, 0.5, 0.9), cusum_threshold=4.0),
        ),
        graph=OnlineCausalGraph(
            root / "graph",
            graph_id=online[0].series_id,
            config=CausalGraphConfig(
                min_observations=4,
                max_lag=4,
                top_k=4,
                min_score=0.05,
            ),
        ),
        crystals=ForecastCrystalBank(
            root / "crystals.json",
            ridge=0.1,
            min_support=4,
            min_sequential_support=2,
            quantile_levels=teacher.quantile_levels,
        ),
        actions=ActionBank(
            root / "actions.json",
            min_support=2 if local_actions else 1_000_000,
            quality_tolerance=quality_tolerance,
            coverage_tolerance=coverage_tolerance,
            latency_cost_weight=0.0,
            forward_cost_weight=10.0,
        ),
        conformal=field,
        local_teacher=local_teacher,
        local_forward_cost=local_forward_cost,
        foundation_refresh_interval=4,
        seasonal_period=seasonal_period,
        enable_seasonal=local_actions and local_teacher is None,
        atlas=atlas,
    )
    started = time.perf_counter()
    actions: list[str] = []
    route_transport: list[float] = []
    scores: list[ForecastScore] = []
    base_scores: list[ForecastScore] = []
    joint: list[float] = []
    forwards = 0
    local_forwards = 0
    forced_refreshes = 0
    profiles = set()
    atlas_receipts = []
    for example in online:
        ticket = runtime.forecast(example.inputs)
        settlement = runtime.settle(ticket, example.require_target())
        actions.append(ticket.action)
        profiles.add(ticket.profile)
        for receipt in (
            ticket.atlas_forecast_receipt,
            ticket.atlas_transport_receipt,
        ):
            if receipt is not None:
                atlas_receipts.append(receipt)
        route_transport.append(ticket.route.transport_w1)
        scores.append(settlement.score)
        base_scores.append(settlement.base_score)
        if settlement.joint_covered is not None:
            joint.append(float(settlement.joint_covered))
        forwards += settlement.foundation_forwards
        local_forwards += settlement.local_forwards
        forced_refreshes += int(settlement.forced_foundation_refresh)
        if settlement.atlas_settlement_receipt is not None:
            atlas_receipts.append(settlement.atlas_settlement_receipt)
    elapsed = time.perf_counter() - started
    verified: dict[str, dict[str, float | int]] = {}
    for profile in profiles:
        for action, stats in runtime.actions.describe(profile).items():
            row = verified.setdefault(
                action,
                {
                    "coverage_sum": 0.0,
                    "forward_cost_sum": 0.0,
                    "quality_sum": 0.0,
                    "support": 0,
                },
            )
            row["support"] = int(row["support"]) + stats.support
            row["coverage_sum"] = float(row["coverage_sum"]) + (
                stats.mean_coverage * stats.support
            )
            row["forward_cost_sum"] = float(row["forward_cost_sum"]) + (
                stats.mean_forward_cost * stats.support
            )
            row["quality_sum"] = float(row["quality_sum"]) + (
                stats.mean_quality * stats.support
            )
    verified_actions = {
        action: {
            "mean_coverage": float(row["coverage_sum"]) / int(row["support"]),
            "mean_forward_cost": float(row["forward_cost_sum"]) / int(row["support"]),
            "mean_quality": float(row["quality_sum"]) / int(row["support"]),
            "support": int(row["support"]),
        }
        for action, row in sorted(verified.items())
    }
    return {
        "actions": {
            "counts": {
                action: actions.count(action)
                for action in ("foundation", "local", "crystal", "seasonal")
            },
            "trace": actions,
        },
        "atlas": (
            None
            if atlas is None
            else {
                "receipts": [
                    {
                        "kind": receipt.kind,
                        "record_count": receipt.record_count,
                        "revision_sequence": receipt.revision_sequence,
                        "revision_sha256": receipt.revision_sha256,
                        "segment_sha256": receipt.segment_sha256,
                    }
                    for receipt in atlas_receipts
                ],
                "revision": list(atlas.store.revision()),
                "verified": atlas.verify(),
            }
        ),
        "base_quality": _score_summary(base_scores),
        "calibrated_quality": _score_summary(scores),
        "conformal": field.metrics(),
        "foundation_forwards": forwards,
        "forced_foundation_refreshes": forced_refreshes,
        "local_forwards": local_forwards,
        "graph": {
            **runtime.graph.metrics(),
            "route_transport_w1_max": max(route_transport, default=0.0),
            "route_transport_w1_mean": _mean(route_transport),
        },
        "joint_coverage": None if not joint else _mean(joint),
        "o1": runtime.o1.metrics(),
        "runtime_seconds": elapsed,
        "verified_actions": verified_actions,
    }


def run_open_teacher_demo(
    *,
    teacher_name: str = "toto2",
    checkpoint: str | None = None,
    local_checkpoint: str | None = None,
    enable_local_teacher: bool = True,
    quality_tolerance: float = 0.005,
    coverage_tolerance: float = 0.05,
    atlas_dir: str | Path | None = None,
    device: str = "cpu",
    seed: int = 23,
    state_dir: str | Path | None = None,
) -> dict[str, object]:
    """Load one permissive teacher and run control plus adaptive policies."""

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
    online = tuple(row for row in dataset.online if row.series_id == series_id)
    load_started = time.perf_counter()
    teacher = _load_teacher(
        teacher_name,
        checkpoint=checkpoint,
        device=device,
    )
    load_seconds = time.perf_counter() - load_started
    local_teacher = None
    local_load_seconds = 0.0
    if enable_local_teacher and teacher_name == "toto2":
        local_started = time.perf_counter()
        local_teacher = _load_teacher(
            "toto2",
            checkpoint=(
                "Datadog/Toto-2.0-4m" if local_checkpoint is None else local_checkpoint
            ),
            device=device,
        )
        local_load_seconds = time.perf_counter() - local_started
    model = getattr(teacher, "model", None)
    if model is None:
        model = getattr(getattr(teacher, "pipeline", None), "model", None)
    weights_sha256 = _module_sha256(model)
    model_identity = weights_sha256 or teacher.identity  # type: ignore[attr-defined]
    parameter_count = (
        None
        if model is None or not callable(getattr(model, "parameters", None))
        else sum(parameter.numel() for parameter in model.parameters())
    )
    local_model = (
        None if local_teacher is None else getattr(local_teacher, "model", None)
    )
    local_weights_sha256 = _module_sha256(local_model)
    local_parameter_count = (
        None
        if local_model is None
        else sum(parameter.numel() for parameter in local_model.parameters())
    )
    local_forward_cost = (
        0.0
        if local_parameter_count is None or parameter_count in (None, 0)
        else local_parameter_count / parameter_count
    )

    direct_started = time.perf_counter()
    direct_scores: list[ForecastScore] = []
    direct_joint: list[float] = []
    for example in online:
        prediction = teacher.predict(example.inputs)
        direct_scores.append(score_prediction(prediction, example.require_target()))
        direct_joint.append(_joint_hit(prediction, example.require_target()))
    direct_seconds = time.perf_counter() - direct_started
    local_direct_scores: list[ForecastScore] = []
    local_direct_joint: list[float] = []
    local_direct_seconds = 0.0
    if local_teacher is not None:
        local_direct_started = time.perf_counter()
        for example in online:
            prediction = local_teacher.predict(example.inputs)
            local_direct_scores.append(
                score_prediction(prediction, example.require_target())
            )
            local_direct_joint.append(_joint_hit(prediction, example.require_target()))
        local_direct_seconds = time.perf_counter() - local_direct_started

    temporary = tempfile.TemporaryDirectory() if state_dir is None else None
    context = temporary if temporary is not None else nullcontext()
    with context:
        root = (
            Path(temporary.name)
            if temporary is not None
            else Path(state_dir).expanduser().absolute()
        )
        root.mkdir(parents=True, exist_ok=True)
        atlas = None if atlas_dir is None else ForecastAtlas.from_immer(atlas_dir)
        control = _run_policy(
            teacher,
            online,
            root=root / "control",
            model_identity=model_identity,
            seasonal_period=config.seasonal_period,
            local_actions=False,
            quality_tolerance=quality_tolerance,
            coverage_tolerance=coverage_tolerance,
            atlas=None,
        )
        adaptive = _run_policy(
            teacher,
            online,
            root=root / "adaptive",
            model_identity=model_identity,
            seasonal_period=config.seasonal_period,
            local_actions=True,
            local_teacher=local_teacher,
            local_forward_cost=local_forward_cost,
            quality_tolerance=quality_tolerance,
            coverage_tolerance=coverage_tolerance,
            atlas=atlas,
        )

    windows = len(online)
    saved = windows - int(adaptive["foundation_forwards"])
    local_forwards = int(adaptive["local_forwards"])
    equivalent_forwards = int(adaptive["foundation_forwards"]) + (
        local_forwards * local_forward_cost
    )
    control_quality = control["calibrated_quality"]
    adaptive_quality = adaptive["calibrated_quality"]
    assert isinstance(control_quality, dict) and isinstance(adaptive_quality, dict)
    result = {
        "adaptive": adaptive,
        "control": control,
        "data": {
            "history": config.history_length,
            "horizon": config.horizon,
            "known_covariates": dataset.known_covariate_dim,
            "missing_rate": config.missing_rate,
            "online_windows": windows,
            "past_covariates": config.past_covariate_dim,
            "regime_switch_indices": list(dataset.regime_switch_indices),
            "targets": config.target_dim,
        },
        "deltas_vs_calibrated_control": {
            "coverage": float(adaptive_quality["coverage"])
            - float(control_quality["coverage"]),
            "mae": float(adaptive_quality["mae"]) - float(control_quality["mae"]),
            "pinball": float(adaptive_quality["pinball"])
            - float(control_quality["pinball"]),
        },
        "direct_raw_teacher": {
            **_score_summary(direct_scores),
            "joint_coverage": _mean(direct_joint),
            "seconds": direct_seconds,
        },
        "economics": {
            "baseline_teacher_forwards": windows,
            "executed_teacher_forwards": int(adaptive["foundation_forwards"]),
            "executed_local_forwards": local_forwards,
            "foundation_equivalent_forwards": equivalent_forwards,
            "forward_reduction": saved / windows,
            "foundation_equivalent_reduction": 1.0 - equivalent_forwards / windows,
            "saved_teacher_forwards": saved,
        },
        "environment": {
            "device": device,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "schema": "o1fm.open-teacher-demo/v1",
        "policy": {
            "coverage_tolerance": coverage_tolerance,
            "quality_tolerance": quality_tolerance,
        },
        "local_teacher": (
            None
            if local_teacher is None
            else {
                "adapter_identity_sha256": local_teacher.identity,  # type: ignore[attr-defined]
                "capabilities": local_teacher.capabilities.to_record(),  # type: ignore[attr-defined]
                "checkpoint": local_teacher.checkpoint,  # type: ignore[attr-defined]
                "forward_cost": local_forward_cost,
                "load_seconds": local_load_seconds,
                "parameter_count": local_parameter_count,
                "weights_sha256": local_weights_sha256,
            }
        ),
        "local_raw_teacher": (
            None
            if not local_direct_scores
            else {
                **_score_summary(local_direct_scores),
                "joint_coverage": _mean(local_direct_joint),
                "seconds": local_direct_seconds,
            }
        ),
        "teacher": {
            "adapter_identity_sha256": teacher.identity,  # type: ignore[attr-defined]
            "capabilities": teacher.capabilities.to_record(),  # type: ignore[attr-defined]
            "checkpoint": teacher.checkpoint,  # type: ignore[attr-defined]
            "load_seconds": load_seconds,
            "name": teacher_name,
            "package_versions": {
                "chronos-forecasting": _package_version("chronos-forecasting"),
                "toto-2": _package_version("toto-2"),
            },
            "parameter_count": parameter_count,
            "quantile_levels": [float(level) for level in teacher.quantile_levels],
            "weights_sha256": weights_sha256,
        },
    }
    if saved <= 0:
        raise RuntimeError("adaptive policy did not replace a teacher forward")
    return result


__all__ = ["run_open_teacher_demo"]
