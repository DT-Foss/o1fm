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
            quality_tolerance=0.005,
            coverage_tolerance=0.05,
            latency_cost_weight=0.0,
            forward_cost_weight=10.0,
        ),
        conformal=field,
        seasonal_period=seasonal_period,
        enable_seasonal=local_actions,
    )
    started = time.perf_counter()
    actions: list[str] = []
    scores: list[ForecastScore] = []
    base_scores: list[ForecastScore] = []
    joint: list[float] = []
    forwards = 0
    for example in online:
        ticket = runtime.forecast(example.inputs)
        settlement = runtime.settle(ticket, example.require_target())
        actions.append(ticket.action)
        scores.append(settlement.score)
        base_scores.append(settlement.base_score)
        if settlement.joint_covered is not None:
            joint.append(float(settlement.joint_covered))
        forwards += settlement.foundation_forwards
    elapsed = time.perf_counter() - started
    return {
        "actions": {
            "counts": {
                action: actions.count(action)
                for action in ("foundation", "crystal", "seasonal")
            },
            "trace": actions,
        },
        "base_quality": _score_summary(base_scores),
        "calibrated_quality": _score_summary(scores),
        "conformal": field.metrics(),
        "foundation_forwards": forwards,
        "graph": runtime.graph.metrics(),
        "joint_coverage": None if not joint else _mean(joint),
        "o1": runtime.o1.metrics(),
        "runtime_seconds": elapsed,
    }


def run_open_teacher_demo(
    *,
    teacher_name: str = "toto2",
    checkpoint: str | None = None,
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

    direct_started = time.perf_counter()
    direct_scores: list[ForecastScore] = []
    direct_joint: list[float] = []
    for example in online:
        prediction = teacher.predict(example.inputs)
        direct_scores.append(score_prediction(prediction, example.require_target()))
        direct_joint.append(_joint_hit(prediction, example.require_target()))
    direct_seconds = time.perf_counter() - direct_started

    temporary = tempfile.TemporaryDirectory() if state_dir is None else None
    context = temporary if temporary is not None else nullcontext()
    with context:
        root = (
            Path(temporary.name)
            if temporary is not None
            else Path(state_dir).expanduser().absolute()
        )
        root.mkdir(parents=True, exist_ok=True)
        control = _run_policy(
            teacher,
            online,
            root=root / "control",
            model_identity=model_identity,
            seasonal_period=config.seasonal_period,
            local_actions=False,
        )
        adaptive = _run_policy(
            teacher,
            online,
            root=root / "adaptive",
            model_identity=model_identity,
            seasonal_period=config.seasonal_period,
            local_actions=True,
        )

    windows = len(online)
    saved = windows - int(adaptive["foundation_forwards"])
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
            "forward_reduction": saved / windows,
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
        "teacher": {
            "adapter_identity_sha256": teacher.identity,  # type: ignore[attr-defined]
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
