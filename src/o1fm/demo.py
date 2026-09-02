"""End-to-end synthetic proof for the O1FM execution architecture."""

from __future__ import annotations

import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .action_bank import ActionBank
from .baselines import SeasonalNaiveForecaster
from .causal_graph import CausalGraphConfig, OnlineCausalGraph
from .crystals import ForecastCrystalBank
from .metrics import score_prediction
from .model import O1FMConfig, O1FMForecaster
from .o1 import O1Config, O1ErrorStream
from .runtime import O1FMRuntime
from .synthetic import SyntheticConfig, generate_synthetic_dataset
from .training import model_identity, predict, train_forecaster


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def run_demo(
    *,
    seed: int = 7,
    epochs: int = 20,
    state_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train a compact backbone, then replace verified online forwards."""

    config = SyntheticConfig(
        seed=seed,
        total_steps=320,
        train_fraction=0.55,
        missing_rate=0.0,
        history_length=48,
        horizon=12,
        train_stride=6,
        online_stride=12,
    )
    dataset = generate_synthetic_dataset(config)
    torch.manual_seed(seed)
    model = O1FMForecaster(
        O1FMConfig(
            patch_len=4,
            d_model=48,
            num_heads=4,
            num_layers=2,
        )
    )
    started = time.perf_counter()
    training = train_forecaster(
        model,
        dataset.train,
        epochs=epochs,
        batch_size=16,
        learning_rate=1e-3,
        seed=seed,
    )
    training_seconds = time.perf_counter() - started
    identity = model_identity(model)
    series_id = dataset.series[0].series_id
    online = tuple(row for row in dataset.online if row.series_id == series_id)

    direct_mae = []
    direct_pinball = []
    seasonal_mae = []
    seasonal = SeasonalNaiveForecaster(
        period=config.seasonal_period,
        quantile_levels=model.config.quantiles,
    )
    for example in online:
        direct = score_prediction(
            predict(model, example.inputs), example.require_target()
        )
        baseline = score_prediction(
            seasonal.predict(example.inputs), example.require_target()
        )
        direct_mae.append(direct.mae)
        direct_pinball.append(direct.mae if direct.pinball is None else direct.pinball)
        seasonal_mae.append(baseline.mae)

    temporary = tempfile.TemporaryDirectory() if state_dir is None else None
    context = temporary if temporary is not None else nullcontext()
    with context:
        root = (
            Path(temporary.name)
            if temporary is not None
            else Path(state_dir).expanduser().absolute()
        )
        root.mkdir(parents=True, exist_ok=True)
        crystals = ForecastCrystalBank(
            root / "forecast-crystals.json",
            ridge=0.1,
            min_support=4,
            min_sequential_support=2,
            quantile_levels=model.config.quantiles,
        )
        actions = ActionBank(
            root / "action-bank.json",
            min_support=2,
            quality_tolerance=0.08,
            coverage_tolerance=0.20,
            latency_cost_weight=0.0,
            forward_cost_weight=10.0,
        )
        runtime = O1FMRuntime(
            model,
            model_identity=identity,
            o1=O1ErrorStream(
                root / "o1-state.json",
                stream_id=series_id,
                config=O1Config(
                    quantiles=(0.1, 0.5, 0.9),
                    cusum_threshold=4.0,
                ),
            ),
            graph=OnlineCausalGraph(
                root / "causal-graph",
                graph_id=series_id,
                config=CausalGraphConfig(
                    min_observations=4,
                    max_lag=4,
                    top_k=4,
                    min_score=0.05,
                ),
            ),
            crystals=crystals,
            actions=actions,
            seasonal_period=config.seasonal_period,
            enable_seasonal=False,
        )
        selected_mae = []
        selected_pinball = []
        action_trace = []
        foundation_forwards = 0
        saved_forwards = 0
        runtime_started = time.perf_counter()
        for example in online:
            ticket = runtime.forecast(example.inputs)
            settlement = runtime.settle(ticket, example.require_target())
            action_trace.append(ticket.action)
            selected_mae.append(settlement.score.mae)
            selected_pinball.append(
                settlement.score.mae
                if settlement.score.pinball is None
                else settlement.score.pinball
            )
            foundation_forwards += settlement.foundation_forwards
            saved_forwards += settlement.saved_foundation_forwards
        runtime_seconds = time.perf_counter() - runtime_started
        graph_metrics = runtime.graph.metrics()
        o1_metrics = runtime.o1.metrics()

    baseline_forwards = len(online)
    result = {
        "actions": {
            "crystal": action_trace.count("crystal"),
            "foundation": action_trace.count("foundation"),
            "seasonal": action_trace.count("seasonal"),
            "trace": action_trace,
        },
        "architecture": {
            "causal_temporal_attention": True,
            "forecast_crystals": True,
            "known_future_covariates": True,
            "non_autoregressive_horizon": True,
            "o1_regime_state": True,
            "sinkhorn_variate_attention": True,
        },
        "data": {
            "horizon": config.horizon,
            "online_windows": len(online),
            "regime_switch_indices": list(dataset.regime_switch_indices),
            "target_dim": config.target_dim,
            "train_windows": len(dataset.train),
        },
        "economics": {
            "baseline_foundation_forwards": baseline_forwards,
            "executed_foundation_forwards": foundation_forwards,
            "forward_reduction": saved_forwards / max(1, baseline_forwards),
            "saved_foundation_forwards": saved_forwards,
        },
        "graph": graph_metrics,
        "model_identity_sha256": identity,
        "o1": o1_metrics,
        "quality": {
            "direct_foundation_mae": _mean(direct_mae),
            "direct_foundation_pinball": _mean(direct_pinball),
            "runtime_mae": _mean(selected_mae),
            "runtime_pinball": _mean(selected_pinball),
            "seasonal_mae": _mean(seasonal_mae),
        },
        "schema": "o1fm.demo-result/v1",
        "timing": {
            "runtime_seconds": runtime_seconds,
            "training_seconds": training_seconds,
        },
        "training": asdict(training),
    }
    if result["economics"]["saved_foundation_forwards"] <= 0:  # type: ignore[index]
        raise RuntimeError("demo did not release any foundation forward")
    if result["quality"]["runtime_mae"] > (  # type: ignore[index,operator]
        result["quality"]["direct_foundation_mae"] * 1.20  # type: ignore[index,operator]
    ):
        raise RuntimeError("stored compute exceeded the runtime quality contract")
    return result


__all__ = ["run_demo"]
