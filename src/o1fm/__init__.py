"""O1FM: a compact multivariate forecasting backbone."""

from .attention import (
    CrossVariateSinkhornAttention,
    ForecastBlock,
    RMSNorm,
    SwiGLU,
    TemporalSelfAttention,
    patchify,
    revin,
    sinkhorn_transport,
)
from .contracts import (
    ForecastExample,
    ForecastInputs,
    ForecastOutput,
    ForecastPrediction,
    ForecastTarget,
    ForecastTargets,
)
from .losses import forecast_loss, monotone_penalty, pinball_loss
from .model import ForecastBatch, ForecastResult, O1FMConfig, O1FMForecaster
from .synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    SyntheticSeries,
    generate_synthetic_dataset,
)

__all__ = [
    "CrossVariateSinkhornAttention",
    "ForecastBatch",
    "ForecastBlock",
    "ForecastExample",
    "ForecastInputs",
    "ForecastOutput",
    "ForecastPrediction",
    "ForecastResult",
    "ForecastTarget",
    "ForecastTargets",
    "O1FMConfig",
    "O1FMForecaster",
    "RMSNorm",
    "SwiGLU",
    "SyntheticConfig",
    "SyntheticDataset",
    "SyntheticSeries",
    "TemporalSelfAttention",
    "forecast_loss",
    "generate_synthetic_dataset",
    "monotone_penalty",
    "patchify",
    "pinball_loss",
    "revin",
    "sinkhorn_transport",
]
