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
from .action_bank import ActionBank, ActionProfileKey, ActionReceipt
from .contracts import (
    ForecastExample,
    ForecastInputs,
    ForecastOutput,
    ForecastPrediction,
    ForecastTarget,
    ForecastTargets,
)
from .crystals import ForecastCrystalBank, ForecastCrystalKey
from .identity import CanonicalIdentity, canonical_json_bytes, sha256_json
from .losses import forecast_loss, monotone_penalty, pinball_loss
from .o1 import O1Config, O1State
from .model import ForecastBatch, ForecastResult, O1FMConfig, O1FMForecaster
from .synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    SyntheticSeries,
    generate_synthetic_dataset,
)

__all__ = [
    "ActionBank",
    "ActionProfileKey",
    "ActionReceipt",
    "CrossVariateSinkhornAttention",
    "CanonicalIdentity",
    "ForecastBatch",
    "ForecastBlock",
    "ForecastExample",
    "ForecastInputs",
    "ForecastCrystalBank",
    "ForecastCrystalKey",
    "ForecastOutput",
    "ForecastPrediction",
    "ForecastResult",
    "ForecastTarget",
    "ForecastTargets",
    "O1FMConfig",
    "O1FMForecaster",
    "O1Config",
    "O1State",
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
    "canonical_json_bytes",
    "sha256_json",
    "sinkhorn_transport",
]
