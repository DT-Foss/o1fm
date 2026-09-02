"""O1FM: continuous causal forecasting with stored compute."""

from .action_bank import ActionBank, ActionProfileKey, ActionReceipt
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
from .baselines import SeasonalNaiveForecaster
from .causal_graph import CausalEdge, CausalGraphConfig, OnlineCausalGraph
from .contracts import (
    ForecastExample,
    ForecastInputs,
    ForecastOutput,
    ForecastPrediction,
    ForecastTarget,
    ForecastTargets,
)
from .crystals import ForecastCrystalBank, ForecastCrystalKey
from .features import FEATURE_ABI, extract_features, feature_schema
from .identity import CanonicalIdentity, canonical_json_bytes, sha256_json
from .losses import forecast_loss, monotone_penalty, pinball_loss
from .metrics import ForecastScore, score_prediction, verify_prediction
from .model import ForecastBatch, ForecastResult, O1FMConfig, O1FMForecaster
from .o1 import O1Config, O1ErrorStream, O1State
from .runtime import ForecastSettlement, ForecastTicket, O1FMRuntime
from .synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    SyntheticSeries,
    generate_synthetic_dataset,
)
from .training import model_identity, predict, tensorize_inputs, train_forecaster

__version__ = "0.1.0"

__all__ = [
    "ActionBank",
    "ActionProfileKey",
    "ActionReceipt",
    "CanonicalIdentity",
    "CausalEdge",
    "CausalGraphConfig",
    "CrossVariateSinkhornAttention",
    "FEATURE_ABI",
    "ForecastBatch",
    "ForecastBlock",
    "ForecastCrystalBank",
    "ForecastCrystalKey",
    "ForecastExample",
    "ForecastInputs",
    "ForecastOutput",
    "ForecastPrediction",
    "ForecastResult",
    "ForecastScore",
    "ForecastSettlement",
    "ForecastTarget",
    "ForecastTargets",
    "ForecastTicket",
    "O1Config",
    "O1ErrorStream",
    "O1FMConfig",
    "O1FMForecaster",
    "O1FMRuntime",
    "O1State",
    "OnlineCausalGraph",
    "RMSNorm",
    "SeasonalNaiveForecaster",
    "SwiGLU",
    "SyntheticConfig",
    "SyntheticDataset",
    "SyntheticSeries",
    "TemporalSelfAttention",
    "canonical_json_bytes",
    "extract_features",
    "feature_schema",
    "forecast_loss",
    "generate_synthetic_dataset",
    "model_identity",
    "monotone_penalty",
    "patchify",
    "pinball_loss",
    "predict",
    "revin",
    "score_prediction",
    "sha256_json",
    "sinkhorn_transport",
    "tensorize_inputs",
    "train_forecaster",
    "verify_prediction",
]
