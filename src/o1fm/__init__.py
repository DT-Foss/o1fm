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
from .conformal import (
    ConformalApplication,
    ConformalConflictError,
    ConformalFieldConfig,
    ConformalFieldKey,
    ConformalIntegrityError,
    ConformalUpdate,
    O1ConformalField,
)
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
from .open_teacher_demo import run_open_teacher_demo
from .runtime import ForecastSettlement, ForecastTicket, O1FMRuntime
from .synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    SyntheticSeries,
    generate_synthetic_dataset,
)
from .teachers import (
    Chronos2Teacher,
    ForecastTeacher,
    NativeO1FMTeacher,
    Toto2Teacher,
    teacher_identity,
)
from .training import model_identity, predict, tensorize_inputs, train_forecaster
from .transport import (
    CausalTransportResult,
    CausalTransportStackResult,
    RouteTransportResult,
    causal_transport_stack,
    causal_transport_w1,
    route_transport_w1,
)

__version__ = "0.1.0"

__all__ = [
    "ActionBank",
    "ActionProfileKey",
    "ActionReceipt",
    "CanonicalIdentity",
    "CausalEdge",
    "CausalGraphConfig",
    "CausalTransportResult",
    "CausalTransportStackResult",
    "ConformalApplication",
    "ConformalConflictError",
    "ConformalFieldConfig",
    "ConformalFieldKey",
    "ConformalIntegrityError",
    "ConformalUpdate",
    "Chronos2Teacher",
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
    "ForecastTeacher",
    "ForecastTicket",
    "O1Config",
    "O1ErrorStream",
    "O1FMConfig",
    "O1FMForecaster",
    "O1FMRuntime",
    "O1ConformalField",
    "NativeO1FMTeacher",
    "O1State",
    "OnlineCausalGraph",
    "RMSNorm",
    "RouteTransportResult",
    "SeasonalNaiveForecaster",
    "SwiGLU",
    "SyntheticConfig",
    "SyntheticDataset",
    "SyntheticSeries",
    "TemporalSelfAttention",
    "Toto2Teacher",
    "canonical_json_bytes",
    "causal_transport_stack",
    "causal_transport_w1",
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
    "route_transport_w1",
    "run_open_teacher_demo",
    "score_prediction",
    "sha256_json",
    "sinkhorn_transport",
    "tensorize_inputs",
    "teacher_identity",
    "train_forecaster",
    "verify_prediction",
]
