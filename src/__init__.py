"""DeepGate-FaceRec: Face Recognition via Deep Metric & Embedding Learning."""

from src import data, gate, losses, metrics, mining, models, trainer
from src.data import (
    PKBatchSampler,
    build_identity_index_map,
    get_eval_transforms,
    get_train_transforms,
)
from src.dummy_functions import dummy_function
from src.gate import GateDecision, GateEvaluationResult, SmartGate
from src.losses import TripletMarginLoss
from src.models import FaceEmbeddingNet
from src.trainer import MetricTrainer

__version__ = "0.1.0"

__all__ = [
    "FaceEmbeddingNet",
    "GateDecision",
    "GateEvaluationResult",
    "MetricTrainer",
    "PKBatchSampler",
    "SmartGate",
    "TripletMarginLoss",
    "__version__",
    "build_identity_index_map",
    "data",
    "dummy_function",
    "gate",
    "get_eval_transforms",
    "get_train_transforms",
    "losses",
    "metrics",
    "mining",
    "models",
    "trainer",
]
