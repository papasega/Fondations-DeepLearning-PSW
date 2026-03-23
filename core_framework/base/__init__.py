"""Classes de base abstraites pour l'architecture modulaire."""

from src.base.model_base import ModelBase, ModelProtocol
from src.base.trainer_base import TrainerBase, TrainerProtocol
from src.base.visualizer_base import VisualizerBase, VisualizerProtocol

__all__ = [
    "ModelBase",
    "ModelProtocol",
    "TrainerBase",
    "TrainerProtocol",
    "VisualizerBase",
    "VisualizerProtocol",
]
