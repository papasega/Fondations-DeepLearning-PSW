"""
Classes de base abstraites pour l'entraînement des modèles.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Tuple, Dict, Any, Optional
import numpy as np
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import History


class TrainerProtocol(Protocol):
    """
    Protocol définissant l'interface qu'un trainer doit implémenter.
    """

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> History:
        """Entraîne le modèle."""
        ...

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Évalue le modèle."""
        ...


class TrainerBase(ABC):
    """
    Classe de base abstraite pour l'entraînement des modèles.

    Définit l'interface commune pour tous les trainers et implémente
    des fonctionnalités communes d'entraînement.

    Attributes:
        model: Le modèle à entraîner
        batch_size: Taille des batchs
        epochs: Nombre d'epochs
        verbose: Niveau de verbosité
    """

    def __init__(
        self,
        model: KerasModel,
        batch_size: int = 128,
        epochs: int = 10,
        verbose: int = 1,
        **kwargs: Any
    ) -> None:
        """
        Initialise le trainer.

        Args:
            model: Modèle Keras à entraîner
            batch_size: Taille des batchs
            epochs: Nombre d'epochs
            verbose: Niveau de verbosité
            **kwargs: Arguments supplémentaires
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self._training_config = kwargs
        self.history: Optional[History] = None

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> History:
        """
        Entraîne le modèle.

        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_val: Données de validation (optionnel)
            y_val: Labels de validation (optionnel)
            **kwargs: Arguments supplémentaires

        Returns:
            Historique d'entraînement
        """
        pass

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Évalue le modèle sur les données de test.

        Args:
            X_test: Données de test
            y_test: Labels de test

        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        results = self.model.evaluate(X_test, y_test, verbose=self.verbose)
        metric_names = self.model.metrics_names

        return dict(zip(metric_names, results))

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Fait des prédictions.

        Args:
            X: Données d'entrée
            **kwargs: Arguments supplémentaires

        Returns:
            Prédictions
        """
        return self.model.predict(X, **kwargs)

    def get_history(self) -> Optional[History]:
        """
        Retourne l'historique d'entraînement.

        Returns:
            Historique ou None si pas encore entraîné
        """
        return self.history

    @property
    def training_config(self) -> Dict[str, Any]:
        """Retourne la configuration d'entraînement."""
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "verbose": self.verbose,
            **self._training_config
        }

    def __repr__(self) -> str:
        """Représentation string du trainer."""
        return (
            f"{self.__class__.__name__}("
            f"batch_size={self.batch_size}, "
            f"epochs={self.epochs})"
        )
