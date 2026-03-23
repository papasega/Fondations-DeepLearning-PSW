"""
Classes de base abstraites pour les modèles de deep learning.

Utilise ABC (Abstract Base Classes) et Protocol pour définir
les interfaces des modèles.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Tuple, Any, Optional
import numpy as np
from tensorflow.keras.models import Model as KerasModel


class ModelProtocol(Protocol):
    """
    Protocol définissant l'interface qu'un modèle doit implémenter.

    Utilise le Protocol de typing pour le duck typing structurel.
    """

    def build(self) -> KerasModel:
        """Construit et retourne le modèle."""
        ...

    def compile_model(self, model: KerasModel) -> None:
        """Compile le modèle avec l'optimiseur et la loss."""
        ...

    def get_model(self) -> KerasModel:
        """Retourne le modèle Keras."""
        ...

    def summary(self) -> None:
        """Affiche un résumé du modèle."""
        ...


class ModelBase(ABC):
    """
    Classe de base abstraite pour tous les modèles de deep learning.

    Définit l'interface commune que tous les modèles doivent implémenter.
    Utilise le pattern Template Method pour structurer la création des modèles.

    Attributes:
        input_shape: Forme de l'entrée du modèle
        num_classes: Nombre de classes pour la classification
        model: Instance du modèle Keras
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        **kwargs: Any
    ) -> None:
        """
        Initialise le modèle de base.

        Args:
            input_shape: Forme de l'entrée (ex: (28, 28, 1))
            num_classes: Nombre de classes de sortie
            **kwargs: Arguments supplémentaires spécifiques au modèle
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model: Optional[KerasModel] = None
        self._hyperparameters = kwargs

    @abstractmethod
    def build(self) -> KerasModel:
        """
        Construit l'architecture du modèle.

        Cette méthode doit être implémentée par les sous-classes
        pour définir l'architecture spécifique du modèle.

        Returns:
            Instance du modèle Keras
        """
        pass

    @abstractmethod
    def compile_model(self, model: KerasModel) -> None:
        """
        Compile le modèle avec l'optimiseur, la loss et les métriques.

        Args:
            model: Le modèle Keras à compiler
        """
        pass

    def get_model(self) -> KerasModel:
        """
        Retourne le modèle Keras.

        Construit le modèle s'il n'existe pas encore.

        Returns:
            Instance du modèle Keras

        Raises:
            ValueError: Si le modèle n'a pas été construit
        """
        if self.model is None:
            self.model = self.build()
            self.compile_model(self.model)
        return self.model

    def summary(self) -> None:
        """Affiche un résumé de l'architecture du modèle."""
        model = self.get_model()
        model.summary()

    def save(self, filepath: str) -> None:
        """
        Sauvegarde le modèle.

        Args:
            filepath: Chemin où sauvegarder le modèle
        """
        if self.model is None:
            raise ValueError("Le modèle doit être construit avant d'être sauvegardé")
        self.model.save(filepath)
        print(f"Modèle sauvegardé: {filepath}")

    def load_weights(self, filepath: str) -> None:
        """
        Charge les poids du modèle.

        Args:
            filepath: Chemin du fichier de poids
        """
        model = self.get_model()
        model.load_weights(filepath)
        print(f"Poids chargés depuis: {filepath}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions sur les données.

        Args:
            X: Données d'entrée

        Returns:
            Prédictions du modèle
        """
        model = self.get_model()
        return model.predict(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: int = 1
    ) -> Tuple[float, ...]:
        """
        Évalue le modèle sur les données.

        Args:
            X: Données d'entrée
            y: Labels
            verbose: Niveau de verbosité

        Returns:
            Métriques d'évaluation (loss, accuracy, etc.)
        """
        model = self.get_model()
        return model.evaluate(X, y, verbose=verbose)

    @property
    def hyperparameters(self) -> dict:
        """Retourne les hyperparamètres du modèle."""
        return self._hyperparameters

    def __repr__(self) -> str:
        """Représentation string du modèle."""
        return (
            f"{self.__class__.__name__}("
            f"input_shape={self.input_shape}, "
            f"num_classes={self.num_classes})"
        )
