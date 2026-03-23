"""
Utilitaires pour le chargement et la préparation des données.
"""

from typing import Tuple, Optional
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


class MnistDataLoader:
    """
    Classe pour charger et préparer les données MNIST.

    Attributes:
        normalize: Si True, normalise les données entre 0 et 1
        flatten: Si True, aplatit les images en vecteurs
        one_hot: Si True, encode les labels en one-hot
    """

    def __init__(
        self,
        normalize: bool = True,
        flatten: bool = False,
        one_hot: bool = True
    ) -> None:
        """
        Initialise le data loader.

        Args:
            normalize: Normaliser les pixels entre 0 et 1
            flatten: Aplatir les images en vecteurs 1D
            one_hot: Encoder les labels en one-hot
        """
        self.normalize = normalize
        self.flatten = flatten
        self.one_hot = one_hot
        self.num_classes = 10

    def load_data(
        self
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Charge les données MNIST brutes.

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        return mnist.load_data()

    def preprocess_images(
        self,
        X: np.ndarray,
        for_cnn: bool = False
    ) -> np.ndarray:
        """
        Prétraite les images.

        Args:
            X: Images à prétraiter
            for_cnn: Si True, reshape pour CNN (ajoute dimension canal)

        Returns:
            Images prétraitées
        """
        X = X.astype('float32')

        if self.normalize:
            X = X / 255.0

        if for_cnn:
            # Reshape pour CNN: (samples, height, width, channels)
            if len(X.shape) == 3:
                X = X.reshape(-1, 28, 28, 1)
        elif self.flatten:
            # Reshape pour DNN: (samples, features)
            X = X.reshape(X.shape[0], -1)

        return X

    def preprocess_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Prétraite les labels.

        Args:
            y: Labels à prétraiter

        Returns:
            Labels prétraités
        """
        if self.one_hot:
            return to_categorical(y, self.num_classes)
        return y

    def load_and_preprocess(
        self,
        for_cnn: bool = False
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Charge et prétraite les données MNIST.

        Args:
            for_cnn: Si True, prépare les données pour un CNN

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        # Chargement
        (X_train, y_train), (X_test, y_test) = self.load_data()

        # Prétraitement des images
        X_train = self.preprocess_images(X_train, for_cnn=for_cnn)
        X_test = self.preprocess_images(X_test, for_cnn=for_cnn)

        # Prétraitement des labels
        y_train = self.preprocess_labels(y_train)
        y_test = self.preprocess_labels(y_test)

        print(f"Forme X_train: {X_train.shape}")
        print(f"Forme y_train: {y_train.shape}")
        print(f"Forme X_test: {X_test.shape}")
        print(f"Forme y_test: {y_test.shape}")

        return (X_train, y_train), (X_test, y_test)

    def create_validation_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_split: float = 0.1
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Crée un split de validation à partir des données d'entraînement.

        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            validation_split: Proportion pour la validation

        Returns:
            ((X_train, y_train), (X_val, y_val))
        """
        split_idx = int(len(X_train) * (1 - validation_split))

        X_train_split = X_train[:split_idx]
        y_train_split = y_train[:split_idx]
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]

        print(f"Train: {len(X_train_split)} samples")
        print(f"Validation: {len(X_val)} samples")

        return (X_train_split, y_train_split), (X_val, y_val)

    def __repr__(self) -> str:
        """Représentation string du data loader."""
        return (
            f"MnistDataLoader("
            f"normalize={self.normalize}, "
            f"flatten={self.flatten}, "
            f"one_hot={self.one_hot})"
        )
