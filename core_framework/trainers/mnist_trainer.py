"""
Trainer pour les modèles MNIST.
"""

from typing import Optional, Any
import numpy as np
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import History, EarlyStopping, ReduceLROnPlateau

from src.base.trainer_base import TrainerBase


class MnistTrainer(TrainerBase):
    """
    Trainer spécialisé pour l'entraînement des modèles MNIST.

    Implémente des callbacks utiles comme early stopping et
    réduction du learning rate.

    Attributes:
        use_early_stopping: Active l'early stopping
        use_reduce_lr: Active la réduction du learning rate
        patience: Patience pour l'early stopping
    """

    def __init__(
        self,
        model: KerasModel,
        batch_size: int = 128,
        epochs: int = 10,
        verbose: int = 1,
        use_early_stopping: bool = False,
        use_reduce_lr: bool = False,
        patience: int = 5,
        **kwargs: Any
    ) -> None:
        """
        Initialise le trainer MNIST.

        Args:
            model: Modèle Keras à entraîner
            batch_size: Taille des batchs
            epochs: Nombre d'epochs
            verbose: Niveau de verbosité
            use_early_stopping: Utiliser l'early stopping
            use_reduce_lr: Utiliser la réduction du learning rate
            patience: Patience pour les callbacks
            **kwargs: Arguments supplémentaires
        """
        super().__init__(model, batch_size, epochs, verbose, **kwargs)
        self.use_early_stopping = use_early_stopping
        self.use_reduce_lr = use_reduce_lr
        self.patience = patience

    def _get_callbacks(self) -> list:
        """
        Crée la liste des callbacks pour l'entraînement.

        Returns:
            Liste des callbacks
        """
        callbacks = []

        if self.use_early_stopping:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=self.verbose
            )
            callbacks.append(early_stop)

        if self.use_reduce_lr:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2,
                min_lr=1e-7,
                verbose=self.verbose
            )
            callbacks.append(reduce_lr)

        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> History:
        """
        Entraîne le modèle MNIST.

        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_val: Données de validation (optionnel)
            y_val: Labels de validation (optionnel)
            **kwargs: Arguments supplémentaires pour fit()

        Returns:
            Historique d'entraînement
        """
        # Préparer les données de validation
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Obtenir les callbacks
        callbacks = self._get_callbacks()

        # Entraînement
        print("\n" + "=" * 70)
        print("Début de l'entraînement")
        print("=" * 70)

        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_data=validation_data,
            callbacks=callbacks,
            **kwargs
        )

        print("\n" + "=" * 70)
        print("Entraînement terminé")
        print("=" * 70)

        return self.history

    def __repr__(self) -> str:
        """Représentation string du trainer."""
        return (
            f"MnistTrainer("
            f"batch_size={self.batch_size}, "
            f"epochs={self.epochs}, "
            f"early_stopping={self.use_early_stopping}, "
            f"reduce_lr={self.use_reduce_lr})"
        )
