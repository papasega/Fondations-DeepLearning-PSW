"""
Dense Neural Network pour la classification MNIST.
"""

from typing import Tuple
from tensorflow.keras.models import Sequential, Model as KerasModel
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

from src.base.model_base import ModelBase


class DNNMnist(ModelBase):
    """
    Dense Neural Network pour MNIST.

    Architecture:
    - Dense (hidden_neurons) + ReLU
    - Dropout (optionnel)
    - Dense (num_classes) + Softmax

    Attributes:
        hidden_neurons: Nombre de neurones dans la couche cachée
        dropout_rate: Taux de dropout (0 pour désactiver)
        learning_rate: Learning rate pour l'optimiseur
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (784,),
        num_classes: int = 10,
        hidden_neurons: int = 600,
        dropout_rate: float = 0.0,
        learning_rate: float = 0.001,
        **kwargs
    ) -> None:
        """
        Initialise le modèle DNN MNIST.

        Args:
            input_shape: Forme de l'entrée (par défaut 784 pour images aplaties)
            num_classes: Nombre de classes (10 pour MNIST)
            hidden_neurons: Nombre de neurones dans la couche cachée
            dropout_rate: Taux de dropout
            learning_rate: Learning rate
            **kwargs: Arguments supplémentaires
        """
        super().__init__(input_shape, num_classes, **kwargs)
        self.hidden_neurons = hidden_neurons
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def build(self) -> KerasModel:
        """
        Construit l'architecture du DNN.

        Returns:
            Modèle Keras compilé
        """
        model = Sequential(name="DNN_MNIST")

        # Couche d'entrée + première couche cachée
        model.add(Dense(
            self.hidden_neurons,
            input_dim=self.input_shape[0],
            name='hidden_layer'
        ))
        model.add(Activation('relu', name='relu_activation'))

        # Dropout optionnel
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate, name='dropout'))

        # Couche de sortie
        model.add(Dense(self.num_classes, name='output_layer'))
        model.add(Activation('softmax', name='softmax_activation'))

        return model

    def compile_model(self, model: KerasModel) -> None:
        """
        Compile le modèle avec l'optimiseur Adam et la categorical crossentropy.

        Args:
            model: Le modèle Keras à compiler
        """
        optimizer = Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def __repr__(self) -> str:
        """Représentation string du modèle."""
        return (
            f"DNNMnist("
            f"hidden_neurons={self.hidden_neurons}, "
            f"dropout_rate={self.dropout_rate}, "
            f"learning_rate={self.learning_rate})"
        )
