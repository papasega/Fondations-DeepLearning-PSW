"""
Convolutional Neural Network pour la classification MNIST.
"""

from typing import Tuple, List
from tensorflow.keras.models import Sequential, Model as KerasModel
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
)
from tensorflow.keras.optimizers import Adam

from src.base.model_base import ModelBase


class CNNMnist(ModelBase):
    """
    Convolutional Neural Network pour MNIST.

    Architecture:
    - Conv2D (32 filters) + ReLU + MaxPooling
    - Conv2D (64 filters) + ReLU + MaxPooling
    - Conv2D (128 filters) + ReLU
    - Flatten
    - Dense (embedding_dim) + ReLU (Embeddings layer)
    - Dropout
    - Dense (num_classes) + Softmax

    Attributes:
        conv_filters: Liste du nombre de filtres pour chaque couche conv
        embedding_dim: Dimension de la couche d'embeddings
        dropout_rate: Taux de dropout
        learning_rate: Learning rate pour l'optimiseur
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (28, 28, 1),
        num_classes: int = 10,
        conv_filters: List[int] = None,
        embedding_dim: int = 128,
        dropout_rate: float = 0.5,
        learning_rate: float = 0.001,
        **kwargs
    ) -> None:
        """
        Initialise le modèle CNN MNIST.

        Args:
            input_shape: Forme de l'entrée (height, width, channels)
            num_classes: Nombre de classes (10 pour MNIST)
            conv_filters: Liste du nombre de filtres par couche conv
            embedding_dim: Dimension de la couche d'embeddings
            dropout_rate: Taux de dropout
            learning_rate: Learning rate
            **kwargs: Arguments supplémentaires
        """
        super().__init__(input_shape, num_classes, **kwargs)

        if conv_filters is None:
            conv_filters = [32, 64, 128]

        self.conv_filters = conv_filters
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def build(self) -> KerasModel:
        """
        Construit l'architecture du CNN.

        Returns:
            Modèle Keras compilé
        """
        model = Sequential(name="CNN_MNIST")

        # Première couche convolutive
        model.add(Conv2D(
            self.conv_filters[0],
            kernel_size=(3, 3),
            activation='relu',
            input_shape=self.input_shape,
            name='conv1'
        ))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

        # Deuxième couche convolutive
        if len(self.conv_filters) > 1:
            model.add(Conv2D(
                self.conv_filters[1],
                kernel_size=(3, 3),
                activation='relu',
                name='conv2'
            ))
            model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

        # Troisième couche convolutive (sans pooling)
        if len(self.conv_filters) > 2:
            model.add(Conv2D(
                self.conv_filters[2],
                kernel_size=(3, 3),
                activation='relu',
                name='conv3'
            ))

        # Aplatissement
        model.add(Flatten(name='flatten'))

        # Couche d'embeddings
        model.add(Dense(
            self.embedding_dim,
            activation='relu',
            name='embeddings'
        ))

        # Dropout
        model.add(Dropout(self.dropout_rate, name='dropout'))

        # Couche de sortie
        model.add(Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        ))

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

    def get_embedding_model(self) -> KerasModel:
        """
        Crée un modèle qui extrait les embeddings.

        Returns:
            Modèle Keras qui retourne les embeddings
        """
        from tensorflow.keras.models import Model

        full_model = self.get_model()
        embedding_layer = full_model.get_layer('embeddings')

        embedding_model = Model(
            inputs=full_model.input,
            outputs=embedding_layer.output,
            name='embedding_extractor'
        )

        return embedding_model

    def __repr__(self) -> str:
        """Représentation string du modèle."""
        return (
            f"CNNMnist("
            f"conv_filters={self.conv_filters}, "
            f"embedding_dim={self.embedding_dim}, "
            f"dropout_rate={self.dropout_rate}, "
            f"learning_rate={self.learning_rate})"
        )
