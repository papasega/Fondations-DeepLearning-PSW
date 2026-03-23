#!/usr/bin/env python3
"""
Script d'entraînement pour le DNN MNIST.

Utilise l'architecture modulaire avec ABC et Protocol.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dnn_mnist import DNNMnist
from src.trainers.mnist_trainer import MnistTrainer
from src.visualizers.training_visualizer import TrainingVisualizer
from src.utils.data_loader import MnistDataLoader


def main() -> None:
    """Fonction principale."""
    print("=" * 70)
    print("Entraînement DNN MNIST")
    print("=" * 70)

    # Configuration
    HIDDEN_NEURONS = 600
    DROPOUT_RATE = 0.0
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 25

    # 1. Chargement des données
    print("\n1. Chargement des données...")
    data_loader = MnistDataLoader(
        normalize=True,
        flatten=True,  # Pour DNN
        one_hot=True
    )

    (X_train, y_train), (X_test, y_test) = data_loader.load_and_preprocess(
        for_cnn=False
    )

    # 2. Création du modèle
    print("\n2. Création du modèle DNN...")
    model = DNNMnist(
        input_shape=(784,),
        num_classes=10,
        hidden_neurons=HIDDEN_NEURONS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )

    print(f"\nModèle: {model}")
    model.summary()

    # 3. Création du trainer
    print("\n3. Configuration du trainer...")
    trainer = MnistTrainer(
        model=model.get_model(),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        use_early_stopping=False,
        use_reduce_lr=False
    )

    print(f"Trainer: {trainer}")

    # 4. Entraînement
    print("\n4. Entraînement...")
    history = trainer.train(
        X_train, y_train,
        validation_split=0.1
    )

    # 5. Évaluation
    print("\n5. Évaluation sur l'ensemble de test...")
    results = trainer.evaluate(X_test, y_test)

    print("\nRésultats:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # 6. Visualisation
    print("\n6. Visualisation de l'historique...")
    visualizer = TrainingVisualizer(output_dir="outputs/dnn")

    fig = visualizer.plot(history)
    visualizer.save("training_history.png", fig)
    visualizer.close(fig)

    # 7. Sauvegarde du modèle
    print("\n7. Sauvegarde du modèle...")
    model.save("outputs/dnn/dnn_mnist_model.h5")

    print("\n" + "=" * 70)
    print("Entraînement terminé avec succès!")
    print("=" * 70)


if __name__ == "__main__":
    main()
