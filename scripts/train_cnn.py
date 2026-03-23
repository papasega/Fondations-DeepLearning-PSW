#!/usr/bin/env python3
"""
Script d'entraînement pour le CNN MNIST avec visualisation des embeddings.

Utilise l'architecture modulaire avec ABC et Protocol.
"""

import sys
from pathlib import Path
import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_mnist import CNNMnist
from src.trainers.mnist_trainer import MnistTrainer
from src.visualizers.training_visualizer import TrainingVisualizer
from src.visualizers.embeddings_visualizer import EmbeddingsVisualizer
from src.utils.data_loader import MnistDataLoader


def main() -> None:
    """Fonction principale."""
    print("=" * 70)
    print("Entraînement CNN MNIST avec Visualisation des Embeddings")
    print("=" * 70)

    # Configuration
    CONV_FILTERS = [32, 64, 128]
    EMBEDDING_DIM = 128
    DROPOUT_RATE = 0.5
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 10
    N_SAMPLES_VIZ = 5000

    # 1. Chargement des données
    print("\n1. Chargement des données...")
    data_loader = MnistDataLoader(
        normalize=True,
        flatten=False,  # Pour CNN
        one_hot=True
    )

    (X_train, y_train), (X_test, y_test) = data_loader.load_and_preprocess(
        for_cnn=True
    )

    # Garder aussi les labels non one-hot pour la visualisation
    _, (_, y_test_original) = data_loader.load_data()

    # 2. Création du modèle
    print("\n2. Création du modèle CNN...")
    model = CNNMnist(
        input_shape=(28, 28, 1),
        num_classes=10,
        conv_filters=CONV_FILTERS,
        embedding_dim=EMBEDDING_DIM,
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

    # 6. Visualisation de l'historique d'entraînement
    print("\n6. Visualisation de l'historique d'entraînement...")
    train_visualizer = TrainingVisualizer(output_dir="outputs/cnn")

    fig = train_visualizer.plot(history)
    train_visualizer.save("training_history.png", fig)
    train_visualizer.close(fig)

    # 7. Extraction des embeddings
    print("\n7. Extraction des embeddings...")
    embedding_model = model.get_embedding_model()

    X_sample = X_test[:N_SAMPLES_VIZ]
    y_sample = y_test_original[:N_SAMPLES_VIZ]

    embeddings = embedding_model.predict(X_sample, verbose=0)
    print(f"Forme des embeddings: {embeddings.shape}")

    # 8. Visualisation des embeddings avec t-SNE
    print("\n8. Visualisation des embeddings avec t-SNE...")
    emb_visualizer = EmbeddingsVisualizer(
        output_dir="outputs/cnn",
        method="tsne",
        n_samples=N_SAMPLES_VIZ
    )

    fig_tsne = emb_visualizer.plot(
        (embeddings, y_sample),
        reduce=True,
        method='tsne'
    )
    emb_visualizer.save("embeddings_tsne.png", fig_tsne)
    emb_visualizer.close(fig_tsne)

    # 9. Visualisation des embeddings avec UMAP (si disponible)
    print("\n9. Visualisation des embeddings avec UMAP...")
    try:
        fig_umap = emb_visualizer.plot(
            (embeddings, y_sample),
            reduce=True,
            method='umap'
        )
        emb_visualizer.save("embeddings_umap.png", fig_umap)
        emb_visualizer.close(fig_umap)

        # 10. Comparaison t-SNE vs UMAP
        print("\n10. Comparaison t-SNE vs UMAP...")
        fig_comparison = emb_visualizer.plot_comparison(embeddings, y_sample)
        emb_visualizer.save("embeddings_comparison.png", fig_comparison)
        emb_visualizer.close(fig_comparison)
    except Exception as e:
        print(f"UMAP non disponible ou erreur: {e}")
        print("Pour installer UMAP: pip install umap-learn")

    # 11. Sauvegarde du modèle
    print("\n11. Sauvegarde du modèle...")
    model.save("outputs/cnn/cnn_mnist_model.h5")

    print("\n" + "=" * 70)
    print("Entraînement et visualisation terminés avec succès!")
    print("=" * 70)
    print("\nFichiers générés dans outputs/cnn/:")
    print("  - training_history.png")
    print("  - embeddings_tsne.png")
    print("  - embeddings_umap.png (si UMAP disponible)")
    print("  - embeddings_comparison.png (si UMAP disponible)")
    print("  - cnn_mnist_model.h5")


if __name__ == "__main__":
    main()
