"""
Visualiseur pour l'historique d'entraînement.
"""

from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History

from src.base.visualizer_base import VisualizerBase


class TrainingVisualizer(VisualizerBase):
    """
    Visualiseur pour l'historique d'entraînement.

    Crée des graphiques montrant l'évolution de la loss
    et de l'accuracy pendant l'entraînement.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        dpi: int = 300,
        figsize: tuple = (14, 5),
        **kwargs: Any
    ) -> None:
        """
        Initialise le visualiseur d'entraînement.

        Args:
            output_dir: Répertoire de sortie
            dpi: Résolution des images
            figsize: Taille des figures
            **kwargs: Arguments supplémentaires
        """
        super().__init__(output_dir, dpi, figsize, **kwargs)

    def plot(
        self,
        history: History,
        metrics: Optional[List[str]] = None,
        **kwargs: Any
    ) -> plt.Figure:
        """
        Crée un graphique de l'historique d'entraînement.

        Args:
            history: Historique d'entraînement Keras
            metrics: Liste des métriques à afficher (None = toutes)
            **kwargs: Arguments supplémentaires

        Returns:
            Figure matplotlib
        """
        if metrics is None:
            # Détecter automatiquement les métriques
            metrics = ['loss']
            if 'accuracy' in history.history:
                metrics.append('accuracy')

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=self.figsize)

        # S'assurer que axes est toujours une liste
        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Tracer la métrique d'entraînement
            if metric in history.history:
                ax.plot(
                    history.history[metric],
                    label=f'Train {metric.capitalize()}',
                    linewidth=2
                )

            # Tracer la métrique de validation si disponible
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                ax.plot(
                    history.history[val_metric],
                    label=f'Validation {metric.capitalize()}',
                    linewidth=2
                )

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f'Evolution de {metric.capitalize()}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.current_fig = fig

        return fig

    def plot_metric_comparison(
        self,
        histories: Dict[str, History],
        metric: str = 'accuracy',
        **kwargs: Any
    ) -> plt.Figure:
        """
        Compare plusieurs historiques d'entraînement.

        Args:
            histories: Dictionnaire {nom_modèle: history}
            metric: Métrique à comparer
            **kwargs: Arguments supplémentaires

        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for name, history in histories.items():
            if metric in history.history:
                ax.plot(
                    history.history[metric],
                    label=name,
                    linewidth=2
                )

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'Comparaison: {metric.capitalize()}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.current_fig = fig

        return fig
