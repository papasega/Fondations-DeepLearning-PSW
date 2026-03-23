"""
Visualiseur pour les embeddings avec t-SNE et UMAP.
"""

from typing import Optional, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.base.visualizer_base import VisualizerBase

# Import optionnel de UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class EmbeddingsVisualizer(VisualizerBase):
    """
    Visualiseur pour les embeddings haute dimension.

    Utilise t-SNE et UMAP pour réduire la dimensionalité
    et visualiser les embeddings en 2D.

    Attributes:
        method: Méthode de réduction ('tsne' ou 'umap')
        n_samples: Nombre d'échantillons à visualiser
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        dpi: int = 300,
        figsize: tuple = (14, 10),
        method: str = "tsne",
        n_samples: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialise le visualiseur d'embeddings.

        Args:
            output_dir: Répertoire de sortie
            dpi: Résolution des images
            figsize: Taille des figures
            method: Méthode de réduction ('tsne' ou 'umap')
            n_samples: Nombre max d'échantillons (None = tous)
            **kwargs: Arguments supplémentaires
        """
        super().__init__(output_dir, dpi, figsize, **kwargs)
        self.method = method
        self.n_samples = n_samples

        if method == 'umap' and not UMAP_AVAILABLE:
            print("UMAP non disponible. Utilisation de t-SNE.")
            self.method = 'tsne'

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: Optional[str] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Réduit les dimensions des embeddings.

        Args:
            embeddings: Embeddings haute dimension
            method: Méthode ('tsne' ou 'umap', None = self.method)
            **kwargs: Paramètres pour le reducer

        Returns:
            Embeddings 2D
        """
        if method is None:
            method = self.method

        # Limiter le nombre d'échantillons si nécessaire
        if self.n_samples is not None and len(embeddings) > self.n_samples:
            embeddings = embeddings[:self.n_samples]

        print(f"\nRéduction de dimension avec {method.upper()}...")
        print(f"Forme des embeddings: {embeddings.shape}")

        if method == 'tsne':
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=kwargs.get('perplexity', 30),
                n_iter=kwargs.get('n_iter', 1000),
                verbose=kwargs.get('verbose', 1)
            )
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1)
            )
        else:
            raise ValueError(f"Méthode '{method}' non supportée")

        embeddings_2d = reducer.fit_transform(embeddings)
        print(f"Réduction terminée. Forme: {embeddings_2d.shape}")

        return embeddings_2d

    def plot(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        reduce: bool = True,
        method: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs: Any
    ) -> plt.Figure:
        """
        Visualise les embeddings.

        Args:
            data: Tuple (embeddings, labels)
            reduce: Si True, applique la réduction de dimension
            method: Méthode de réduction (None = self.method)
            title: Titre du graphique
            **kwargs: Arguments pour la réduction

        Returns:
            Figure matplotlib
        """
        embeddings, labels = data

        # Limiter le nombre d'échantillons
        if self.n_samples is not None and len(embeddings) > self.n_samples:
            embeddings = embeddings[:self.n_samples]
            labels = labels[:self.n_samples]

        # Réduire les dimensions si nécessaire
        if reduce:
            embeddings_2d = self.reduce_dimensions(embeddings, method, **kwargs)
        else:
            embeddings_2d = embeddings

        # Créer la figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Palette de couleurs
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        # Tracer chaque classe
        for digit in range(10):
            mask = labels == digit
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[digit]],
                label=str(digit),
                alpha=0.6,
                s=20
            )

        # Configuration du graphique
        method_name = method if method else self.method
        if title is None:
            title = f'Visualisation des Embeddings avec {method_name.upper()}'

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(f'{method_name.upper()} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method_name.upper()} Dimension 2', fontsize=12)
        ax.legend(title='Chiffre', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.current_fig = fig

        return fig

    def plot_comparison(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        **kwargs: Any
    ) -> plt.Figure:
        """
        Compare t-SNE et UMAP côte à côte.

        Args:
            embeddings: Embeddings haute dimension
            labels: Labels des échantillons
            **kwargs: Arguments pour la réduction

        Returns:
            Figure matplotlib
        """
        if not UMAP_AVAILABLE:
            print("UMAP non disponible. Comparaison impossible.")
            return self.plot((embeddings, labels), method='tsne', **kwargs)

        # Limiter les échantillons
        if self.n_samples is not None and len(embeddings) > self.n_samples:
            embeddings = embeddings[:self.n_samples]
            labels = labels[:self.n_samples]

        # Réductions
        embeddings_tsne = self.reduce_dimensions(embeddings, 'tsne', **kwargs)
        embeddings_umap = self.reduce_dimensions(embeddings, 'umap', **kwargs)

        # Figure
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        # t-SNE
        for digit in range(10):
            mask = labels == digit
            axes[0].scatter(
                embeddings_tsne[mask, 0],
                embeddings_tsne[mask, 1],
                c=[colors[digit]],
                label=str(digit),
                alpha=0.6,
                s=20
            )
        axes[0].set_title('t-SNE', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Dimension 1')
        axes[0].set_ylabel('Dimension 2')
        axes[0].legend(title='Chiffre')
        axes[0].grid(True, alpha=0.3)

        # UMAP
        for digit in range(10):
            mask = labels == digit
            axes[1].scatter(
                embeddings_umap[mask, 0],
                embeddings_umap[mask, 1],
                c=[colors[digit]],
                label=str(digit),
                alpha=0.6,
                s=20
            )
        axes[1].set_title('UMAP', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Dimension 1')
        axes[1].set_ylabel('Dimension 2')
        axes[1].legend(title='Chiffre')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            'Comparaison des méthodes de réduction de dimension',
            fontsize=16,
            fontweight='bold'
        )
        plt.tight_layout()
        self.current_fig = fig

        return fig
