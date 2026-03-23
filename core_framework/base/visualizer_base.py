"""
Classes de base abstraites pour la visualisation des résultats.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, Any, Dict
from pathlib import Path
import matplotlib.pyplot as plt


class VisualizerProtocol(Protocol):
    """
    Protocol définissant l'interface qu'un visualizer doit implémenter.
    """

    def plot(self, data: Any, **kwargs: Any) -> None:
        """Crée une visualisation."""
        ...

    def save(self, filepath: str) -> None:
        """Sauvegarde la visualisation."""
        ...


class VisualizerBase(ABC):
    """
    Classe de base abstraite pour la visualisation.

    Définit l'interface commune pour tous les visualizers
    et fournit des méthodes utilitaires pour la sauvegarde.

    Attributes:
        output_dir: Répertoire de sortie pour les visualisations
        dpi: Résolution des images sauvegardées
        figsize: Taille des figures par défaut
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        dpi: int = 300,
        figsize: tuple = (12, 8),
        **kwargs: Any
    ) -> None:
        """
        Initialise le visualizer.

        Args:
            output_dir: Répertoire de sortie
            dpi: Résolution des images
            figsize: Taille par défaut des figures
            **kwargs: Arguments supplémentaires
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        self._config = kwargs
        self.current_fig: Optional[plt.Figure] = None

    @abstractmethod
    def plot(self, data: Any, **kwargs: Any) -> plt.Figure:
        """
        Crée une visualisation à partir des données.

        Args:
            data: Données à visualiser
            **kwargs: Arguments supplémentaires pour la visualisation

        Returns:
            Figure matplotlib
        """
        pass

    def save(
        self,
        filepath: str,
        fig: Optional[plt.Figure] = None,
        **kwargs: Any
    ) -> None:
        """
        Sauvegarde la visualisation.

        Args:
            filepath: Chemin du fichier de sortie
            fig: Figure à sauvegarder (utilise current_fig si None)
            **kwargs: Arguments supplémentaires pour savefig
        """
        if fig is None:
            fig = self.current_fig

        if fig is None:
            raise ValueError("Aucune figure à sauvegarder")

        # Construire le chemin complet
        full_path = self.output_dir / filepath

        # Sauvegarder
        save_kwargs = {
            "dpi": self.dpi,
            "bbox_inches": "tight",
            **kwargs
        }
        fig.savefig(full_path, **save_kwargs)
        print(f"Figure sauvegardée: {full_path}")

    def show(self, fig: Optional[plt.Figure] = None) -> None:
        """
        Affiche la figure.

        Args:
            fig: Figure à afficher (utilise current_fig si None)
        """
        if fig is None:
            fig = self.current_fig

        if fig is not None:
            plt.show()
        else:
            raise ValueError("Aucune figure à afficher")

    def close(self, fig: Optional[plt.Figure] = None) -> None:
        """
        Ferme la figure.

        Args:
            fig: Figure à fermer (utilise current_fig si None)
        """
        if fig is None:
            fig = self.current_fig

        if fig is not None:
            plt.close(fig)

    def plot_and_save(
        self,
        data: Any,
        filename: str,
        show: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Crée, sauvegarde et optionnellement affiche une visualisation.

        Args:
            data: Données à visualiser
            filename: Nom du fichier de sortie
            show: Si True, affiche la figure
            **kwargs: Arguments pour plot()
        """
        fig = self.plot(data, **kwargs)
        self.save(filename, fig)

        if show:
            self.show(fig)
        else:
            self.close(fig)

    @property
    def config(self) -> Dict[str, Any]:
        """Retourne la configuration du visualizer."""
        return {
            "output_dir": str(self.output_dir),
            "dpi": self.dpi,
            "figsize": self.figsize,
            **self._config
        }

    def __repr__(self) -> str:
        """Représentation string du visualizer."""
        return (
            f"{self.__class__.__name__}("
            f"output_dir={self.output_dir})"
        )
