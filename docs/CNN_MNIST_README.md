# CNN MNIST avec Visualisation des Embeddings

🇬🇧 [Read in English / Lire en anglais](CNN_MNIST_README_en.md)

Ce projet implémente un réseau de neurones convolutif (CNN) pour classifier les chiffres manuscrits du dataset MNIST et visualise les embeddings appris par le réseau.

## 📋 Description

Le projet contient deux implémentations :
1. **Notebook Jupyter** (`CNN_MNIST_Embeddings.ipynb`) : pour une exploration interactive
2. **Script Python** (`cnn_mnist_embeddings.py`) : pour une exécution directe

## 🎯 Objectifs

- Construire un CNN performant pour la classification MNIST
- Extraire les représentations (embeddings) apprises par le réseau
- Visualiser les embeddings en 2D avec t-SNE et UMAP
- Comprendre comment le réseau organise les données dans l'espace des features

## 🏗️ Architecture du CNN

```
Input (28x28x1)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3) + ReLU
    ↓
Flatten
    ↓
Dense (128) + ReLU  ← Couche d'embeddings
    ↓
Dropout (0.5)
    ↓
Dense (10) + Softmax
```

## 📦 Installation

### Installer les dépendances

```bash
pip install -r requirements_cnn.txt
```

Ou manuellement :

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn umap-learn jupyter notebook
```

## 🚀 Utilisation

### Option 1 : Jupyter Notebook (Recommandé)

```bash
jupyter notebook CNN_MNIST_Embeddings.ipynb
```

Exécutez les cellules une par une pour une exploration interactive.

### Option 2 : Script Python

```bash
python cnn_mnist_embeddings.py
```

Le script exécutera automatiquement toutes les étapes et générera les visualisations.

## 📊 Résultats

Le modèle génère plusieurs visualisations :

1. **training_history.png** : Courbes d'apprentissage (loss et accuracy)
2. **embeddings_tsne.png** : Visualisation t-SNE des embeddings
3. **embeddings_umap.png** : Visualisation UMAP des embeddings
4. **embeddings_comparison.png** : Comparaison t-SNE vs UMAP

### Performance attendue

- **Test Accuracy** : ~98-99%
- **Temps d'entraînement** : ~5-10 minutes (10 epochs)
- **Temps t-SNE** : ~2-5 minutes (5000 échantillons)
- **Temps UMAP** : ~1-2 minutes (5000 échantillons)

## 🔍 Analyse des Embeddings

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Excellent pour la visualisation locale
- Préserve les structures locales
- Plus lent que UMAP

### UMAP (Uniform Manifold Approximation and Projection)
- Plus rapide que t-SNE
- Préserve mieux la structure globale
- Recommandé pour les grands datasets

## 📈 Interprétation

Les visualisations montrent :
- **Clusters bien séparés** : Le réseau a appris à distinguer les différents chiffres
- **Clusters proches** : Les chiffres similaires (ex: 4 et 9, 3 et 8) sont proches dans l'espace des embeddings
- **Points isolés** : Potentiellement des erreurs de classification ou des exemples ambigus

## 🎓 Concepts clés

### Embeddings
Les embeddings sont des représentations vectorielles de dimension réduite (ici 128) qui capturent les caractéristiques essentielles des images. Le CNN apprend ces représentations de manière à :
- Maximiser la séparation entre les classes
- Minimiser la distance intra-classe
- Faciliter la classification finale

### Réduction de dimension
t-SNE et UMAP réduisent les embeddings de 128 dimensions à 2 dimensions pour la visualisation, tout en préservant autant que possible les relations de similarité entre les points.

## 🛠️ Personnalisation

Vous pouvez modifier :

### Hyperparamètres d'entraînement
```python
batch_size = 128  # Taille des batchs
epochs = 10       # Nombre d'epochs
```

### Architecture du CNN
- Ajouter/retirer des couches convolutives
- Modifier la taille des filtres
- Changer la dimension des embeddings

### Visualisation
```python
n_samples = 5000  # Nombre d'échantillons à visualiser
perplexity = 30   # Paramètre t-SNE
n_neighbors = 15  # Paramètre UMAP
```

## 📚 Extensions possibles

1. **Data Augmentation** : Rotation, translation, zoom
2. **Batch Normalization** : Améliorer la stabilité de l'entraînement
3. **Autres datasets** : Fashion-MNIST, CIFAR-10
4. **Analyse 3D** : Visualisation en 3 dimensions
5. **Animation** : Evolution des embeddings pendant l'entraînement
6. **Transfer Learning** : Utiliser les embeddings pour d'autres tâches

## 👨‍🎓 Auteur

**Papa Séga WADE**
- Portfolio : [PSW Portfolio](https://papasegawade.com)

## 📄 Licence

Ce projet est à but éducatif pour l'apprentissage du Deep Learning.

## 🙏 Remerciements

- Dataset MNIST : Yann LeCun et al.
- TensorFlow/Keras : Google Brain Team
- t-SNE : Laurens van der Maaten
- UMAP : Leland McInnes et al.
