# Fondations Deep Learning PSW 🧠

🇬🇧 [Read in English / Lire en anglais](docs/README_en.md)

![Fondations Deep Learning](assets/images/dls_psw.jpg)

> **Matériel Pédagogique** : Ce dépôt contient les supports de cours, travaux pratiques et solutions pour l'apprentissage du Deep Learning.
> 
> **Note :** La version actuelle utilise **TensorFlow et Keras**. Une version **PyTorch** sera bientôt disponible !

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Collection de modèles et d'outils de Deep Learning avec une architecture modulaire basée sur les **Abstract Base Classes (ABC)** et **Protocol** de Python.

**Auteur:** [Papa Séga WADE](https://papasegawade.com/)

---

## 📋 Table des matières

- [À propos](#à-propos)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Modèles disponibles](#modèles-disponibles)
- [Documentation](#documentation)
- [Contribution](#contribution)

---

## 🎯 À propos

Ce projet fournit une implémentation propre et modulaire de modèles de deep learning pour la classification d'images MNIST. Il utilise les meilleures pratiques de programmation Python avec:

- **ABC (Abstract Base Classes)** pour définir des interfaces claires
- **Protocol** pour le duck typing structurel
- **Type hints** complets pour la sécurité du typage
- **Architecture modulaire** facilement extensible
- **Séparation des préoccupations** (modèles, trainers, visualizers)

### Fonctionnalités principales

✅ Modèles DNN et CNN pour MNIST
✅ Extraction et visualisation des embeddings (t-SNE, UMAP)
✅ Visualisation de l'historique d'entraînement
✅ Scripts d'entraînement prêts à l'emploi
✅ Notebooks Jupyter interactifs
✅ Architecture extensible pour nouveaux modèles

---

## 🏗️ Architecture

Le projet suit une architecture en couches avec des classes de base abstraites:

```
┌─────────────────────────────────────┐
│         Application Layer           │
│  (Scripts & Notebooks)              │
├─────────────────────────────────────┤
│      Visualizers Layer              │
│  (Training, Embeddings)             │
├─────────────────────────────────────┤
│        Trainers Layer               │
│  (Training logic)                   │
├─────────────────────────────────────┤
│         Models Layer                │
│  (DNN, CNN, ...)                    │
├─────────────────────────────────────┤
│        Base Layer (ABC)             │
│  (ModelBase, TrainerBase, ...)      │
├─────────────────────────────────────┤
│         Utils Layer                 │
│  (Data loading, preprocessing)      │
└─────────────────────────────────────┘
```

### Hiérarchie des classes

```python
# Modèles
ModelProtocol (Protocol)
    ↑
ModelBase (ABC)
    ↑
    ├── DNNMnist
    └── CNNMnist

# Trainers
TrainerProtocol (Protocol)
    ↑
TrainerBase (ABC)
    ↑
    └── MnistTrainer

# Visualizers
VisualizerProtocol (Protocol)
    ↑
VisualizerBase (ABC)
    ↑
    ├── TrainingVisualizer
    └── EmbeddingsVisualizer
```

---

## 📦 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip

### Installation des dépendances

```bash
# 1. Cloner le repository
git clone https://github.com/papasega/Fondations-DeepLearning-PSW.git
cd Fondations-DeepLearning-PSW

# 2. Créer et activer un environnement virtuel (Recommandé)
python3 -m venv env_dl
source env_dl/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

- `tensorflow>=2.8.0` - Framework de deep learning
- `numpy>=1.21.0` - Calculs numériques
- `matplotlib>=3.5.0` - Visualisation
- `scikit-learn>=1.0.0` - t-SNE
- `umap-learn>=0.5.0` - UMAP (optionnel mais recommandé)
- `jupyter>=1.0.0` - Notebooks interactifs

---

## 🚀 Utilisation

### Option 1: Scripts Python

#### Entraîner un DNN

```bash
python scripts/train_dnn.py
```

#### Entraîner un CNN avec visualisation des embeddings

```bash
python scripts/train_cnn.py
```

Les résultats seront sauvegardés dans `outputs/dnn/` ou `outputs/cnn/`.

### Option 2: Notebooks Jupyter

```bash
jupyter notebook notebooks/
```

Ouvrez et exécutez:

- `DNN_MNIST_psw.ipynb` - DNN pour MNIST
- `CNN_MNIST_Embeddings.ipynb` - CNN avec visualisation des embeddings
- `psw-Deblurring_PnP_DnCNN_FB.ipynb` - Deblurring avec DnCNN

### Option 3: Utiliser l'API Python

```python
from src.models.cnn_mnist import CNNMnist
from src.trainers.mnist_trainer import MnistTrainer
from src.utils.data_loader import MnistDataLoader

# Charger les données
loader = MnistDataLoader(normalize=True, one_hot=True)
(X_train, y_train), (X_test, y_test) = loader.load_and_preprocess(for_cnn=True)

# Créer le modèle
model = CNNMnist(
    input_shape=(28, 28, 1),
    num_classes=10,
    embedding_dim=128
)

# Entraîner
trainer = MnistTrainer(model.get_model(), epochs=10)
history = trainer.train(X_train, y_train, validation_split=0.1)

# Évaluer
results = trainer.evaluate(X_test, y_test)
print(results)
```

---

## 📁 Structure du projet (Pédagogique)

```
Fondations-DeepLearning-PSW/
│
├── Cours/                           # 📖 Notes & Slides du cours
│   ├── Module_00_Intro_Python/      # Python, NumPy, Matplotlib
│   ├── Module_01_DNN_MNIST/         # Réseau Dense (DNN) - Classification
│   ├── Module_02_CNN_MNIST/         # CNN + Visualisation des Embeddings
│   ├── Module_03_PyTorch/           # Crash Course PyTorch
│   └── Module_04_Computer_Vision/  # CNN sur données réelles (GTSRB ~98%)
│
├── Travaux_Pratiques/               # 📝 Notebooks à trous (Étudiants)
│   ├── Module_00_Intro_Python/
│   ├── Module_01_DNN_MNIST/
│   ├── Module_02_CNN_MNIST/
│   └── Module_03_PyTorch/
│
├── Solutions/                       # ✅ Notebooks corrigés
│   ├── Module_00_Intro_Python/
│   ├── Module_01_DNN_MNIST/
│   ├── Module_02_CNN_MNIST/
│   ├── Module_03_PyTorch/
│   ├── DNN_MNIST_psw.ipynb          # Référence Keras DNN
│   └── CNN_MNIST_Embeddings.ipynb   # Référence Keras CNN
│
├── core_framework/                  # 🔧 Classes de base réutilisables (ABC)
├── scripts/                         # Scripts d'entraînement
├── docs/                            # Documentation bilingue
├── assets/                          # Images & ressources
├── requirements.txt
└── README.md
```

---

## 🤖 Modèles disponibles

### 1. DNN (Dense Neural Network)

Réseau de neurones dense simple pour MNIST.

**Architecture:**

- Dense (600) + ReLU
- Dropout (optionnel)
- Dense (10) + Softmax

**Performance:** ~98% accuracy

**Utilisation:**

```python
from src.models.dnn_mnist import DNNMnist

model = DNNMnist(
    input_shape=(784,),
    hidden_neurons=600,
    dropout_rate=0.0
)
```

### 2. CNN (Convolutional Neural Network)

Réseau de neurones convolutif pour MNIST avec extraction d'embeddings.

**Architecture:**

- Conv2D (32) + MaxPooling
- Conv2D (64) + MaxPooling
- Conv2D (128)
- Dense (128) - Embeddings
- Dropout (0.5)
- Dense (10) + Softmax

**Performance:** ~99% accuracy

**Utilisation:**

```python
from src.models.cnn_mnist import CNNMnist

model = CNNMnist(
    input_shape=(28, 28, 1),
    conv_filters=[32, 64, 128],
    embedding_dim=128
)

# Extraire les embeddings
embedding_model = model.get_embedding_model()
embeddings = embedding_model.predict(X_test)
```

---

## 📊 Visualisations

### Historique d'entraînement

```python
from src.visualizers.training_visualizer import TrainingVisualizer

visualizer = TrainingVisualizer(output_dir="outputs")
fig = visualizer.plot(history)
visualizer.save("training_history.png", fig)
```

### Embeddings (t-SNE / UMAP)

```python
from src.visualizers.embeddings_visualizer import EmbeddingsVisualizer

visualizer = EmbeddingsVisualizer(
    output_dir="outputs",
    method="tsne",
    n_samples=5000
)

# Visualiser avec t-SNE
fig = visualizer.plot((embeddings, labels), method='tsne')
visualizer.save("embeddings_tsne.png", fig)

# Comparer t-SNE et UMAP
fig = visualizer.plot_comparison(embeddings, labels)
visualizer.save("comparison.png", fig)
```

---

## 📚 Documentation

Pour plus de détails:

- [Documentation CNN MNIST](docs/CNN_MNIST_README.md)
- [Notebooks interactifs](notebooks/)
- [Code source documenté](src/)

---

## 🔧 Extensibilité

### Ajouter un nouveau modèle

1. Créer une classe héritant de `ModelBase`:

```python
from src.base.model_base import ModelBase
from tensorflow.keras.models import Sequential

class MyModel(ModelBase):
    def build(self):
        model = Sequential()
        # Définir l'architecture
        return model

    def compile_model(self, model):
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
```

2. Utiliser avec les trainers et visualizers existants!

### Ajouter un nouveau visualizer

```python
from src.base.visualizer_base import VisualizerBase

class MyVisualizer(VisualizerBase):
    def plot(self, data, **kwargs):
        fig, ax = plt.subplots(figsize=self.figsize)
        # Créer votre visualisation
        self.current_fig = fig
        return fig
```

---

## 🧪 Tests

```bash
# Installer les dépendances de test
pip install pytest pytest-cov

# Lancer les tests
pytest tests/ -v

# Avec couverture
pytest tests/ --cov=src --cov-report=html
```

---

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à:

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📝 Licence

Ce projet est sous licence MIT. Vous êtes libre de l'utiliser, le modifier et le distribuer.

---

## 👨‍🎓 Auteur

**Papa Séga WADE**

- Portfolio: [https://papasegawade.com](https://papasegawade.com/)
- GitHub: [@papasega](https://github.com/papasega)

---

## 🙏 Remerciements

- **MNIST Dataset**: Yann LeCun et al.
- **TensorFlow/Keras**: Google Brain Team

---

## 📈 Roadmap

- [ ] Ajouter plus de modèles (ResNet, VGG, etc.)
- [ ] Support pour d'autres datasets (Fashion-MNIST, CIFAR-10)
- [ ] Data augmentation
- [ ] Transfer learning
- [ ] API REST
- [ ] Interface web
- [ ] Tests unitaires complets
- [ ] Documentation complète avec Sphinx

---

**⭐ N'oubliez pas de donner une étoile si vous trouvez ce projet utile!**
