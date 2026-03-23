# CNN MNIST with Embeddings Visualization

🇬🇧 **English Version** | [🇫🇷 Lire en français](CNN_MNIST_README.md)

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset and visualizes the embeddings learned by the network.

## 📋 Description

The project contains two implementations:
1. **Jupyter Notebook** (`CNN_MNIST_Embeddings.ipynb`): for interactive exploration
2. **Python Script** (`cnn_mnist_embeddings.py`): for direct execution

## 🎯 Objectives

- Build a high-performance CNN for MNIST classification
- Extract representations (embeddings) learned by the network
- Visualize embeddings in 2D with t-SNE and UMAP
- Understand how the network organizes data in the feature space

## 🏗️ CNN Architecture

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
Dense (128) + ReLU  ← Embeddings layer
    ↓
Dropout (0.5)
    ↓
Dense (10) + Softmax
```

## 📦 Installation

### Install dependencies

```bash
pip install -r requirements_cnn.txt
```

Or manually:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn umap-learn jupyter notebook
```

## 🚀 Usage

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook CNN_MNIST_Embeddings.ipynb
```

Execute cells one by one for interactive exploration.

### Option 2: Python Script

```bash
python cnn_mnist_embeddings.py
```

The script will automatically execute all steps and generate visualizations.

## 📊 Results

The model generates several visualizations:

1. **training_history.png**: Learning curves (loss and accuracy)
2. **embeddings_tsne.png**: t-SNE visualization of embeddings
3. **embeddings_umap.png**: UMAP visualization of embeddings
4. **embeddings_comparison.png**: t-SNE vs UMAP comparison

### Expected Performance

- **Test Accuracy**: ~98-99%
- **Training time**: ~5-10 minutes (10 epochs)
- **t-SNE time**: ~2-5 minutes (5000 samples)
- **UMAP time**: ~1-2 minutes (5000 samples)

## 🔍 Embeddings Analysis

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Excellent for local visualization
- Preserves local structures
- Slower than UMAP

### UMAP (Uniform Manifold Approximation and Projection)
- Faster than t-SNE
- Better preserves global structure
- Recommended for large datasets

## 📈 Interpretation

Visualizations show:
- **Well-separated clusters**: The network learned to distinguish different digits
- **Close clusters**: Similar digits (e.g., 4 and 9, 3 and 8) are close in the embeddings space
- **Isolated points**: Potentially classification errors or ambiguous examples

## 🎓 Key Concepts

### Embeddings
Embeddings are lower-dimensional vector representations (here 128) that capture the essential features of images. The CNN learns these representations in order to:
- Maximize separation between classes
- Minimize intra-class distance
- Facilitate final classification

### Dimensionality Reduction
t-SNE and UMAP reduce embeddings from 128 dimensions to 2 dimensions for visualization, while preserving similarity relationships between points as much as possible.

## 🛠️ Customization

You can modify:

### Training Hyperparameters
```python
batch_size = 128  # Batch size
epochs = 10       # Number of epochs
```

### CNN Architecture
- Add/remove convolutional layers
- Modify filter sizes
- Change embeddings dimension

### Visualization
```python
n_samples = 5000  # Number of samples to visualize
perplexity = 30   # t-SNE parameter
n_neighbors = 15  # UMAP parameter
```

## 📚 Possible Extensions

1. **Data Augmentation**: Rotation, translation, zoom
2. **Batch Normalization**: Improve training stability
3. **Other datasets**: Fashion-MNIST, CIFAR-10
4. **3D Analysis**: 3-dimensional visualization
5. **Animation**: Evolution of embeddings during training
6. **Transfer Learning**: Use embeddings for other tasks

## 👨‍🎓 Author

**Papa Séga WADE**
- Portfolio: [PSW Portfolio](https://papasegawade.com)

## 📄 License

This project is for educational purposes for learning Deep Learning.

## 🙏 Acknowledgements

- MNIST Dataset: Yann LeCun et al.
- TensorFlow/Keras: Google Brain Team
- t-SNE: Laurens van der Maaten
- UMAP: Leland McInnes et al.
