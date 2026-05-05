# 🧠 Classic CNN Architectures: Research Papers Implementation

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg?style=for-the-badge)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.13%2B-red.svg?style=for-the-badge)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg?style=for-the-badge)]()

**Complete implementations of groundbreaking CNN architectures from seminal research papers**

[📖 Quick Start](#-quick-start) • [📚 Models](#-implemented-architectures) • [📊 Comparison](#-model-comparison) • [🤝 Contributing](#-contributing)

</div>

---

## 📌 Overview

A comprehensive, production-ready collection of implementations of seminal Convolutional Neural Network (CNN) architectures from groundbreaking research papers, recreated using Keras and TensorFlow. This project serves as both an **educational resource** and a **practical toolkit** for understanding and applying these foundational models.

### ✨ Key Features
- ✅ **10+ Classic Architectures** - From LeNet to Xception  
- ✅ **Detailed Jupyter Notebooks** - Step-by-step implementations with explanations  
- ✅ **Pre-trained Models** - Ready-to-use model checkpoints  
- ✅ **Transfer Learning Examples** - Practical applications on custom datasets  
- ✅ **Comprehensive Documentation** - Architecture details and theoretical background  
- ✅ **Production Ready** - Best practices for training and deployment  

---

## 🚀 Quick Start

### ⏱️ 5-Minute Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd "Research Papers Implementation"

# 2. Create virtual environment
python -m venv cnn_env
source cnn_env/bin/activate  # Windows: cnn_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook

# 5. Open any notebook and start learning!
```

### 📋 Prerequisites
- Python 3.8+
- pip or conda
- ~2GB disk space for models
- GPU recommended (CUDA 11.0+) for faster training

---

## 📦 Dependencies

| Package      | Version | Purpose                        |
|:-------------|:--------|:-------------------------------|
| TensorFlow   | 2.13+   | Deep learning framework        |
| Keras        | 2.13+   | High-level neural networks API |
| NumPy        | 1.24+   | Numerical computations         |
| Matplotlib   | 3.7+    | Data visualization             |
| Pandas       | 2.0+    | Data manipulation              |
| Scikit-learn | 1.3+    | ML utilities & metrics         |
| Jupyter      | 1.0+    | Interactive notebooks          |
| Pillow       | 10.0+   | Image processing               |

---

## 📚 Implemented Architectures

### 🏗️ LeNet-5 (1998)

**The Foundational Architecture**

| Property | Value |
|:---------|:------|
| **Paper** | [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) |
| **Notebook** | [LeNetImplementation.ipynb](LeNetImplementation.ipynb) |
| **Primary Task** | Handwritten digit recognition (MNIST) |
| **Model Checkpoint** | [bestLeNet.h5](bestLeNet.h5) |
| **Key Innovations** | Conv layers, pooling, fully-connected layers |
| **Parameters** | ~60K |
| **Accuracy** | 99.2% on MNIST |

**What you'll learn**: Basic CNN architecture, convolution operations, pooling strategies

---

### 🧠 AlexNet (2012)

**The Deep Learning Revolution**

| Property | Value |
|:---------|:------|
| **Paper** | [ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) |
| **Notebook** | [AlexNetImplementation.ipynb](AlexNetImplementation.ipynb) |
| **Primary Task** | ImageNet Large Scale Visual Recognition Challenge |
| **Key Innovations** | Deep architecture, ReLU activation, dropout, data augmentation |
| **Parameters** | 62.4M |
| **Accuracy** | 63.3% (ImageNet top-1) |

**Why it matters**: Demonstrated that deep CNNs could achieve breakthrough performance on large-scale vision tasks.

---

### 🏛️ VGGNet (2014)

**Exploring Depth & Uniformity**

| Property | Value |
|:---------|:------|
| **Paper** | [Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556) |
| **Notebooks** | [VGGImplementation.ipynb](VGGImplementation.ipynb) • [VGGtransferLearning.ipynb](VGGtransferLearning.ipynb) |
| **Key Innovations** | 3×3 convolutional filters, stacked architecture, uniform design |
| **Variants** | VGG-16 (16 layers), VGG-19 (19 layers) |
| **Parameters** | 138M-144M |
| **Accuracy** | 71.3-72.4% (ImageNet top-1) |

**Use Cases**: Image classification, feature extraction, transfer learning on custom datasets

---

### 🔬 InceptionV3 (2015)

**Intelligent Multi-Scale Feature Learning**

| Property | Value |
|:---------|:------|
| **Paper** | [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567) |
| **Notebooks** | [Inception_Pytorch_Pretrained.ipynb](Inception_Pytorch_Pretrained.ipynb) • [InceptionV3_Transfer_Learning_Flowers_Corrected.ipynb](InceptionV3_Transfer_Learning_Flowers_Corrected.ipynb) |
| **Key Innovations** | Factorized convolutions, auxiliary classifiers, aggressive regularization |
| **Parameters** | Efficient (~23M) |
| **Comparison** | [Inception_Comparison.md](Inception_Comparison.md) |

**What makes it special**: Multi-scale feature extraction with computational efficiency

---

### ✨ Xception (2017)

**Extreme Inception: Depthwise Separable Convolutions**

| Property | Value |
|:---------|:------|
| **Paper** | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) |
| **Notebooks** | [XceptionPretrained.ipynb](XceptionPretrained.ipynb) • [XceptionTransferLearning.ipynb](XceptionTransferLearning.ipynb) |
| **Key Innovations** | Depthwise separable convolutions, larger capacity with efficiency |
| **Parameters** | 22.9M |
| **Accuracy** | 79.0% (ImageNet top-1) |

**Advantages**: Efficient than standard convolutions, excellent for mobile deployment

---

### 🔄 ResNet (2015)

**Residual Learning: Breaking the Depth Barrier**

| Property | Value |
|:---------|:------|
| **Paper** | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |
| **Notebooks** | [ResnetImplementation.ipynb](ResnetImplementation.ipynb) • [ResNet_Transfer_RockPaperScissors.ipynb](ResNet_Transfer_RockPaperScissors.ipynb) |
| **Key Innovations** | Skip connections, residual blocks, enables ultra-deep networks |
| **Variants** | ResNet-50, ResNet-101, ResNet-152 |
| **Parameters** | 25.5M (ResNet-50) - 44.5M (ResNet-101) |
| **Accuracy** | 76-77.4% (ImageNet top-1) |

**Why ResNet matters**: Solves degradation problem in very deep networks. Enables training of 150+ layer networks!

---

## 📊 Model Comparison

```
Model        │ Year │ Params  │ Depth      │ ImageNet Accuracy
─────────────┼──────┼─────────┼────────────┼──────────────────
LeNet-5      │ 1998 │ ~60K    │ 5          │ 99.2% (MNIST)
AlexNet      │ 2012 │ 62.4M   │ 8          │ 63.3%
VGG-16       │ 2014 │ 138M    │ 16         │ 71.3%
VGG-19       │ 2014 │ 144M    │ 19         │ 72.4%
InceptionV3  │ 2015 │ 23.8M   │ 48         │ 78.8%
ResNet-50    │ 2015 │ 25.5M   │ 50         │ 76.0%
ResNet-101   │ 2015 │ 44.5M   │ 101        │ 77.4%
Xception     │ 2017 │ 22.9M   │ 71         │ 79.0%
```

---

## 🏗️ Project Structure

```
Research Papers Implementation/
│
├── 📄 README.md                       ← You are here!
├── 📄 ARCHITECTURE.md                 ← Technical details
├── 📄 CONTRIBUTING.md                 ← How to contribute
├── 📄 LICENSE                         ← MIT License
├── 📄 pyproject.toml                  ← Project configuration
├── 📄 requirements.txt                ← Python dependencies
│
├── 📓 Jupyter Notebooks (Click to open)
│   ├── LeNetImplementation.ipynb
│   ├── AlexNetImplementation.ipynb
│   ├── VGGImplementation.ipynb
│   ├── VGGtransferLearning.ipynb
│   ├── Inception_Pytorch_Pretrained.ipynb
│   ├── InceptionV3_Transfer_Learning_Flowers_Corrected.ipynb
│   ├── XceptionPretrained.ipynb
│   ├── XceptionTransferLearning.ipynb
│   ├── ResnetImplementation.ipynb
│   └── ResNet_Transfer_RockPaperScissors.ipynb
│
├── 📊 Pre-trained Models
│   └── bestLeNet.h5                   ← LeNet weights
│
└── 📁 .github/
    ├── ISSUE_TEMPLATE.md
    └── PULL_REQUEST_TEMPLATE.md
```

---

## 💡 Key Concepts Covered

- **Convolutional Layers**: Feature extraction at different scales
- **Pooling Layers**: Dimensionality reduction and noise robustness
- **ReLU Activation**: Non-linearity for deep networks
- **Dropout**: Regularization to prevent overfitting
- **Data Augmentation**: Expanding training data variations
- **Transfer Learning**: Leveraging pre-trained models
- **Batch Normalization**: Improving training stability
- **Model Checkpointing**: Saving best performing models
- **Visualization**: Understanding model decisions

---

## 🎯 Getting Started with Each Model

### Example: Running LeNet

```python
# Import dependencies
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Build model
model = models.Sequential([
    layers.Conv2D(6, 5, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(16, 5, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile & Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_split=0.1, batch_size=128)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')
```

For detailed implementations, check the individual Jupyter notebooks!

---

## 🤝 Contributing

We welcome contributions! This project thrives on community input.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/YourFeature`
3. **Commit** changes: `git commit -m 'Add YourFeature'`
4. **Push** to branch: `git push origin feature/YourFeature`
5. **Submit** a Pull Request

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📚 Learning Resources

### Original Papers

| Architecture | Link |
|:------------|:-----|
| LeNet | [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) |
| AlexNet | [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) |
| VGGNet | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) |
| InceptionV3 | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) |
| Xception | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) |
| ResNet | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |

### Recommended Tutorials

- [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)
- [Stanford CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

---

## ⚙️ Configuration & Settings

### Python Environment

```bash
# Install from requirements
pip install -r requirements.txt

# Or with specific versions
pip install tensorflow==2.13.0 keras==2.13.0 jupyter numpy matplotlib pandas scikit-learn pillow
```

### Jupyter Configuration

```bash
# Generate Jupyter config (optional)
jupyter notebook --generate-config

# Run with custom settings
jupyter notebook --ip=0.0.0.0 --allow-root
```

---

## 📋 FAQ

**Q: Do I need a GPU?**  
A: No, but it's highly recommended for faster training. CPU training is supported but slower.

**Q: Which model should I start with?**  
A: Start with LeNet if you're new to CNNs. It's simple and well-documented.

**Q: Can I use these models for production?**  
A: Yes! All implementations follow best practices. See [ARCHITECTURE.md](ARCHITECTURE.md) for production considerations.

**Q: How do I fine-tune a model on my dataset?**  
A: Check the transfer learning notebooks (VGGtransferLearning.ipynb, XceptionTransferLearning.ipynb, ResNet_Transfer_RockPaperScissors.ipynb)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original architecture authors (LeCun, Krizhevsky, Simonyan, Szegedy, He, et al.)
- TensorFlow and Keras teams
- The deep learning research community

---

<div align="center">


