# Research Papers Implementation: Classic CNN Architectures with Keras

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.4%2B-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive collection of implementations of seminal Convolutional Neural Network (CNN) architectures from groundbreaking research papers, recreated using the Keras deep learning library. This project serves as both an educational resource and a practical toolkit for understanding and applying these foundational models.

## 📚 Implemented Architectures

### 🏗️ LeNet-5 (1998)

- **Paper**: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- **Notebook**: [LeNetImplementation.ipynb](LeNetImplementation.ipynb)
- **Task**: Handwritten digit recognition (MNIST dataset)
- **Key Features**: Convolutional layers, pooling layers, fully connected layers
- **Model Checkpoint**: [bestLeNet.h5](bestLeNet.h5)

### 🧠 AlexNet (2012)

- **Paper**: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- **Notebook**: [AlexNetImplementation.ipynb](AlexNetImplementation.ipynb)
- **Task**: ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
- **Key Features**: Deep architecture, ReLU activation, dropout regularization, data augmentation
- **Achievement**: Breakthrough model that revolutionized computer vision

### 🏛️ VGGNet (2014)

- **Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- **Notebook**: [VGGImplementation.ipynb](VGGImplementation.ipynb)
- **Task**: ImageNet classification with very deep networks
- **Key Features**: Small 3×3 convolutional filters, stacked architecture, uniform structure
- **Variants**: VGG-16 and VGG-19

### 🔄 VGG with Transfer Learning

- **Notebook**: [VGGtransferLearning.ipynb](VGGtransferLearning.ipynb)
- **Task**: Applying pre-trained VGG models to custom datasets
- **Approach**: Fine-tuning and feature extraction strategies
- **Use Cases**: Image classification on new domains with limited data

### 🔬 InceptionV3 (2015)

- **Paper**: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
- **Notebook**: `InceptionV3Implementation.ipynb` (add to repo if not present)
- **Task**: Image classification with efficient inception modules
- **Key Features**: Factorized convolutions, aggressive regularization, auxiliary classifiers

### ✨ Xception (2017)

- **Paper**: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- **Notebook**: [XceptionPretrained.ipynb](XceptionPretrained.ipynb)
- **Task**: High-performance image classification using depthwise separable convolutions
- **Key Features**: Depthwise separable convolutions, larger model capacity with efficiency

### � ResNet (2015)
- **Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Notebooks**: 
  - [ResnetImplementation.ipynb](ResnetImplementation.ipynb) - Core ResNet architecture
  - [ResNet_Transfer_RockPaperScissors.ipynb](ResNet_Transfer_RockPaperScissors.ipynb) - Transfer learning example
- **Task**: Image classification with residual connections and transfer learning
- **Key Features**: Skip connections, residual blocks, enables very deep networks (50/101/152 layers)
- **Approach**: Fine-tuning and feature extraction strategies
- **Use Cases**: Image classification on new domains with limited data

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation (5 minutes)

```bash
# Clone the repository
git clone <repository-url>
cd "Research Papers Implementation"

# Create and activate virtual environment
python -m venv cnn_env
source cnn_env/bin/activate  # On Windows: cnn_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

For detailed setup instructions, see [setup.md](setup.md).

## 📋 Project Structure

```
Research Papers Implementation/
├── README.md                          # This file
├── setup.md                           # Detailed setup guide
├── requirements.txt                   # Project dependencies
├── ARCHITECTURE.md                    # Detailed architecture descriptions
├── CONTRIBUTING.md                    # Contribution guidelines
├── LICENSE                            # MIT License
│
├── LeNetImplementation.ipynb          # LeNet-5 implementation
├── AlexNetImplementation.ipynb        # AlexNet implementation
├── VGGImplementation.ipynb            # VGGNet implementation
├── VGGtransferLearning.ipynb          # Transfer learning with VGG
├── InceptionV3Implementation.ipynb    # InceptionV3 implementation (if added)
├── XceptionPretrained.ipynb           # Xception pretrained model
├── XceptionTransferLearning.ipynb     # Transfer learning with Xception
├── ResnetImplementation.ipynb         # ResNet core implementation
├── ResNet_Transfer_RockPaperScissors.ipynb  # ResNet transfer learning
│
└── bestLeNet.h5                       # Pre-trained LeNet weights
```

## 📦 Dependencies

| Package      | Version | Purpose                        |
| ------------ | ------- | ------------------------------ |
| TensorFlow   | 2.13+   | Deep learning framework        |
| Keras        | 2.13+   | High-level neural networks API |
| NumPy        | 1.24+   | Numerical computations         |
| Matplotlib   | 3.7+    | Data visualization             |
| Pandas       | 2.0+    | Data manipulation              |
| Scikit-learn | 1.3+    | ML utilities & metrics         |
| Jupyter      | 1.0+    | Interactive notebooks          |
| Pillow       | 10.0+   | Image processing               |

## 🎯 Getting Started with Each Model

### Running LeNet

```python
# Data loading
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data preprocessing
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Model training
history = model.fit(x_train, y_train, epochs=10, validation_split=0.1, batch_size=128)

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')
```

### Running AlexNet & VGG

- Similar workflow to LeNet
- Typically use ImageNet or custom datasets
- Longer training time due to model depth
- Check individual notebooks for dataset instructions

## 💡 Key Concepts Covered

- **Convolutional Layers**: Feature extraction at different scales
- **Pooling Layers**: Dimensionality reduction and noise robustness
- **ReLU Activation**: Non-linearity for deep networks
- **Dropout**: Regularization to prevent overfitting
- **Data Augmentation**: Expanding training data variations
- **Transfer Learning**: Leveraging pre-trained models
- **Model Checkpointing**: Saving best performing models
- **Visualization**: Understanding model decisions

## 📊 Model Comparison

| Model       | Year | Params | Depth     | Top-1 Accuracy   |
| ----------- | ---- | ------ | --------- | ---------------- |
| LeNet-5     | 1998 | ~60K   | 5 layers  | 99.2% (MNIST)    |
| AlexNet     | 2012 | 62.4M  | 8 layers  | 63.3% (ImageNet) |
| VGG-16      | 2014 | 138M   | 16 layers | 71.3% (ImageNet) |
| VGG-19      | 2014 | 144M   | 19 layers | 72.4% (ImageNet) |
| Xception    | 2017 | 22.9M  | 71 layers | 79.0% (ImageNet) |
| ResNet-50   | 2015 | 25.5M  | 50 layers | 76.0% (ImageNet) |
| ResNet-101  | 2015 | 44.5M  | 101 layers | 77.4% (ImageNet) |

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m 'Add YourFeature'`
4. Push to branch: `git push origin feature/YourFeature`
5. Submit a Pull Request

## 📚 Learning Resources

### Foundational Papers

- [LeNet Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) - Gradient-Based Learning
- [AlexNet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) - ImageNet Classification
- [VGGNet Paper](https://arxiv.org/abs/1409.1556) - Very Deep Convolutional Networks

### Online Courses

- Fast.ai: Practical Deep Learning
- Andrew Ng's Deep Learning Specialization
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition

### Books

- "Deep Learning" by Goodfellow, Bengio, Courville
- "Hands-On Machine Learning" by Aurélien Géron

## 🐛 Troubleshooting

### GPU Support

To enable GPU acceleration:

```bash
pip install tensorflow[and-cuda]  # For NVIDIA GPUs
```

### Out of Memory Errors

- Reduce batch size in model.fit()
- Reduce image resolution
- Use gradient checkpointing
- Try mixed precision training

### Slow Training

- Check if GPU is being used: `tf.config.list_physical_devices('GPU')`
- Ensure data preprocessing is efficient
- Use data augmentation on-the-fly

See [setup.md](setup.md#troubleshooting) for more solutions.

## 📈 Performance Metrics

Each notebook includes:

- **Training & Validation Curves**: Loss and accuracy plots
- **Confusion Matrices**: Per-class performance analysis
- **Precision, Recall, F1-Score**: Detailed classification metrics
- **Training Time**: Execution duration tracking

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Yann LeCun** - LeNet-5 pioneer
- **Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever** - AlexNet creators
- **Karen Simonyan & Andrew Zisserman** - VGGNet architects
- **Keras & TensorFlow Teams** - Excellent deep learning frameworks
- The global AI research community for foundational contributions

## 📞 Support & Questions

- Open an issue for bugs or questions
- Check existing issues before creating new ones
- Provide detailed error messages and reproducible examples

## 🌟 Citation

If you use this implementation in your research, please cite:

```bibtex
@software{cnn_implementations_2024,
  title={Research Papers Implementation: Classic CNN Architectures with Keras},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/research-papers-implementation}
}
```

---

**Happy Learning! 🚀**

_Last Updated: January 18, 2026_  
_Status: Active Development_
