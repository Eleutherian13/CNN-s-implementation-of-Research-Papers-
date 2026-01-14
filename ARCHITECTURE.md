# CNN Architecture Documentation

Detailed technical documentation of all implemented CNN architectures with their specifications, design decisions, and key innovations.

---

## Table of Contents
1. [LeNet-5](#lenet-5)
2. [AlexNet](#alexnet)
3. [VGGNet](#vggnet)
4. [Comparison](#comparison)

---

## LeNet-5

### Overview
**Year**: 1998  
**Authors**: Yann LeCun, Léon Bottou, Yoann Bengio, Patrick Haffner  
**Paper**: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)  
**Original Task**: Handwritten digit recognition for postal code reading  
**Dataset**: MNIST (70,000 images of 28×28 handwritten digits)

### Architecture Overview

```
Input (28×28×1)
    ↓
Conv1 (6 filters, 5×5) → ReLU → MaxPool (2×2)
    ↓ (24×24×6)
Conv2 (16 filters, 5×5) → ReLU → MaxPool (2×2)
    ↓ (8×8×16)
Conv3 (120 filters, 5×5) → ReLU
    ↓ (1×1×120)
Dense (84 units) → ReLU
    ↓
Dense (10 units) → Softmax
    ↓
Output (10 classes)
```

### Layer Details

| Layer | Type | Filters | Kernel | Stride | Output | Parameters |
|-------|------|---------|--------|--------|--------|-----------|
| Input | - | 1 | - | - | 28×28×1 | 0 |
| Conv1 | Conv2D | 6 | 5×5 | 1 | 24×24×6 | 156 |
| Pool1 | MaxPool | - | 2×2 | 2 | 12×12×6 | 0 |
| Conv2 | Conv2D | 16 | 5×5 | 1 | 8×8×16 | 2,416 |
| Pool2 | MaxPool | - | 2×2 | 2 | 4×4×16 | 0 |
| Conv3 | Conv2D | 120 | 5×5 | 1 | 1×1×120 | 48,120 |
| Dense1 | Dense | 84 | - | - | 84 | 10,164 |
| Dense2 | Dense | 10 | - | - | 10 | 850 |

**Total Parameters**: ~60,000

### Key Features

1. **Convolutional Layers**: Extract local features at multiple scales
2. **Pooling Layers**: Reduce spatial dimensions and computational complexity
3. **ReLU Activation**: Non-linearity (historically used tanh)
4. **Minimal Parameters**: Efficient architecture suitable for embedded systems

### Historical Significance

- First successful deep learning application for digit recognition
- Demonstrated viability of CNNs for real-world tasks
- Inspired modern deep learning approaches

### Training Configuration

```python
# Optimization
Optimizer: SGD with momentum (0.9)
Learning Rate: 0.01 (with decay schedule)
Batch Size: 128
Epochs: 20-50

# Regularization
L2 Penalty: 0.0005
No Dropout (original design)

# Data
Training Set: 60,000 samples
Test Set: 10,000 samples
Data Augmentation: None (minimal in original)
```

### Expected Performance

- **Training Accuracy**: 98-99%
- **Test Accuracy**: 99.2%
- **Training Time**: ~2-5 minutes (on modern GPU)

### Modern Implementation Notes

- Uses ReLU instead of tanh (better for deep networks)
- May include dropout for better generalization
- Batch normalization can improve convergence speed
- Learning rate scheduling improves final accuracy

---

## AlexNet

### Overview
**Year**: 2012  
**Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
**Paper**: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  
**Competition**: ILSVRC 2012 (ImageNet Large Scale Visual Recognition Challenge)  
**Performance**: 63.3% top-1 accuracy, 85.2% top-5 accuracy  
**Impact**: Ignited the deep learning revolution in computer vision

### Architecture Overview

```
Input (227×227×3)
    ↓
Conv1 (96, 11×11, stride=4) → ReLU → LRN → MaxPool (3×3, stride=2)
    ↓ (55×55×96)
Conv2 (256, 5×5, padding=2) → ReLU → LRN → MaxPool (3×3, stride=2)
    ↓ (27×27×256)
Conv3 (384, 3×3, padding=1) → ReLU
    ↓ (27×27×384)
Conv4 (384, 3×3, padding=1) → ReLU
    ↓ (27×27×384)
Conv5 (256, 3×3, padding=1) → ReLU → MaxPool (3×3, stride=2)
    ↓ (13×13×256)
Flatten → Dense (4096) → ReLU → Dropout (0.5)
    ↓
Dense (4096) → ReLU → Dropout (0.5)
    ↓
Dense (1000) → Softmax
    ↓
Output (1000 classes)
```

### Layer Details

| Layer | Type | Filters | Kernel | Stride | Padding | Output | Parameters |
|-------|------|---------|--------|--------|---------|--------|-----------|
| Conv1 | Conv2D | 96 | 11×11 | 4 | 0 | 55×55×96 | 34,848 |
| Pool1 | MaxPool | - | 3×3 | 2 | - | 27×27×96 | 0 |
| Conv2 | Conv2D | 256 | 5×5 | 1 | 2 | 27×27×256 | 614,400 |
| Pool2 | MaxPool | - | 3×3 | 2 | - | 13×13×256 | 0 |
| Conv3 | Conv2D | 384 | 3×3 | 1 | 1 | 13×13×384 | 884,992 |
| Conv4 | Conv2D | 384 | 3×3 | 1 | 1 | 13×13×384 | 1,327,104 |
| Conv5 | Conv2D | 256 | 3×3 | 1 | 1 | 13×13×256 | 884,992 |
| Pool5 | MaxPool | - | 3×3 | 2 | - | 6×6×256 | 0 |
| Dense1 | Dense | 4096 | - | - | - | 4096 | 37,748,736 |
| Dense2 | Dense | 4096 | - | - | - | 4096 | 16,777,216 |
| Dense3 | Dense | 1000 | - | - | - | 1000 | 4,097,000 |

**Total Parameters**: ~62.4 million

### Key Innovations

1. **Depth**: 8 layers (5 convolutional + 3 fully connected)
   - Demonstrates effectiveness of deep architectures
   - Required careful initialization strategies

2. **ReLU Activation**
   - Much faster convergence than tanh/sigmoid
   - Addresses vanishing gradient problem
   - Became standard in modern deep learning

3. **Dropout Regularization**
   - Randomly deactivates 50% of neurons during training
   - Prevents co-adaptation of features
   - Effective regularization strategy

4. **Local Response Normalization (LRN)**
   - Normalizes activations across feature maps
   - Inspired by biological neural response
   - Later replaced by batch normalization

5. **Data Augmentation**
   - Random crops of 224×224 from 256×256 images
   - Random horizontal flips
   - Effective data augmentation technique
   - Essential for preventing overfitting

6. **GPU Computing**
   - Trained on 2 NVIDIA GTX 580 GPUs
   - Split across GPUs for parallelization
   - Pioneered GPU acceleration in deep learning

### Training Configuration

```python
# Optimization
Optimizer: SGD with momentum (0.9)
Learning Rate: 0.01 (reduced by 10x when loss plateaus)
Batch Size: 128 (split across 2 GPUs: 64 each)
Epochs: 90
Weight Decay: 0.0005

# Regularization
Dropout: 0.5 (in fully connected layers)
L2 Regularization: 0.0005
Local Response Normalization: k=2, n=5, α=0.0001, β=0.75

# Data
Training Set: 1.2 million images
Validation Set: 50,000 images
Test Set: 100,000 images
Image Size: 227×227 (cropped from 256×256)
Data Augmentation: Random crops + flips
```

### Expected Performance

- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~80%
- **Test Top-1 Accuracy**: 63.3%
- **Test Top-5 Accuracy**: 85.2%
- **Training Time**: 5-6 days (on 2 GPUs of the era)

### Modern Implementation Considerations

- Replace LRN with Batch Normalization
- Consider removing GPU split (unnecessary with modern hardware)
- Use ImageNet pre-trained weights for transfer learning
- Adjust learning rates for modern optimizers (Adam)
- Data augmentation is crucial for good performance

---

## VGGNet

### Overview
**Year**: 2014  
**Authors**: Karen Simonyan, Andrew Zisserman (Visual Geometry Group, University of Oxford)  
**Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)  
**Competition**: ILSVRC 2014 (Localization and Classification tasks)  
**Variants**: VGG-16 and VGG-19
**Performance**: VGG-16 achieved 72.8% top-5 accuracy

### Core Contribution

**Systematic study of network depth**: Shows that simply stacking small 3×3 convolutional filters enables learning of more complex features while reducing parameters compared to larger filters.

### VGG-16 Architecture Overview

```
Input (224×224×3)
    ↓
Block 1:
Conv (64, 3×3) → ReLU
Conv (64, 3×3) → ReLU
MaxPool (2×2, stride=2)
    ↓ (112×112×64)

Block 2:
Conv (128, 3×3) → ReLU
Conv (128, 3×3) → ReLU
MaxPool (2×2, stride=2)
    ↓ (56×56×128)

Block 3:
Conv (256, 3×3) → ReLU
Conv (256, 3×3) → ReLU
Conv (256, 3×3) → ReLU
MaxPool (2×2, stride=2)
    ↓ (28×28×256)

Block 4:
Conv (512, 3×3) → ReLU
Conv (512, 3×3) → ReLU
Conv (512, 3×3) → ReLU
MaxPool (2×2, stride=2)
    ↓ (14×14×512)

Block 5:
Conv (512, 3×3) → ReLU
Conv (512, 3×3) → ReLU
Conv (512, 3×3) → ReLU
MaxPool (2×2, stride=2)
    ↓ (7×7×512)

Flatten → Dense (4096) → ReLU → Dropout (0.5)
       → Dense (4096) → ReLU → Dropout (0.5)
       → Dense (1000) → Softmax
    ↓
Output (1000 classes)
```

### VGG-16 Layer Details

| Block | Layer | Filters | Kernel | Input | Output | Parameters |
|-------|-------|---------|--------|-------|--------|-----------|
| 1 | Conv2d | 64 | 3×3 | 224×224×3 | 224×224×64 | 1,792 |
| 1 | Conv2d | 64 | 3×3 | 224×224×64 | 224×224×64 | 36,928 |
| 1 | MaxPool | - | 2×2 | 224×224×64 | 112×112×64 | 0 |
| 2 | Conv2d | 128 | 3×3 | 112×112×64 | 112×112×128 | 73,856 |
| 2 | Conv2d | 128 | 3×3 | 112×112×128 | 112×112×128 | 147,584 |
| 2 | MaxPool | - | 2×2 | 112×112×128 | 56×56×128 | 0 |
| 3 | Conv2d | 256 | 3×3 | 56×56×128 | 56×56×256 | 295,168 |
| 3 | Conv2d | 256 | 3×3 | 56×56×256 | 56×56×256 | 590,080 |
| 3 | Conv2d | 256 | 3×3 | 56×56×256 | 56×56×256 | 590,080 |
| 3 | MaxPool | - | 2×2 | 56×56×256 | 28×28×256 | 0 |
| 4 | Conv2d | 512 | 3×3 | 28×28×256 | 28×28×512 | 1,180,160 |
| 4 | Conv2d | 512 | 3×3 | 28×28×512 | 28×28×512 | 2,359,808 |
| 4 | Conv2d | 512 | 3×3 | 28×28×512 | 28×28×512 | 2,359,808 |
| 4 | MaxPool | - | 2×2 | 28×28×512 | 14×14×512 | 0 |
| 5 | Conv2d | 512 | 3×3 | 14×14×512 | 14×14×512 | 2,359,808 |
| 5 | Conv2d | 512 | 3×3 | 14×14×512 | 14×14×512 | 2,359,808 |
| 5 | Conv2d | 512 | 3×3 | 14×14×512 | 14×14×512 | 2,359,808 |
| 5 | MaxPool | - | 2×2 | 14×14×512 | 7×7×512 | 0 |
| Dense | - | 4096 | - | 7×7×512 | 4096 | 102,764,544 |
| Dense | - | 4096 | - | 4096 | 4096 | 16,781,312 |
| Dense | - | 1000 | - | 4096 | 1000 | 4,097,000 |

**Total VGG-16 Parameters**: ~138 million  
**Total VGG-19 Parameters**: ~144 million

### Key Insights

1. **Depth over Width**
   - Multiple 3×3 filters = one 5×5 or 7×7 filter
   - Deeper networks with smaller kernels are more efficient
   - More non-linearities for learning complex patterns

2. **Uniform Architecture**
   - Repeating structure makes implementation simple
   - Systematic progression of channel depths
   - Easy to understand and modify

3. **Small Receptive Fields**
   - 3×3 kernels are minimal for vision tasks
   - Reduces parameters while maintaining expressiveness
   - Two 3×3 convolutions = 5×5 receptive field
   - Three 3×3 convolutions = 7×7 receptive field

4. **Computational Efficiency**
   - Despite more parameters, comparable FLOPs to AlexNet
   - Better generalization due to depth
   - Effective regularization through architecture

### Training Configuration

```python
# Optimization
Optimizer: SGD with momentum (0.9)
Learning Rate: 0.01 (reduced by factor of 10 when validation plateaus)
Batch Size: 256
Epochs: 370 (approximately 74 epochs)
Weight Decay: 0.0005

# Regularization
Dropout: 0.5 (in fully connected layers)
L2 Regularization: 0.0005
No Local Response Normalization

# Data
Training Set: 1.3 million images
Validation Set: 50,000 images
Test Set: 100,000 images
Image Size: 224×224 (isotropically scaled, random crops)
Data Augmentation:
  - Random crops
  - Random horizontal flips
  - Random RGB color shifts
```

### VGG-16 vs VGG-19

| Aspect | VGG-16 | VGG-19 |
|--------|--------|--------|
| Convolutional Layers | 13 | 16 |
| Fully Connected Layers | 3 | 3 |
| Total Layers | 16 | 19 |
| Parameters | 138M | 144M |
| Block 3 Convs | 3 | 4 |
| Block 4 Convs | 3 | 4 |
| Block 5 Convs | 3 | 4 |
| Top-1 Accuracy | 71.3% | 71.0% |
| Top-5 Accuracy | 89.8% | 89.9% |

### Expected Performance

- **Training Accuracy**: ~95-98%
- **Validation Accuracy**: ~70-72%
- **Test Top-1 Accuracy**: 71.3% (VGG-16) / 71.0% (VGG-19)
- **Test Top-5 Accuracy**: 89.8% (VGG-16) / 89.9% (VGG-19)
- **Training Time**: 2-3 weeks (on single GPU)

### Modern Implementation Improvements

```python
# 1. Batch Normalization
Conv2D(64, (3,3)) → BatchNormalization() → ReLU()

# 2. Learning Rate Scheduling
lr_schedule = PolynomialDecay(initial_lr=0.01, decay_steps=100000, end_lr=0.0001, power=0.5)

# 3. Modern Optimizers
optimizer = Adam(learning_rate=lr_schedule)
# Better convergence than SGD

# 4. Data Augmentation
augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

# 5. Reduced Training Time
# Can achieve competitive accuracy in 100-200 epochs with modern techniques
```

---

## Comparison

### Architecture Comparison

| Aspect | LeNet-5 | AlexNet | VGG-16 | VGG-19 |
|--------|---------|---------|--------|--------|
| **Year** | 1998 | 2012 | 2014 | 2014 |
| **Depth** | 5 layers | 8 layers | 16 layers | 19 layers |
| **Parameters** | 60K | 62.4M | 138M | 144M |
| **Conv Kernels** | 5×5, fixed | 11×5, 3×3 | 3×3 (uniform) | 3×3 (uniform) |
| **Input Size** | 28×28 | 227×227 | 224×224 | 224×224 |
| **Activation** | Tanh → ReLU | ReLU | ReLU | ReLU |
| **Regularization** | None → Dropout | Dropout, LRN | Dropout | Dropout |
| **Bottleneck** | None | None | None | None |
| **Skip Connections** | No | No | No | No |

### Performance Comparison

| Dataset | LeNet-5 | AlexNet | VGG-16 | VGG-19 |
|---------|---------|---------|--------|--------|
| **MNIST** | 99.2% | N/A | N/A | N/A |
| **ImageNet Top-1** | N/A | 63.3% | 71.3% | 71.0% |
| **ImageNet Top-5** | N/A | 85.2% | 89.8% | 89.9% |
| **Training Time** | Hours | Days | Weeks | Weeks |

### Computational Efficiency

| Model | FLOPs | Memory | Speed |
|-------|-------|--------|-------|
| LeNet-5 | ~20M | ~1 MB | Very Fast |
| AlexNet | ~1.5B | ~100 MB | Fast |
| VGG-16 | ~15B | ~140 MB | Slow |
| VGG-19 | ~19B | ~150 MB | Slower |

### Key Evolutionary Steps

```
LeNet-5 (1998)
    ↓ [Key advance: Depth & Scale]
AlexNet (2012)
    ↓ [Key advance: Systematic Depth Study]
VGGNet (2014)
    ↓ [Future: Residual Connections]
ResNet (2015)
    ↓ [Further: Inception Modules]
GoogLeNet (2014)
```

---

## Transfer Learning

### Why Use Transfer Learning?

1. **Limited Training Data**: Learn from ImageNet-pretrained features
2. **Faster Training**: Start from learned representations
3. **Better Generalization**: Leverage millions of training examples
4. **Reduced Computational Cost**: Freeze base layers

### VGG Transfer Learning Strategies

**Strategy 1: Feature Extraction (Frozen Backbone)**
```python
# Load pre-trained VGG
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom classifier
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

**Strategy 2: Fine-Tuning (Partial Unfreezing)**
```python
# Freeze initial layers, unfreeze later layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

for layer in base_model.layers[-4:]:
    layer.trainable = True

# Use lower learning rate
optimizer = Adam(learning_rate=1e-5)
```

**Strategy 3: Full Fine-Tuning**
```python
# Unfreeze all layers
base_model.trainable = True

# Use very low learning rate
optimizer = Adam(learning_rate=1e-6)
```

---

## References

### Original Papers
1. LeNet: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
2. AlexNet: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
3. VGGNet: https://arxiv.org/abs/1409.1556

### Implementations
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)

### Related Architectures
- ResNet (2015): Residual connections for very deep networks
- Inception/GoogLeNet (2014): Multi-scale feature extraction
- MobileNet: Efficient networks for mobile devices
- EfficientNet: Scaling networks for different computational budgets

---

*Last Updated: January 2024*