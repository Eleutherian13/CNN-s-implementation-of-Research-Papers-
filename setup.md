# Setup Guide - Research Papers Implementation

Complete step-by-step guide for setting up and running the CNN implementations.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Verification](#verification)
- [GPU Setup (Optional)](#gpu-setup-optional)
- [Troubleshooting](#troubleshooting)
- [Project Organization](#project-organization)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4 GB (8 GB recommended)
- **Storage**: 5 GB for dependencies and datasets
- **Internet**: Required for downloading packages and pre-trained models

### Recommended Specifications
- **Python**: 3.10 or 3.11
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 20 GB SSD for faster I/O

## Installation Steps

### Step 1: Clone or Download the Repository

```bash
# Using Git
git clone https://github.com/your-username/research-papers-implementation.git
cd "Research Papers Implementation"

# Or download as ZIP and extract
```

### Step 2: Create a Virtual Environment

Creating a virtual environment isolates project dependencies from system Python.

**On Windows:**
```bash
python -m venv cnn_env
cnn_env\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv cnn_env
source cnn_env/bin/activate
```

### Step 3: Upgrade pip, setuptools, and wheel

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow 2.13+
- Keras
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Jupyter
- Pillow
- And other supporting libraries

### Step 5: Verify Installation

```bash
# Test Python
python --version

# Test TensorFlow/Keras
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Test Jupyter
jupyter notebook --version
```

### Step 6: Launch Jupyter Notebook

```bash
jupyter notebook
```

This opens Jupyter in your default browser at `http://localhost:8888`

## Verification

### Quick Test Script

Create a file `test_setup.py` and run it:

```python
#!/usr/bin/env python3
"""Verify the setup is correct."""

import sys
import subprocess

def check_import(module_name, pretty_name=None):
    """Check if a module can be imported."""
    pretty_name = pretty_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {pretty_name} is installed")
        return True
    except ImportError:
        print(f"✗ {pretty_name} is NOT installed")
        return False

def main():
    """Run all checks."""
    print("=" * 50)
    print("Setup Verification")
    print("=" * 50)
    
    print(f"\nPython Version: {sys.version}")
    print(f"Python Executable: {sys.executable}\n")
    
    modules = [
        ("tensorflow", "TensorFlow"),
        ("keras", "Keras"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("jupyter", "Jupyter"),
        ("PIL", "Pillow"),
    ]
    
    results = [check_import(mod, pretty) for mod, pretty in modules]
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All dependencies installed successfully!")
        print("You can now run: jupyter notebook")
    else:
        print("✗ Some dependencies are missing.")
        print("Run: pip install -r requirements.txt")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

Run it:
```bash
python test_setup.py
```

## GPU Setup (Optional)

GPU acceleration significantly speeds up training. If you have an NVIDIA GPU:

### Prerequisites
- NVIDIA GPU (GeForce, Tesla, Quadro)
- NVIDIA Driver installed
- CUDA 11.8+ installed
- cuDNN 8.6+ installed

### Installation

```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]
```

### Verification

```python
import tensorflow as tf

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPUs found: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("✗ No GPU found. Using CPU.")

# Test GPU computation
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    c = tf.matmul(a, b)
    print(f"GPU Computation Result:\n{c}")
```

### Common CUDA/cuDNN Issues

**Issue**: CUDA not found
```bash
# Reinstall with specific CUDA version
pip uninstall tensorflow
pip install tensorflow==2.13.0
```

**Issue**: Out of Memory during GPU training
```python
# Reduce GPU memory usage
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

## Project Organization

```
Research Papers Implementation/
│
├── Documentation Files
│   ├── README.md                     # Project overview
│   ├── setup.md                      # This file
│   ├── ARCHITECTURE.md               # Model architecture details
│   ├── CONTRIBUTING.md               # Contribution guidelines
│   ├── LICENSE                       # MIT License
│   └── requirements.txt              # Python dependencies
│
├── Implementation Notebooks
│   ├── LeNetImplementation.ipynb      # LeNet-5 (1998)
│   ├── AlexNetImplementation.ipynb    # AlexNet (2012)
│   ├── VGGImplementation.ipynb        # VGGNet (2014)
│   └── VGGtransferLearning.ipynb      # VGG Transfer Learning
│
└── Model Checkpoints
    └── bestLeNet.h5                  # Pre-trained LeNet weights
```

### Running Individual Models

#### LeNet-5
```bash
jupyter notebook LeNetImplementation.ipynb
# Recommended for learning, uses MNIST dataset (~50 MB)
```

#### AlexNet
```bash
jupyter notebook AlexNetImplementation.ipynb
# Requires more GPU memory, uses ImageNet-like datasets
```

#### VGG Models
```bash
jupyter notebook VGGImplementation.ipynb
# Very deep, requires good GPU or patience
```

#### Transfer Learning
```bash
jupyter notebook VGGtransferLearning.ipynb
# Shows how to use pre-trained models on new tasks
```

## Data Management

### Dataset Locations

By default, Keras downloads datasets to:
- **Windows**: `C:\Users\<username>\.keras\datasets\`
- **Linux/Mac**: `~/.keras/datasets/`

### Managing Large Datasets

```python
# Change download location
import os
os.environ['KERAS_HOME'] = '/path/to/custom/location'

# Download dataset
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### Clearing Cache

```bash
# Windows
rmdir /s %USERPROFILE%\.keras

# Linux/Mac
rm -rf ~/.keras
```

## Troubleshooting

### Issue: Python Version Mismatch

**Error**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
# Verify Python version
python --version

# Ensure virtual environment is activated
which python  # Linux/Mac
where python  # Windows

# Should point to your venv, not system Python
```

### Issue: pip Installation Fails

**Error**: `ERROR: Could not find a version that satisfies the requirement`

**Solutions**:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install with specific Python version
python -m pip install -r requirements.txt

# Retry with timeout
pip install -r requirements.txt --default-timeout=1000
```

### Issue: Out of Memory

**Error**: `tensorflow.python.framework.errors_impl.ResourceExhaustedError`

**Solutions**:
```python
# Reduce batch size
model.fit(x_train, y_train, batch_size=32, ...)  # Default 32, try 16

# Use mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Clear GPU memory
import tensorflow as tf
tf.keras.backend.clear_session()
```

### Issue: Jupyter Won't Start

**Error**: `Address already in use`

**Solution**:
```bash
# Use different port
jupyter notebook --port 8889

# Or kill existing process
# Windows: taskkill /IM jupyter.exe /F
# Linux/Mac: pkill -f jupyter
```

### Issue: GPU Not Detected

**Error**: `No GPU available`

**Solutions**:
```bash
# Check CUDA installation
nvcc --version  # Should show CUDA version

# Verify GPU is recognized
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])"

# Force CPU usage for testing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Issue: CUDA Out of Memory During Training

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce batch size: 128 → 64 → 32
2. Reduce image size: 224 → 112 → 64
3. Use gradient checkpointing
4. Close other GPU programs
5. Update NVIDIA drivers

## Performance Optimization

### Training Speed Tips

```python
# 1. Use mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 2. Use tf.data for efficient data loading
def create_dataset(x, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# 3. Use callbacks for early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(..., callbacks=[early_stop])

# 4. Use learning rate scheduling
from tensorflow.keras.optimizers.schedules import ExponentialDecay
lr_schedule = ExponentialDecay(1e-3, 100000, 0.96)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

## Environment Variables

Useful environment variables for TensorFlow:

```bash
# Control GPU visibility
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1

# Control logging level
export TF_CPP_MIN_LOG_LEVEL=2  # Suppress INFO and WARNING

# Enable memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# For Windows (Command Prompt):
set CUDA_VISIBLE_DEVICES=0
set TF_CPP_MIN_LOG_LEVEL=2
```

## Next Steps

1. **Read the README.md** for project overview
2. **Start with LeNetImplementation.ipynb** to learn basics
3. **Progress to more complex models** (AlexNet, VGG)
4. **Experiment with Transfer Learning** (VGGtransferLearning.ipynb)
5. **Modify and improve** the implementations for your use cases

## Getting Help

- Check [ARCHITECTURE.md](ARCHITECTURE.md) for model details
- Review individual notebook comments and docstrings
- Open an issue on GitHub with:
  - Python version
  - TensorFlow version
  - Operating system
  - Full error message
  - Steps to reproduce

---

**Happy implementing!** 🚀

*For the latest updates and discussions, visit the project GitHub page.*