# Quick Start Guide

Get up and running in 5 minutes! ⚡

## Installation (Windows/Mac/Linux)

```bash
# 1. Clone repository
git clone https://github.com/your-username/research-papers-implementation.git
cd "Research Papers Implementation"

# 2. Create virtual environment
python -m venv cnn_env

# 3. Activate virtual environment
# Windows:
cnn_env\Scripts\activate
# Mac/Linux:
source cnn_env/bin/activate

# 4. Install packages
pip install -r requirements.txt

# 5. Launch Jupyter
jupyter notebook
```

## Run Models

### LeNet-5 (Easiest - Start Here!)

```bash
jupyter notebook LeNetImplementation.ipynb
```

- MNIST dataset (~50 MB)
- Training time: 2-5 minutes
- Expected accuracy: 99%

### AlexNet

```bash
jupyter notebook AlexNetImplementation.ipynb
```

- ImageNet-like dataset
- Training time: 30+ minutes (GPU recommended)
- Expected accuracy: 63%+

### VGGNet

```bash
jupyter notebook VGGImplementation.ipynb
```

- ImageNet dataset
- Training time: Hours (GPU strongly recommended)
- Expected accuracy: 70%+

### Transfer Learning

```bash
jupyter notebook VGGtransferLearning.ipynb
```

- Shows how to use pre-trained models
- Fast training on custom datasets
- Great for production use

### InceptionV3

```bash
jupyter notebook InceptionV3Implementation.ipynb
```

- Input size: 299×299, uses Inception modules

### Xception

```bash
jupyter notebook XceptionPretrained.ipynb
```

- Uses depthwise separable convolutions; demonstrates pretrained weights

### Xception Transfer Learning

```bash
jupyter notebook XceptionTransferLearning.ipynb
```

- Shows how to use pre-trained Xception model on custom datasets
- Fast training and high performance

## Troubleshooting

### No module named 'tensorflow'

```bash
pip install -r requirements.txt
```

### Out of Memory

Edit notebook and reduce batch size:

```python
model.fit(x_train, y_train, batch_size=32)  # Try 16 or 8
```

### GPU not detected

```bash
pip install tensorflow[and-cuda]
```

### Port already in use

```bash
jupyter notebook --port 8889
```

## Next Steps

1. Read [README.md](README.md) for project overview
2. Check [setup.md](setup.md) for detailed setup
3. Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
4. Start with LeNet notebook to learn basics
5. Experiment with AlexNet and VGG

## Common Commands

```bash
# Update packages
pip install --upgrade -r requirements.txt

# Stop Jupyter (in terminal)
Ctrl + C

# Deactivate virtual environment
deactivate

# Reinstall fresh
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Project Structure

```
📦 Research Papers Implementation/
 ├── 📄 README.md                    ← Start here
 ├── 📄 setup.md                     ← Detailed setup
 ├── 📄 QUICK_START.md               ← This file
 ├── 📄 ARCHITECTURE.md              ← Technical details
 ├── 📄 requirements.txt             ← Python packages
 ├── 📄 pyproject.toml               ← Project config
 │
 ├── 📓 LeNetImplementation.ipynb     ← Best for learning
 ├── 📓 AlexNetImplementation.ipynb   ← Advanced
 ├── 📓 VGGImplementation.ipynb       ← Very advanced
 ├── 📓 VGGtransferLearning.ipynb     ← Production use
 ├── 📓 InceptionV3Implementation.ipynb ← InceptionV3 (if added)
 ├── 📓 XceptionPretrained.ipynb       ← Xception pretrained
 ├── 📓 XceptionTransferLearning.ipynb ← Transfer learning with Xception
 │
 └── 🏋️  bestLeNet.h5                 ← Pre-trained model
```

## Key Features

✅ **Educational** - Learn CNNs from first principles  
✅ **Production-Ready** - Transfer learning examples  
✅ **Well-Documented** - Every model explained  
✅ **Modern Keras** - Uses latest TensorFlow 2.13+  
✅ **GPU Support** - NVIDIA CUDA acceleration  
✅ **Active Development** - Community contributions welcome

## Resources

- 📚 [LeNet Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- 📚 [AlexNet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- 📚 [VGGNet Paper](https://arxiv.org/abs/1409.1556)
- 🎓 [Fast.ai Course](https://www.fast.ai/)
- 🎓 [Andrew Ng's Course](https://www.deeplearning.ai/)

## Need Help?

1. Check [setup.md](setup.md) troubleshooting section
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for model details
3. Open an issue on GitHub
4. Check existing issues and discussions

---

**Ready? Start with LeNet!** 🚀

```bash
jupyter notebook LeNetImplementation.ipynb
```
