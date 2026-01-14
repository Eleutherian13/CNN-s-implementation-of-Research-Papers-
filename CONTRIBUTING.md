# Contributing Guidelines

Thank you for your interest in contributing to the Research Papers Implementation project! We welcome contributions from everyone, whether you're a beginner or an experienced developer.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to make participation in our project and our community a harassment-free experience for everyone.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:
- Using welcoming and inclusive language
- Being respectful of differing opinions, viewpoints, and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior include:
- Harassment or derogatory comments
- Trolling or insulting comments
- Personal attacks of any kind
- Unwelcome sexual attention or advances
- Any other conduct which could reasonably be considered inappropriate

## How to Contribute

### Types of Contributions

1. **Bug Reports**: Report issues you've found
2. **Documentation**: Improve or add documentation
3. **Code Improvements**: Optimize existing code
4. **New Features**: Add new implementations or features
5. **Tests**: Improve test coverage
6. **Examples**: Create educational examples

### Areas We Need Help With

- [ ] Implementing additional CNN architectures (ResNet, Inception, MobileNet)
- [ ] Creating tutorial notebooks for beginners
- [ ] Improving documentation and examples
- [ ] Performance optimizations
- [ ] Fixing bugs and issues
- [ ] Improving code quality and readability
- [ ] Adding comprehensive tests
- [ ] Creating visualization tools

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Understanding of CNNs and deep learning basics

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   # Go to GitHub and click "Fork"
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/research-papers-implementation.git
   cd "Research Papers Implementation"
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/original-owner/research-papers-implementation.git
   ```

4. **Create virtual environment**
   ```bash
   python -m venv dev_env
   source dev_env/bin/activate  # Windows: dev_env\Scripts\activate
   ```

5. **Install dependencies in development mode**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

6. **Create development branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/descriptive-name
# or for bug fixes:
git checkout -b bugfix/issue-number
```

### 2. Make Your Changes

Follow the [Coding Standards](#coding-standards) while making changes.

### 3. Test Your Changes

```bash
# Run tests
pytest tests/

# For notebooks, manually verify execution
jupyter notebook

# Check code quality
pylint your_file.py
black --check your_file.py
```

### 4. Commit Your Changes

Follow the [Commit Message Guidelines](#commit-message-guidelines).

```bash
git add .
git commit -m "feat: add new model implementation"
```

### 5. Keep Updated with Main Branch

```bash
git fetch upstream
git rebase upstream/main
```

### 6. Push Your Changes

```bash
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```python
# Good: Clear variable names, proper spacing
def create_convolution_layer(filters, kernel_size, activation='relu'):
    """
    Create a convolutional layer.
    
    Args:
        filters (int): Number of convolutional filters
        kernel_size (tuple): Size of the convolutional kernel
        activation (str): Activation function name
        
    Returns:
        tf.keras.layers.Conv2D: Configured convolutional layer
    """
    return Conv2D(
        filters,
        kernel_size,
        padding='same',
        activation=activation
    )
```

### Notebook Style

For Jupyter notebooks:

```python
# Clear cell organization
# 1. Imports at the top
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 2. Configuration and hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# 3. Data loading and preprocessing
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 4. Model building
model = tf.keras.Sequential([
    # Architecture here
])

# 5. Training and evaluation
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
```

### Documentation Requirements

Every function should have:
- Docstring with description
- Args section with types
- Returns section with type
- Raises section if applicable

```python
def train_model(model, train_data, validation_data, epochs=100):
    """
    Train a Keras model with provided data.
    
    Args:
        model (tf.keras.Model): The model to train
        train_data (tuple): (X_train, y_train) training data
        validation_data (tuple): (X_val, y_val) validation data
        epochs (int, optional): Number of training epochs. Defaults to 100.
        
    Returns:
        tf.keras.callbacks.History: Training history object
        
    Raises:
        ValueError: If data shapes are incompatible
        TypeError: If model is not a Keras model
    """
    # Implementation
    pass
```

### Code Quality Tools

We recommend using these tools:

```bash
# Format code
black your_file.py

# Check style
flake8 your_file.py

# Type checking
mypy your_file.py

# Linting
pylint your_file.py
```

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring without feature changes
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```bash
# Good commit messages
git commit -m "feat(lenet): implement LeNet-5 architecture"
git commit -m "fix(alexnet): correct dropout rate initialization"
git commit -m "docs(setup): add GPU installation instructions"
git commit -m "refactor(vgg): simplify block creation with function"
git commit -m "test(models): add architecture validation tests"
```

### Bad Commit Messages (Avoid)

```bash
# Don't do this
git commit -m "fixed stuff"
git commit -m "updates"
git commit -m "WIP: incomplete changes"
```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**
   ```bash
   pytest tests/
   ```

2. **Update documentation**
   - Update README.md if needed
   - Add docstrings to new functions
   - Include examples if adding new features

3. **Check for conflicts**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Clean commit history**
   ```bash
   # Squash commits if needed
   git rebase -i upstream/main
   ```

### Submitting a PR

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request on GitHub**
   - Use a descriptive title
   - Reference any related issues
   - Describe your changes clearly
   - Include before/after if visual changes

3. **PR Title Format**
   ```
   [Type] Brief description
   
   Examples:
   [Feature] Implement ResNet architecture
   [Bug Fix] Fix gradient explosion in VGG training
   [Documentation] Add transfer learning guide
   ```

4. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Motivation and Context
   Why is this change needed?
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   
   ## Testing
   How was this tested?
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation is updated
   - [ ] Tests are included
   - [ ] No new warnings generated
   ```

### During Review

- Be responsive to reviewer feedback
- Update your branch with requested changes
- Push updates to the same branch (creates new commits automatically)
- Request re-review when ready

## Reporting Bugs

### Before Reporting

- Check existing issues to avoid duplicates
- Test with the latest version
- Gather system information:
  - Python version
  - TensorFlow version
  - Operating system
  - GPU information if applicable

### Bug Report Template

**Title**: Brief description of the bug

**Environment**:
- Python version: 3.10
- TensorFlow version: 2.13.0
- OS: Windows 10 / macOS / Linux
- GPU: Yes/No (NVIDIA RTX 3090 if yes)

**Steps to Reproduce**:
1. First step
2. Second step
3. ...

**Expected Behavior**:
What should happen?

**Actual Behavior**:
What actually happened?

**Logs and Errors**:
```
Full error message and traceback
```

**Screenshots** (if applicable):
Attach images if relevant

## Suggesting Enhancements

### Enhancement Request Template

**Title**: Brief description of enhancement

**Motivation**:
Why is this enhancement needed?

**Proposed Solution**:
How should this be implemented?

**Alternative Solutions**:
Other possible approaches?

**Additional Context**:
Any other information?

### Enhancement Ideas

Some areas we're interested in:

- [ ] Implementing additional architectures (ResNet, DenseNet, EfficientNet)
- [ ] Creating interactive visualization tools
- [ ] Adding more transfer learning examples
- [ ] Performance optimization tutorials
- [ ] Model compression techniques (quantization, pruning)
- [ ] Explainability methods (GradCAM, etc.)

## Community

- **Discussions**: Use GitHub Discussions for questions and ideas
- **Issues**: Report bugs and request features
- **Pull Requests**: Submit improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:
- Open a GitHub issue with the "question" label
- Start a discussion on GitHub Discussions
- Check existing documentation

---

## Thank You! 🙏

We appreciate your contributions to making this project better for everyone. Your efforts help advance open-source deep learning education!

### Recognition

Contributors will be recognized in:
- Project README.md (with permission)
- Release notes for feature additions
- Our contributors page

---

*Last Updated: January 2024*