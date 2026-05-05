
# Comparison of Inception v1, v2, and v3

This document outlines the key differences between Inception v1, v2, and v3, which are popular convolutional neural network architectures developed by Google for image classification tasks.

Inception v1 (GoogLeNet - 2014)
-------------------------------
Key Innovations:
- Introduced the Inception module: Parallel convolutions with filters of different sizes (1x1, 3x3, 5x5) and a pooling layer.
- Employed 1x1 convolutions for dimensionality reduction before applying expensive filters.
- Deep architecture with 22 layers, yet computationally efficient.
- Replaced fully connected layers with Global Average Pooling to reduce parameters.

Limitations:
- 5x5 convolutions were still computationally heavy.
- Hand-crafted architecture without formal optimization techniques.

Inception v2 (2015)
-------------------
Key Improvements:
- Replaced 5x5 convolutions with two stacked 3x3 convolutions to reduce cost.
- Factorized convolutions:
  - 3x3 -> 1x3 followed by 3x1
- Batch Normalization added to auxiliary classifiers for improved training stability.

Benefits:
- Reduced computation without compromising performance.
- Enabled deeper networks and better generalization.

Inception v3 (2015-2016)
------------------------
Enhancements over v2:
- Further factorization of larger convolutions:
  - e.g., 7x7 -> 1x7 followed by 7x1
- Introduced Label Smoothing for regularization.
- Used RMSProp optimizer with auxiliary classifiers.
- Incorporated grid size reduction modules using stride-2 convolutions.

Advantages:
- Improved classification accuracy and efficiency.
- More complex but modular and scalable architecture (~48 layers).

Summary Comparison Table
------------------------
Feature                     | Inception v1         | Inception v2                     | Inception v3
---------------------------|----------------------|----------------------------------|---------------------------------------
Year                       | 2014                 | 2015                             | 2015/2016
Core Innovation            | Inception modules    | Factorized convolutions          | Asymmetric convolutions
Dimensionality Reduction   | 1x1 convolutions     | Improved + more factorization    | Further factorization + asymmetry
Optimizer                  | SGD                  | SGD / RMSProp                    | RMSProp
Regularization             | Dropout              | BatchNorm (aux classifiers)      | Label smoothing + BatchNorm
Computational Efficiency   | High                 | Higher (via smarter modules)     | Even higher (deeper yet efficient)
Final Classifier           | Global Avg Pooling   | Same                             | Same

References
----------
- Szegedy et al. (2014). Going Deeper with Convolutions. https://arxiv.org/abs/1409.4842
- Szegedy et al. (2015). Rethinking the Inception Architecture for Computer Vision. https://arxiv.org/abs/1512.00567
