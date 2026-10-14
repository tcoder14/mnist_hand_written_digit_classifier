# Handwritten Digit Classifier (MNIST)

A deep neural network built with **TensorFlow/Keras** that recognises handwritten digits (0–9) from the **MNIST** dataset. The model is a fully connected network with **Batch Normalization** and **Dropout**, and reaches roughly **98%+ test accuracy**.

## Dataset

 MNIST dataset contains grayscale images of handwritten digits:

- 60,000 training images
- 10,000 test images
- 28×28 pixels per image, 10 classes (digits 0–9)

The dataset is downloaded automatically through `keras.datasets.mnist`, so there is nothing to download manually.

## What the Notebook Covers

1. **Dataset loading**: load MNIST and preview sample images
2. **Preprocessing**: normalise pixels to [0, 1], flatten 28×28 images to 784-length vectors, one-hot encode the labels
3. **Model architecture**: 3 hidden layers with Batch Normalization and Dropout
4. **Training**: Adam optimizer with a 10% validation split
5. **Visualisation**: training vs. validation accuracy and loss curves
6. **Evaluation**: test accuracy, classification report, confusion matrix, and misclassified examples

## Model Architecture

Each hidden block follows the pattern `Dense → BatchNormalization → ReLU → Dropout`.

| Layer | Units | Dropout |
|-------|-------|---------|
| Input | 784 | – |
| Hidden 1 | 256 | 0.3 |
| Hidden 2 | 128 | 0.3 |
| Hidden 3 | 64 | 0.2 |
| Output (softmax) | 10 | – |

- **Batch Normalization** stabilises and speeds up training.
- **Dropout** reduces overfitting by randomly disabling neurons during training.

## Training Setup

| Setting | Value |
|---------|-------|
| Loss | `categorical_crossentropy` |
| Optimizer | Adam (learning rate `1e-3`) |
| Epochs | 15 |
| Batch size | 128 |
| Validation split | 10% of training data |
| Random seed | 42 |

## Results

- Test accuracy of roughly **98%+** (the exact value varies slightly between runs).
- Training and validation curves stay close together, which indicates limited overfitting.
- Remaining errors are mostly between visually similar digits (e.g. 4 ↔ 9, 3 ↔ 5, 7 ↔ 2), and many misclassified images are genuinely ambiguous.


## Project Structure

```
.
├── Handwritten_Digit_Classifier__MNIST_.ipynb   # Full notebook: data, model, training, evaluation
└── README.md
```

## Possible Improvements

- Replace the dense network with a **Convolutional Neural Network (CNN)**
- Add **data augmentation** (small rotations, shifts, zooms)
- Use **early stopping** and **learning-rate scheduling**
- Tune dropout rates and layer sizes

## Tech Stack

Python · TensorFlow / Keras · NumPy · Matplotlib · scikit-learn
