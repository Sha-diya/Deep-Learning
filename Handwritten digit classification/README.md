# Handwritten Digit Classification

This project implements a Convolutional Neural Network (CNN) model to classify handwritten digits using the **MNIST** dataset. The model is built with **PyTorch**, trained to achieve high accuracy in recognizing digits (0–9).

---

## 📌 Project Overview

### 🎯 Objective
To develop a reliable CNN-based classifier for handwritten digit recognition using the MNIST dataset, following modern deep learning practices.

### ✨ Key Features
- **Data Loading & Preprocessing**  
  Utilizes PyTorch's `torchvision` to load the MNIST dataset, apply transformations (normalization, tensor conversion), and prepare training/validation splits.

- **Model Architecture (CNN)**  
  A multi-layer convolutional neural network featuring convolution, pooling, activation, and dense layers to capture spatial patterns effectively.

- **Training Pipeline**  
  - Uses cross-entropy loss and **Adam optimizer**.  
  - Tracks metrics such as training/validation loss and accuracy across epochs.  
  - Includes checkpoints for saving best-performing model weights.

- **Evaluation & Inference**  
  - Computes classification accuracy on the test set.  
  - Supports inference on arbitrary handwritten digit images.
