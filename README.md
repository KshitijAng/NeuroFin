# NeuroFin: Intelligent Fish Classification using Transfer Learning

![Alt Text](Logo.png)


## 📌 Project Overview

**NeuroFin** is an image classification project that accurately identifies and classifies fish species across **9 distinct categories** using deep learning. Leveraging the power of **Transfer Learning** with **MobileNetV3-Large**, this project achieves an impressive **90% validation accuracy**.

The model is trained on a curated fish image dataset taken from Kaggle and built using **TensorFlow** and **Keras**, with techniques like data augmentation and fine-tuning to boost model performance.

---

## 🐟 Dataset

* The dataset includes images of 9 different fish species.
* Images had to be:
  * Organized into category-wise folders
  * Split into training and validation datasets using an 80:20 ratio
* Preprocessing included filtering zero-byte images and applying data augmentation to increase robustness.

### [Kaggle Dataset Link](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)

---

## Note on Transfer Learning and MobileNetV3

**Transfer Learning** is a technique in deep learning where a model trained on a large, general dataset is reused as the starting point for a related but different task. Instead of training a neural network from scratch — which requires massive data and computational power — transfer learning leverages the learned features of a pre-trained model. This approach significantly speeds up training and often improves performance, especially when your dataset is limited.

**MobileNetV3** is a state-of-the-art Convolutional Neural Network (CNN) architecture optimized for mobile and edge devices. It balances accuracy and efficiency by combining techniques like depthwise separable convolutions, squeeze-and-excitation modules, and lightweight attention mechanisms. MobileNetV3 models come in two main variants—Large and Small—designed for different trade-offs between latency and accuracy.

In this project, we use MobileNetV3-Large pre-trained on the ImageNet dataset (which contains1000 classes) as a base. By removing its classification head and adding custom dense layers, we fine-tune the model to classify fish species accurately while keeping training time and resource use manageable.

---

## 🧠 Model Architecture

* **Base Model:** MobileNetV3-Large (pretrained on ImageNet)
* **Custom Layers:**
  * Global Average Pooling
  * Dense Layers: 512 → 256 → 128 neurons (ReLU activation)
  * Final Layer: 9 neurons (Softmax activation for multi-class classification)
* **Training Configuration:**

  * Optimizer: Adam with learning rate = 0.0001
  * Loss: Categorical Crossentropy
  * Epochs: 5

---

## 📈 Performance

* **Validation Accuracy:** \~90%
* The model demonstrates reliable generalization across unseen validation data.

---

