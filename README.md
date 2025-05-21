# 🩺 Pediatric Pneumonia Detection using CNN & Transfer Learning

This project focuses on detecting **pediatric pneumonia** from chest X-ray images using **deep learning**. The model leverages **transfer learning** with the **Xception architecture**, followed by custom classification layers, to achieve a test accuracy of over **95%**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-95.7%25-brightgreen)

---

## 🔍 Model Overview
- **Base Model**: Xception (pretrained on ImageNet)
- **Custom Head**:
  - GlobalAveragePooling
  - Dense (128 → 64 → 32 → 8 → 1)
  - Layer Normalization & Dropout after each dense layer
- **Test Accuracy**: **95.7%**
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

---

## 📂 Dataset
- **Source**: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: `NORMAL`, `PNEUMONIA`
- Images are grayscale chest X-rays of pediatric patients.

---

## 📊 Evaluation Metrics

| Metric        | Score          |
|---------------|----------------|
| **Test Accuracy** | 95.7%          |
| **Test Loss**     | 0.253          |
| **Precision**     | 99.08%         |
| **Recall**        | 95.28%         |
| **F1 Score**      | 97.14%         |

---

## 🧠 Model Architecture Summary

```text
Base Model: Xception (Output: 7×7×2048) — 20,861,480 parameters

Custom Layers:
→ GlobalAveragePooling2D
→ Dense (128 units) + LayerNormalization + Dropout
→ Dense (64 units) + LayerNormalization + Dropout
→ Dense (32 units) + LayerNormalization + Dropout
→ Dense (8 units) + LayerNormalization + Dropout
→ Dense (1 unit, Sigmoid output)

