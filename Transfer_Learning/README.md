# 🦜 Transfer Learning - Bird Image Classifiers

This project uses **Transfer Learning** to classify bird images using pre-trained convolutional neural networks (CNNs). Models are fine-tuned on custom datasets to distinguish between various bird species and between birds and squirrels.

---

## 🎯 Objectives

- Apply transfer learning with pre-trained CNN models (e.g., MobileNetV2, EfficientNet)
- Classify images of birds or differentiate birds vs squirrels
- Explore fine-tuning strategies using small datasets

---

## 📊 Main Packages Used

- `tensorflow`
- `keras`
- `matplotlib`, `numpy`
- `PIL`, `os`, `glob`

> 💡 GPU is recommended for better performance.

---

## 🧠 Notes

- Data augmentation is applied to improve generalization
- Early stopping and model checkpointing implemented
- Achieved good accuracy with fewer training images due to transfer learning

---

## 📁 Folder Structure

```bash
Transfer_Learning/
├── buildAndTrainBirder.ipynb            # Classifies among different bird species
├── buildAndTrainBirdsVsSquirrels.ipynb  # Binary classifier: birds vs squirrels
└── README.md                            # This file
