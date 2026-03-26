# 🚦 Road Sign Classification — Image Processing & Pattern Recognition

> **UTS 31256 Image Processing and Pattern Recognition** | Spring 2025  
> Team 41 — Assessment Task 2: Project Implementation

---

## 📌 Project Overview

This project explores and compares three approaches to **road sign classification** using a publicly available Kaggle dataset of ~9,000 images, augmented to ~25,000, alongside a supplementary 5,000-image Roboflow dataset for CNN fine-tuning.

The work is motivated by the real-world demands of autonomous vehicle navigation — accurately identifying road signs under varying lighting, occlusion, weather, and perspective conditions.

---

## 👥 Team Members

| Name | Student ID | Method |
|------|-----------|--------|
| Anush Harutyunyan | 24834362 | Transfer Learning (YOLO11n) |
| Minseok Lee | 25410085 | Transfer Learning (YOLO11n) |
| Fabiha Tabassum Priti | 14505578 | HOG/LBP + SVM |
| Jayden Tran | 14514614 | HOG/LBP + SVM |
| **Yaeesh Mahomed** | **24957692** | **Hybrid CNN + SVM** |
| Qian Zhao | 25777830 | Hybrid CNN + SVM |

---

## 🧠 Methods Implemented

### 1. Transfer Learning with Pretrained CNN (YOLO11n)
Leveraging the YOLO11n pretrained model with two-phase transfer learning:
- **Base training** on the Kaggle dataset (640×640, 20 epochs)
- **Fine-tuning** on the Roboflow dataset (416×416, layer-freezing strategy)

### 2. Colour-Shape Heuristic + HOG/LBP Features + SVM
A classical computer vision pipeline:
- Colour-shape heuristic segmentation for region of interest detection
- **HOG** (Histogram of Oriented Gradients) + **LBP** (Local Binary Patterns) feature extraction
- **SVM** classifier with GridSearchCV hyperparameter tuning (RBF kernel)
- Binary classification: sign vs. no-sign

### 3. Hybrid CNN + SVM *(Yaeesh Mahomed & Qian Zhao)*
A hybrid deep learning approach combining:
- **ResNet50** as a deep feature extractor (pretrained on ImageNet)
- **SVM** as the final classification layer (replacing the softmax head)
- **Test-Time Augmentation (TTA)** — 7 transformations per image (flips, rotations, brightness)
- StandardScaler feature normalisation + GridSearchCV SVM tuning
- Evaluation across 15 road sign classes

---

## 📊 Results Summary

| Metric | YOLO11n (CNN) | HOG/LBP + SVM | Hybrid CNN+SVM |
|---|---|---|---|
| Overall Accuracy | **0.872** | 0.800 | 0.803 |
| Top-3 Accuracy | **0.953** | — | 0.942 |
| Macro Precision | 0.881 | — | 0.802 |
| Macro Recall | 0.764 | — | 0.742 |
| Macro F1 | 0.815 | — | 0.757 |
| ROC-AUC | **0.989** | 0.444 | 0.980 |
| PR-AUC | **0.966** | 0.857 | 0.836 |
| Brier Score | **0.0155** | — | 0.0192 |
| ECE | **0.00736** | — | 0.117 |
| Latency (no TTA) | 12.25 ms/img | 25–40 ms/img | 19.02 ms/img |
| Throughput (no TTA) | **81.6 FPS** | 30–40 FPS | 52.6 FPS |
| Model Size | 5.19 MB | ~1.2 MB | — |

### Hybrid CNN+SVM Stress Testing (150-sample subset)
| Corruption | Accuracy |
|---|---|
| Clean baseline | 80.26% |
| Gaussian Noise | 77.73% |
| Blur | 79.33% |
| Brightness increase | 76.67% |
| Brightness decrease | 79.33% |
| Rotation +5° | 70.00% |
| Rotation −5° | 72.00% |

---

## 🗂️ Repository Structure

```
road-sign-classification-ippr/
├── README.md
├── hybrid_cnn_svm/
│   ├── main.py
│   └── requirements.txt
├── transfer_learning_yolo/
│   ├── main.py
│   └── requirements.txt
└── hog_lbp_svm/
│   ├── main.py
│   └── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
pip install -r hybrid_cnn_svm/requirements.txt
```

### Key dependencies (Hybrid CNN+SVM)
```bash
pip install torch torchvision scikit-learn scikit-image opencv-python numpy matplotlib
```

### Datasets
Download and place in the `datasets/` folder:
- [Kaggle Traffic Sign Detection Dataset](https://www.kaggle.com/datasets/icebearogo/traffic-sign-detection-dataset)
- [Roboflow Self-Driving Cars Dataset](https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou/dataset/6/images)

### Running the Hybrid CNN+SVM model
```bash
cd hybrid_cnn_svm
python main.py
```

---

## 🛠️ Tech Stack

`Python` `PyTorch` `ResNet50` `Scikit-learn` `OpenCV` `Scikit-image` `NumPy` `Matplotlib` `Albumentations` `YOLO11n` `Google Colab` `VS Code` `GitHub`

---

## 📄 License

This project was submitted as academic coursework at the University of Technology Sydney (UTS). Code is shared for portfolio and reference purposes only.
