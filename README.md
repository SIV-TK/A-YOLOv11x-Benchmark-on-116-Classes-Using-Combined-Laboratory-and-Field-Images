# Multi-Domain Plant Disease Detection: A YOLOv11x Benchmark on 116 Classes

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11x-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images/blob/main/YOLO(2).ipynb)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/sivtk/plant-disease-yolov11x-116classes)

</div>

---

## 📊 Model Performance Summary

| Metric | Value |
|--------|-------|
| **mAP50** | **69.8%** |
| **mAP50-95** | 68.4% |
| **Recall** | 69.5% |
| **Precision** | 62.2% |
| **Total Classes** | 116 |
| **Total Images** | 23,689 |
| **Architecture** | YOLOv11x (57M parameters) |

---

## 📋 Overview

This repository contains a complete implementation of a state-of-the-art plant disease detection system using **YOLOv11x** trained on **23,689 images** covering **116 disease classes** across major food crops.

The model combines two complementary datasets:
- **plantwild** (18,533 lab images, 89 classes) - Controlled environment, uniform backgrounds
- **FieldPlant** (5,156 field images, 27 classes) - Real field conditions, pathologist-annotated

**The model achieves 69.8% mAP50**, substantially outperforming existing benchmarks on field data (~38-48%) despite handling 4× more classes.

---

## 🏆 Top Performing Classes

| Rank | Class | mAP50 |
|------|-------|-------|
| 1 | Corn Stripe | 99.5% |
| 2 | Corn leaf blight | 98.8% |
| 3 | Tomato Brown Spots | 97.5% |
| 4 | raspberry leaf | 96.1% |
| 5 | Cassava Mosaic | 95.5% |
| 6 | celery leaf | 95.4% |
| 7 | peach leaf curl | 95.3% |
| 8 | Corn Smut | 93.9% |
| 9 | Cassava Healthy | 92.6% |
| 10 | basil leaf | 92.4% |

---

## 🧠 Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv11x (57M parameters) |
| Input size | 640 × 640 pixels |
| Batch size | 64 (A100) / 8 (T4) |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Final learning rate | 0.00001 (cosine decay) |
| Weight decay | 0.0005 |
| Momentum | 0.937 |
| Warmup epochs | 3 |
| Total epochs | 100 (early stopping at 49) |
| Mixed precision | Enabled |
| Augmentations | Mosaic (100%), MixUp (20%), Copy-Paste (20%), Rotation (±10°), Translation (±10%), Scaling (±50%), Flip (50%), HSV adjust |

---

## 📈 Results

### Comparison with Published Benchmarks

| Study | Dataset | Classes | Architecture | mAP50 |
|-------|---------|---------|--------------|-------|
| Moupojou et al. (2023) | FieldPlant | 27 | Faster R-CNN | ~38% |
| PlantDoc benchmark | PlantDoc | 27 | YOLOv5 | 43-48% |
| PlantVillage typical | PlantVillage | 38 | YOLOv5/v8 | 85-90% |
| **Our model** | **Combined** | **116** | **YOLOv11x** | **69.8%** |

### Model Export Sizes

| Format | Size | mAP50 (validation) |
|--------|------|---------------------|
| PyTorch (FP32) | 457 MB | 69.8% |
| TFLite (FP16) | 114 MB | 69.5% |
| TFLite (INT8) | 62 MB | 68.2% |
| ONNX | 113 MB | 69.8% |

### Inference Speed

| Hardware | Precision | Time per image (ms) | FPS |
|----------|-----------|---------------------|-----|
| NVIDIA A100 | FP16 | 4.4 | 227 |
| NVIDIA T4 | FP16 | 15.2 | 66 |
| iPhone 14 | FP16 (Core ML) | ~120 | 8 |
| Android flagship | INT8 (TFLite) | ~180 | 5 |

---

## 🎯 Key Features

- 116 disease classes across major food crops (corn, tomato, cassava, apple, grape, potato, banana, citrus, and more)
- Combined domain training (laboratory + field) for real-world generalization
- Complete training pipeline from dataset download to model export
- Comprehensive evaluation with per-class mAP50, confusion matrix, and test results
- Export to TensorFlow Lite for mobile deployment (FP16 and INT8)
- Publication-ready visualizations automatically generated
- Resume training capability with checkpoint saving
- Fully reproducible with fixed random seeds

---

## 📊 Dataset Statistics

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 16,582 | 70% |
| Validation | 4,737 | 20% |
| Test | 2,370 | 10% |

**Source Composition:**
- **plantwild** (laboratory): 18,533 images, 89 classes (78.2%)
- **FieldPlant** (field): 5,156 images, 27 classes (21.8%)

---

## 🚀 Quick Start

### Option 1: Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images/blob/main/YOLO(2).ipynb)

1. Click the Colab badge above
2. Select **Runtime → Change runtime type → A100 GPU** (or T4 if A100 unavailable)
3. Run cells sequentially from top to bottom

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images.git
cd A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images

# Install dependencies
pip install ultralytics opencv-python matplotlib seaborn scikit-learn pandas pyyaml tqdm kagglehub

# Run the training notebook
jupyter notebook YOLO\(2\).ipynb
