# Multi-Domain Plant Disease Detection: A YOLOv11x Benchmark on 116 Classes

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11x-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images/blob/main/YOLO(2).ipynb)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/sivtk/plant-disease-yolov11x-116classes)

</div>

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

### Split Distribution

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 16,582 | 70% |
| Validation | 4,737 | 20% |
| Test | 2,370 | 10% |
| **Total** | **23,689** | **100%** |

### Source Composition

| Source | Environment | Images | Classes | Percentage |
|--------|-------------|--------|---------|------------|
| plantwild | Laboratory | 18,533 | 89 | 78.2% |
| FieldPlant | Field | 5,156 | 27 | 21.8% |

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

Option 3: Load Pre-trained Model
python

from ultralytics import YOLO

# Download model from GitHub releases or Hugging Face
model = YOLO('best.pt')  # or 'sivtk/plant-disease-yolov11x-116classes'

# Run inference
results = model('path/to/leaf_image.jpg', conf=0.25)

# Display results
results[0].show()

# Get detections
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]
    print(f"Detected: {class_name} ({confidence:.2%})")

📁 Repository Structure
text

├── YOLO(2).ipynb              # Complete training notebook
├── README.md                   # This file
├── LICENSE                     # MIT License
├── best.pt                     # Pre-trained model weights (69.8% mAP50)
├── dataset/                    # Dataset folder (created by notebook)
│   ├── train/images/           # 16,582 training images
│   ├── train/labels/           # Training labels
│   ├── val/images/             # 4,737 validation images
│   ├── val/labels/             # Validation labels
│   ├── test/images/            # 2,370 test images
│   ├── test/labels/            # Test labels
│   └── data.yaml               # Dataset configuration
└── publication/               # Generated figures and tables
    ├── comparison_bar_chart.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── dataset_source_pie.png
    ├── dataset_split_bar.png
    ├── overall_performance.csv
    └── comparison_table.csv

🔄 Resume Training

The notebook automatically saves checkpoints. To resume interrupted training:
python

from ultralytics import YOLO

# Load the last checkpoint
model = YOLO('/content/drive/MyDrive/PlantDisease_TrainingBackup_*/training_run/weights/last.pt')

# Resume training
model.train(data='dataset/data.yaml', epochs=100, resume=True)

🧪 Test on Custom Images
Using the notebook:
python

from google.colab import files
from ultralytics import YOLO

model = YOLO('best.pt')
uploaded = files.upload()

for img_name, img_bytes in uploaded.items():
    results = model(img_name, conf=0.25)
    results[0].show()  # displays annotated image

Using command line:
bash

yolo predict model=best.pt source='path/to/image.jpg' conf=0.25

📝 Citation

If you use this model or dataset in your research, please cite:
bibtex

@software{kariuki_2026_plant_disease,
  author = {Kariuki, James Kariuki},
  title = {Multi-Domain Plant Disease Detection: A YOLOv11x Benchmark on 116 Classes},
  year = {2026},
  url = {https://github.com/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images},
  note = {mAP50: 69.8%, 116 disease classes, 23,689 images}
}

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    plantwild dataset (thuanai1/plantwild) for providing high-quality laboratory images

    FieldPlant dataset (manhhoangvan/fieldplant) for real-world field annotations

    Ultralytics team for the YOLOv11x implementation and training framework

    Google Colab for providing free A100 GPU access

    KaggleHub for easy dataset downloading

📞 Contact

Author: Kariuki James Kariuki
GitHub: SIV-TK
Email: sivtkariuki@gmail.com

For questions, issues, or collaboration requests, please open an issue on GitHub.
⭐ Star this Repository

If you find this work useful for your research or application, please consider starring the repository and citing it in your publications.
