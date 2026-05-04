# Multi-Domain Plant Disease Detection: A YOLOv11x Benchmark on 116 Classes

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11x-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images/tree/main/YOLO.ipynb)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/JK-TK/PlantDiseaseDetection)

</div>

---

## Overview

This repository contains a complete implementation of a state-of-the-art plant disease detection system using **YOLOv11x** trained on **23,689 images** across **116 disease classes** covering major food crops. The model combines two complementary datasets — controlled laboratory images and real-world field images — to achieve strong generalization across domains.

- **plantwild** — 18,533 lab images, 89 classes (controlled environment, uniform backgrounds)
- **FieldPlant** — 5,156 field images, 27 classes (real field conditions, pathologist-annotated)

The model achieves **69.8% mAP50**, substantially outperforming existing benchmarks on field data (~38–48%) while handling 4× more classes.

---

## Model Performance

| Metric | Value |
|---|---|
| mAP50 | 69.8% |
| mAP50-95 | 68.4% |
| Recall | 69.5% |
| Precision | 62.2% |
| Total Classes | 116 |
| Total Images | 23,689 |
| Architecture | YOLOv11x (57M parameters) |

### Top Performing Classes

| Rank | Class | mAP50 |
|---|---|---|
| 1 | Corn Stripe | 99.5% |
| 2 | Corn Leaf Blight | 98.8% |
| 3 | Tomato Brown Spots | 97.5% |
| 4 | Raspberry Leaf | 96.1% |
| 5 | Cassava Mosaic | 95.5% |
| 6 | Celery Leaf | 95.4% |
| 7 | Peach Leaf Curl | 95.3% |
| 8 | Corn Smut | 93.9% |
| 9 | Cassava Healthy | 92.6% |
| 10 | Basil Leaf | 92.4% |

### Comparison with Published Benchmarks

| Study | Dataset | Classes | Architecture | mAP50 |
|---|---|---|---|---|
| Moupojou et al. (2023) | FieldPlant | 27 | Faster R-CNN | ~38% |
| PlantDoc Benchmark | PlantDoc | 27 | YOLOv5 | 43–48% |
| PlantVillage Typical | PlantVillage | 38 | YOLOv5/v8 | 85–90% |
| **Our Model** | **Combined** | **116** | **YOLOv11x** | **69.8%** |

---

## Dataset

| Split | Images | Percentage |
|---|---|---|
| Training | 16,582 | 70% |
| Validation | 4,737 | 20% |
| Test | 2,370 | 10% |

**Source Composition:**

| Source | Type | Images | Classes | Share |
|---|---|---|---|---|
| plantwild | Laboratory | 18,533 | 89 | 78.2% |
| FieldPlant | Field | 5,156 | 27 | 21.8% |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Architecture | YOLOv11x (57M parameters) |
| Input Size | 640 × 640 pixels |
| Batch Size | 64 (A100) / 8 (T4) |
| Optimizer | AdamW |
| Initial Learning Rate | 0.001 |
| Final Learning Rate | 0.00001 (cosine decay) |
| Weight Decay | 0.0005 |
| Momentum | 0.937 |
| Warmup Epochs | 3 |
| Total Epochs | 100 (early stopping at epoch 49) |
| Mixed Precision | Enabled |
| Augmentations | Mosaic (100%), MixUp (20%), Copy-Paste (20%), Rotation (±10°), Translation (±10°), Scaling (±50%), Flip (50%), HSV adjust |

---

## Export & Inference

### Model Export Sizes

| Format | Size | mAP50 (Validation) |
|---|---|---|
| PyTorch (FP32) | 457 MB | 69.8% |
| TFLite (FP16) | 114 MB | 69.5% |
| TFLite (INT8) | 62 MB | 68.2% |
| ONNX | 113 MB | 69.8% |

### Inference Speed

| Hardware | Precision | Time per Image (ms) | FPS |
|---|---|---|---|
| NVIDIA A100 | FP16 | 4.4 | 227 |
| NVIDIA T4 | FP16 | 15.2 | 66 |
| iPhone 14 | FP16 (Core ML) | ~120 | 8 |
| Android Flagship | INT8 (TFLite) | ~180 | 5 |

---

## Quick Start

### Option 1: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images/tree/main/YOLO.ipynb)

1. Click the badge above.
2. Select **Runtime → Change runtime type → A100 GPU** (or T4 if unavailable).
3. Run cells sequentially from top to bottom.

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images.git
cd A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images

# Install dependencies
pip install ultralytics opencv-python matplotlib seaborn scikit-learn pandas pyyaml tqdm kagglehub

# Run the training notebook
jupyter notebook "YOLO(2).ipynb"
```

### Option 3: Load Pre-trained Model

```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')  # or load from https://huggingface.co/JK-TK/PlantDiseaseDetection

# Run inference
results = model('path/to/leaf_image.jpg', conf=0.25)

# Display results
results[0].show()

# Print detections
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]
    print(f"Detected: {class_name} ({confidence:.2%})")
```

---

## Resume Training

The notebook automatically saves checkpoints. To resume an interrupted training run:

```python
from ultralytics import YOLO

model = YOLO('/content/drive/MyDrive/PlantDisease_TrainingBackup_*/training_run/weights/last.pt')
model.train(data='dataset/data.yaml', epochs=100, resume=True)
```

---

## Test on Custom Images

**Using Python:**

```python
from google.colab import files
from ultralytics import YOLO

model = YOLO('best.pt')
uploaded = files.upload()

for img_name in uploaded:
    results = model(img_name, conf=0.25)
    results[0].show()
```

**Using the command line:**

```bash
yolo predict model=best.pt source='path/to/image.jpg' conf=0.25
```

---

## Repository Structure

```
├── YOLO(2).ipynb                  # Complete training notebook
├── README.md                      # This file
├── LICENSE                        # MIT License
├── best.pt                        # Pre-trained weights (69.8% mAP50)
├── dataset/
│   ├── train/images/              # 16,582 training images
│   ├── train/labels/
│   ├── val/images/                # 4,737 validation images
│   ├── val/labels/
│   ├── test/images/               # 2,370 test images
│   ├── test/labels/
│   └── data.yaml                  # Dataset configuration
└── publication/
    ├── comparison_bar_chart.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── dataset_source_pie.png
    ├── dataset_split_bar.png
    ├── overall_performance.csv
    └── comparison_table.csv
```

---

## Key Features

- 116 disease classes across major food crops (corn, tomato, cassava, apple, grape, potato, banana, citrus, and more)
- Combined-domain training (laboratory + field) for real-world generalization
- Complete pipeline from dataset download to model export
- Per-class mAP50, confusion matrix, and comprehensive test evaluation
- TensorFlow Lite export for mobile deployment (FP16 and INT8)
- Publication-ready visualizations generated automatically
- Resume training with checkpoint saving
- Fully reproducible with fixed random seeds

---

## Citation

If you use this model or dataset in your research, please cite:

```bibtex
@software{kariuki_2026_plant_disease,
  author = {Kariuki, James Kariuki},
  title  = {Multi-Domain Plant Disease Detection: A YOLOv11x Benchmark on 116 Classes},
  year   = {2026},
  url    = {https://github.com/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images/tree/main/YOLO.ipynb},
  note   = {mAP50: 69.8\%, 116 disease classes, 23,689 images}
}
```

---

## Acknowledgments

- [plantwild](https://www.kaggle.com/datasets/thuanai1/plantwild) — high-quality laboratory images
- [FieldPlant](https://www.kaggle.com/datasets/manhhoangvan/fieldplant) — real-world field annotations
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv11x implementation and training framework
- Google Colab — free A100 GPU access
- KaggleHub — easy dataset downloading

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

**Author:** Kariuki James Kariuki  
**GitHub:** [SIV-TK](https://github.com/SIV-TK)  
**Hugging Face:** [JK-TK/PlantDiseaseDetection](https://huggingface.co/JK-TK/PlantDiseaseDetection)  
**Paper / Notebook:** [YOLO.ipynb](https://github.com/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images/tree/main/YOLO.ipynb)  
**Email:** ST62550392025@students.ouk.ac.ke  
**Phone:** +254 718 845 849  

For questions, issues, or collaboration requests, please [open an issue](https://github.com/SIV-TK/A-YOLOv11x-Benchmark-on-116-Classes-Using-Combined-Laboratory-and-Field-Images/issues) on GitHub.

---

*If you find this work useful, please consider starring the repository and citing it in your publications.*
