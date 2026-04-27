# Mixed Wafer Defect Detection

[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project for **multi-label classification** of semiconductor wafer defects using a fine-tuned **ResNet-18** with **Grad-CAM** visual explainability. Trained on the [MixedWM38 dataset](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets).

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Explainability](#model-explainability)
- [Key Concepts](#key-concepts)
- [Future Work](#future-work)
- [References](#references)
- [Hardware](#hardware)

---

## Overview

Semiconductor wafer maps encode spatial patterns of passing and failing dies. This project treats defect detection as a **multi-label classification problem**: any wafer can exhibit multiple simultaneous defect types. 

A pretrained **ResNet-18** is adapted to detect 8 defect categories, and **Grad-CAM** is used to generate visual heatmaps showing exactly which regions of a wafer influenced each prediction — critical for trust and validation in manufacturing environments.

### The 8 Defect Classes

| Defect | Description |
|--------|-------------|
| **Center** | Cluster of failures in the middle of the wafer |
| **Donut** | Ring of failures around the center |
| **Edge-Loc** | Failures localized to one section of the edge |
| **Edge-Ring** | Failures forming a complete ring around the edge |
| **Loc** | Localized cluster anywhere on the wafer |
| **Near-full** | Almost the entire wafer failing |
| **Scratch** | Linear streak of failures |
| **Random** | Scattered failures with no spatial pattern |

---

## Dataset

- **Source:** [MixedWM38 on Kaggle](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets)
- **Format:** `.npz` (compressed NumPy archive)
- **Size:** 38,015 wafer maps (52×52 pixels each)
- **Labels:** 8-dimensional binary vectors (multi-label)
- **Pixel values:** `0` = background, `1` = passing die, `2` = failing die

The dataset is highly imbalanced. For example, `Near-full` appears in only ~149 wafers, while `Loc` and `Random` appear in ~18,000–19,000.

---

## Approach

### Model Architecture

- **Backbone:** ResNet-18 (pretrained on ImageNet)
- **Output layer:** Fully connected layer with 8 outputs (one per defect)
- **Input preprocessing:**
  - Normalize pixel values from `[0, 1, 2]` to `[0.0, 0.5, 1.0]`
  - Upsample from 52×52 to 224×224 (ResNet's expected input size)
  - Repeat single channel to 3 channels for ImageNet compatibility

### Two-Phase Training Strategy

1. **Phase 1 — Transfer Learning (Frozen Backbone):**
   - Freeze all pretrained layers
   - Train only the final fully connected layer (~4,104 parameters)
   - Learning rate: `1e-3`
   - Goal: Establish a strong baseline without overfitting

2. **Phase 2 — Fine-Tuning:**
   - Unfreeze `layer3`, `layer4`, and the final FC layer (~10.5M parameters)
   - Lower learning rate: `1e-4` to preserve pretrained features
   - Goal: Adapt deep feature extractors to wafer-specific patterns

### Loss Function

- **`BCEWithLogitsLoss`** with per-class `pos_weight`
- `pos_weight` = (number of negatives) / (number of positives)
- This heavily penalizes missing rare defects (e.g., Near-Full weight ≈ 278)

### Optimizer & Scheduler

- **Optimizer:** Adam
- **Scheduler:** `ReduceLROnPlateau` (halves LR after 3 epochs of no validation improvement)

---

## Results

Performance on the held-out test set (20% split, **7,603 samples**):

| Class     | Precision | Recall | F1   | Support |
|-----------|-----------|--------|------|---------|
| Center    | 1.00      | 1.00   | 1.00 | ~3,500  |
| Donut     | 1.00      | 1.00   | 1.00 | ~500    |
| Edge-Loc  | 1.00      | 0.97   | 0.98 | ~3,800  |
| Edge-Ring | 0.98      | 1.00   | 0.99 | ~2,500  |
| Loc       | 1.00      | 0.98   | 0.99 | ~7,000  |
| Near-full | 0.27      | 1.00   | 0.43 | ~40     |
| Scratch   | 0.82      | 1.00   | 0.90 | ~177    |
| Random    | 1.00      | 0.97   | 0.98 | ~4,300  |
| **Weighted Avg** | **0.99** | **0.98** | **0.99** | — |

**Key Observations:**
- Most classes achieve **98–100% F1**.
- **Near-full** suffers from extremely low precision due to severe class imbalance, but achieves **100% recall** — no real Near-full defects are missed.
- **Scratch** also shows slightly lower precision but perfect recall.

---

## Project Structure

```
mixed-wafer-detection/
├── main.ipynb              # Main notebook: EDA → training → evaluation → Grad-CAM
├── dataset.npz             # MixedWM38 dataset (not tracked in git)
├── best_model.pth          # Best model checkpoint (lowest validation loss)
├── weights.pth             # Final trained weights
├── pyproject.toml          # Project metadata
├── README.md               # This file
├── STUDY_GUIDE.md          # Detailed cell-by-cell breakdown for study/review
├── sources.md              # Links to dataset, papers, and references
├── .python-version         # Python 3.14
├── .gitignore
├── img/                    # Generated visualizations
│   ├── 8defecttypes.png
│   ├── confusion_matrices.png
│   ├── defectoccurence.png
│   ├── gradcam_misclassified_*.png
│   ├── gradcam_mixed_defect_*.png
│   ├── gradcam_single_defect_*.png
│   ├── testimage.png
│   └── training_curves.png
└── articles/               # Relevant papers (local copies)
    └── ...
```

---

## Setup & Installation

### Prerequisites

- Python 3.14+
- CUDA-capable GPU (recommended; tested on RTX 4080M 12 GB)
- [uv](https://github.com/astral-sh/uv) or `pip`

### 1. Clone the Repository

```bash
git clone <repo-url>
cd mixed-wafer-detection
```

### 2. Install Dependencies

Using `uv` (recommended):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install torch torchvision numpy matplotlib seaborn scikit-learn pytorch-grad-cam
```

Using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy matplotlib seaborn scikit-learn pytorch-grad-cam
```

### 3. Download the Dataset

Download `dataset.npz` from [Kaggle](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets) and place it in the project root.

---

## Usage

Open and run `main.ipynb` sequentially in Jupyter Notebook or VS Code:

```bash
jupyter notebook main.ipynb
```

The notebook is organized into the following sections:

1. **Data Loading & EDA** — Load `.npz`, inspect shapes, visualize sample wafers
2. **Defect Visualization** — Display pure examples of all 8 defect types
3. **Class Distribution** — Analyze label imbalance
4. **Train/Test Split** — 80/20 split with `random_state=1337`
5. **Dataset & DataLoader** — PyTorch `Dataset` class with preprocessing
6. **Model Setup** — Load pretrained ResNet-18, define loss with `pos_weight`
7. **Phase 1 Training** — Train final FC layer only
8. **Phase 2 Training** — Fine-tune deeper layers
9. **Training Curves** — Plot loss over epochs
10. **Evaluation** — Classification report, precision/recall/F1 per class
11. **Grad-CAM Setup** — Configure `pytorch-grad-cam` on `layer4[-1]`
12. **Visual Explanations** — Generate heatmaps for single, mixed, and misclassified examples
13. **Confusion Matrices** — Per-class 2×2 confusion matrices
14. **Grad-CAM Gallery** — Save categorized Grad-CAM overlays to `img/`

---

## Model Explainability

**Grad-CAM** (Gradient-weighted Class Activation Mapping) produces heatmaps that highlight which spatial regions of a wafer map most influenced the model's prediction for each defect class.

**Why this matters:**
- Manufacturing engineers can **verify** that the model's reasoning aligns with domain knowledge.
- **Failure analysis** becomes visual — e.g., when the model misses a `Loc` defect, Grad-CAM reveals it overlapped spatially with a `Center` defect, making separation difficult.
- Builds **trust** in AI-driven quality control systems.

Examples are saved in `img/gradcam_*.png`.

---

## Key Concepts

### Multi-Label vs. Multi-Class

| | Multi-Class | Multi-Label |
|---|---|---|
| **Output** | Softmax (probabilities sum to 1) | Sigmoid (each output independent, 0–1) |
| **Loss** | `CrossEntropyLoss` | `BCEWithLogitsLoss` |
| **Use case** | One label per sample | Multiple labels per sample |

Wafer defects are multi-label: a single wafer can simultaneously have `Center` + `Scratch` + `Loc`.

### Transfer Learning

Early CNN layers learn generic visual features (edges, textures, gradients) that transfer across domains. By starting with an ImageNet-pretrained ResNet-18, we leverage millions of pre-learned features and only adapt the later layers for wafer-specific patterns.

### Class Imbalance Handling

Using `pos_weight` in `BCEWithLogitsLoss` ensures that missing a rare defect (e.g., `Near-full`) is penalized much more heavily than missing a common one. This biases the model toward high recall for rare classes — the correct tradeoff in manufacturing, where a false alarm is cheap but a missed catastrophic defect is expensive.

---

## Future Work

- [ ] **Data Augmentation** — Add rotations/flips (defect patterns are rotation-invariant)
- [ ] **Address Near-Full Precision** — Cap `pos_weight` or experiment with Focal Loss
- [ ] **Additional Explainability** — Integrate SHAP for comparison with Grad-CAM
- [ ] **Cross-Dataset Validation** — Evaluate generalization on WM-811K
- [ ] **Production Pipeline** — Add inference script, threshold tuning, and monitoring
- [ ] **Alternative Architectures** — Benchmark against Tiny Vision Transformers (ViT)

---

## References

- [MixedWM38 Dataset](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets)
- [Mixed-Type Wafer Defect Detection (IEEE)](https://ieeexplore.ieee.org/document/9184890)
- [Semiconductor Wafer Map Defect Classification with Tiny Vision Transformers](https://arxiv.org/pdf/2504.02494)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391)
- [Implementing Grad-CAM in PyTorch (Medium)](https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82)
- [ResNet-18 Grad-CAM Example Repo](https://github.com/ZizZu94/resnet-grad-cam)

---

## Hardware

Trained locally on:
- **GPU:** NVIDIA RTX 4080M (12 GB VRAM)
- **RAM:** 64 GB DDR5

Training time: ~10–15 minutes for both phases combined.

---

## License

MIT License — feel free to use, modify, and distribute.

---

*Built as a weekend project by Darrell Cheng.*
