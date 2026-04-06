# Mixed Wafer Defect Detection

A weekend project using Grad-CAM and a fine-tuned ResNet-18 to classify and visually explain semiconductor wafer defects. Trained on the [MixedWM38 dataset](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets).

## Overview

Semiconductor wafer maps encode spatial patterns of passing and failing dies. This project treats defect detection as a multi-label classification problem.  Any arbitrary wafer in the dataset can have multiple simultaneous defect types. The model uses Grad-CAM on a pretrained ResNet-18 model to highlight which regions of the wafer are defective and which defect they represent.

The 8 defect classes are: `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Scratch`, `Random` (These can also be found in the paper associated with the dataset)

## Approach

**Model:** ResNet-18 pretrained on ImageNet, adapted for 8-class multi-label output.

**Training strategy (two-phase):**
1. Freeze all backbone layers, train only the final FC layer (4,104 trainable params)
2. Unfreeze `layer3` and `layer4` for fine-tuning at a lower learning rate (10.5M trainable params)

**Loss:** `BCEWithLogitsLoss` with per-class `pos_weight` to handle class imbalance (e.g. `Near-full` appears in only ~40 of 7,603 test samples).

## Results

Weighted F1 on the held-out test set (20% split, 7,603 samples):

| Class     | Precision | Recall | F1   |
|-----------|-----------|--------|------|
| Center    | 1.00      | 1.00   | 1.00 |
| Donut     | 1.00      | 1.00   | 1.00 |
| Edge-Loc  | 1.00      | 0.97   | 0.98 |
| Edge-Ring | 0.98      | 1.00   | 0.99 |
| Loc       | 1.00      | 0.98   | 0.99 |
| Near-full | 0.27      | 1.00   | 0.43 |
| Scratch   | 0.82      | 1.00   | 0.90 |
| Random    | 1.00      | 0.97   | 0.98 |
| **weighted avg** | **0.99** | **0.98** | **0.99** |

`Near-full` and `Scratch` are the hardest classes due to severe class imbalance (40 and 177 samples respectively out of ~17,500 total labels).

## Setup

**Dataset:** Download `dataset.npz` from [Kaggle](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets) and place it in the project root.

Notebook is written in a Python 3.14.3 kernel. Dependencies are handled via UV and a virtual environment. 

## Usage

Open and run `main.ipynb` sequentially. The notebook covers:

1. Data loading and EDA
2. Dataset class + DataLoader setup
3. Model definition and phase-1 training
4. Fine-tuning (phase 2)
5. Evaluation + classification report
6. Grad-CAM visualization for single-defect, mixed-defect, and misclassified examples

Trained weights are saved to `weights.pth`.

## Hardware

Trained locally on an RTX 4080M (12 GB VRAM) + 64 GB DDR5.

## References

- [MixedWM38 Dataset](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets)
- [Semiconductor Wafer Map Defect Classification with Tiny Vision Transformers](https://arxiv.org/pdf/2504.02494)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391)
- [Mixed-Type Wafer Defect Detection (IEEE)](https://ieeexplore.ieee.org/document/9184890)
