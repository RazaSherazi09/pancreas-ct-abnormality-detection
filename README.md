# Automated Pancreatic Abnormality Detection and Segmentation in Abdominal CT Scans Using Multi-Scale U-Net

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Modal](https://img.shields.io/badge/Cloud%20Compute-Modal%20Labs-000000.svg)](https://modal.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-ffcc00.svg)](https://huggingface.co/razaasherazi/pancreas-multiscale-unet-baseline)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

---

## 📌 Project Overview

This repository contains the official implementation, evaluation protocols, and ablation studies for **Part 1** of our Computer Vision research project: **Automated Pancreatic Abnormality Detection and Segmentation in Abdominal CT Scans Using Multi-Scale U-Net**.

We replicate and systematically ablate the deep learning framework proposed by **Qiu et al. (Frontiers in Oncology, 2024)** on the **Medical Segmentation Decathlon (MSD) Task07 Pancreas** benchmark. The study investigates input-level adaptive contrast transformations (CLAHE), asymmetric loss formulations (Focal Tversky Loss), and composite objectives (Hybrid Tversky–BCE) to mitigate boundary over-segmentation and extreme class imbalance in abdominal CT volumes.

---

## 👥 Group & Academic Information

* **Institution:** FAST National University of Computer and Emerging Sciences (NUCES)
* **Degree Program:** BS Artificial Intelligence
* **Course:** Computer Vision
* **Group Number:** 18
* **Team Members:**
  * **Syed Muhammad Raza** (Roll No: `23F-0040`)
  * **Muhammad Ahmad** (Roll No: `23F-0022`)

---

## 📄 Base Paper Reference

* **Title:** *Multi-scale feature fusion network for accurate pancreas and tumor segmentation in CT images*
* **Authors:** X. Qiu, Y. Zhang, L. Wang, et al.
* **Year & Venue:** 2024 | *Frontiers in Oncology*
* **SJR Ranking:** **Q1** (Oncology / Radiology, Nuclear Medicine and Imaging)
* **Reported Base Metrics:** Dice Similarity Coefficient (DSC) $\approx \mathbf{74.20\%}$, IoU $\approx \mathbf{58.90\%}$, Sensitivity $\approx \mathbf{84.10\%}$, Specificity $\approx \mathbf{99.80\%}$

---

## 🔬 Experimental Methodology & Systematic Ablation Pipeline

Our study systematically documents four sequential experimental regimes:

1. **Experiment 1 — Baseline Replication (Strict Deterministic Setup):**
   * Multi-Scale U-Net featuring a dual-branch encoder with parallel $1\times1$, $3\times3$, and $5\times5$ convolutional kernels.
   * Standard soft-tissue Hounsfield Unit windowing ($[-100, +240]\text{ HU}$) with min-max normalization to $[0, 1]$.
   * Symmetric composite loss: $\mathcal{L} = 0.5\,\mathcal{L}_{\text{Dice}} + 0.5\,\mathcal{L}_{\text{BCE}}$.
   * No data augmentation.
   * **Outcome:** Achieved **71.13% DSC**, **55.73% IoU**, **82.73% Sensitivity**, and **99.76% Specificity**.

2. **Experiment 2 — Adaptive Contrast Enhancement via CLAHE + Augmentation (`exp1` in codebase):**
   * Applied localized Contrast-Limited Adaptive Histogram Equalization ($\alpha_{\text{clip}} = 2.0$, $8\times8$ grid).
   * Synchronized stochastic geometric augmentations: horizontal/vertical flips ($p=0.5$) and affine rotations ($\pm 10^\circ$).
   * **Outcome:** Sensitivity jumped to **93.31%** ($+10.58\%$ over baseline), but symmetric BCE caused boundary dilation into retroperitoneal fat (DSC dropped to $0.6731$).

3. **Experiment 3 — Standalone Asymmetric Loss: Pure Focal Tversky Loss (`exp2` in codebase):**
   * Implemented pure Focal Tversky Loss (FTL) with $\alpha=0.6$ (heavier penalty on False Positives), $\beta=0.4$, and $\gamma=1.33$, alongside calibrated CLAHE ($\alpha_{\text{clip}} = 1.2$).
   * **Outcome:** Severe optimization collapse (DSC dropped to **0.4972**, validation loss stalled at $0.7111$).
   * **Root Cause Analysis:** Without point-wise convex BCE gradients, regional scalar fractions vanish during early training under extreme class imbalance ($<1.5\%$ foreground pancreas), causing contour instability.

4. **Experiment 4 — Composite Hybrid Tversky–BCE Optimization (`exp3` in codebase):**
   * Formulated a composite objective combining point-wise BCE anchors with an asymmetric Tversky penalty:
     $$\mathcal{L}_{\text{Hybrid}} = 0.5\,\mathcal{L}_{\text{BCE}} + 0.5\,\mathcal{L}_{\text{Tversky}} \quad (\alpha=0.7, \beta=0.3)$$
   * Integrated calibrated CLAHE ($\alpha_{\text{clip}} = 1.2$) and regularized spatial augmentations ($\pm 8^\circ$).
   * **Outcome:** Immediate stabilization and recovery—validation DSC climbed to **0.6680** ($+17.08\%$ over pure FTL), maintaining a clinical sensitivity of **92.80%** and specificity of **99.57%**.

---

## 📊 Comprehensive Ablation Results

Evaluated on $1,348$ validation axial CT slices from the MSD Task07 Pancreas cohort:

| Metric | Base Paper | Exp 1: Baseline (`baseline`) | Exp 2: CLAHE (`exp1`) | Exp 3: Pure FTL (`exp2`) | Exp 4: Hybrid Tversky–BCE (`exp3`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Loss Objective** | Hybrid Regional+BCE | Symmetric Dice+BCE | Symmetric Dice+BCE | Pure Focal Tversky | **Hybrid Tversky–BCE** |
| **Preprocessing** | Standard HU | $[-100, +240]\text{ HU}$ | CLAHE ($\alpha=2.0$) | CLAHE ($\alpha=1.2$) | **CLAHE ($\alpha=1.2$)** |
| **Augmentation** | Spatial Affine | Deterministic (None) | Flips + Rot ($\pm10^\circ$) | Flips + Rot ($\pm8^\circ$) | **Flips + Rot ($\pm8^\circ$)** |
| **Sensitivity (Recall)** | 0.8410 | 0.8273 | **0.9331** | 0.9081 | **0.9280** |
| **Specificity** | 0.9980 | **0.9976** | 0.9958 | 0.9873 | **0.9957** |
| **Dice (DSC)** | 0.7420 | **0.7113** | 0.6731 | 0.4972 | **0.6680** |
| **IoU (Jaccard Index)** | 0.5890 | **0.5573** | 0.5118 | 0.3570 | **0.5018** |
| **Final Validation Loss** | --- | 0.5102 | **0.4528** | 0.7111 | **0.5448** |
| **Final Training Loss** | --- | 0.4780 | **0.4202** | 0.6577 | **0.5330** |

---

## 🖼️ Qualitative & Diagnostic Visualizations

All evaluation figures and comparative panels generated during the experiments are preserved in the `results/` directory:

| Experiment | Convergence Trajectories | Pixel-Level Confusion Matrix | Segmentation Artifacts |
| :--- | :---: | :---: | :---: |
| **Exp 1: Baseline** | `results/baseline/baseline_loss_curves.png` | `results/baseline/baseline_confusion_matrix.png` | `results/baseline/baseline_segmentation_overlays.png` |
| **Exp 2: CLAHE (`exp1`)** | `results/exp1/exp1_loss_curves.png` | `results/exp1/exp1_confusion_matrix.png` | `results/exp1/baseline_vs_exp1_comparison.png` |
| **Exp 3: Pure FTL (`exp2`)** | `results/exp2/exp2_loss_curves.png` | `results/exp2/exp2_confusion_matrix.png` | `results/exp2/complete_tri_ablation_comparison.png` |
| **Exp 4: Hybrid (`exp3`)** | `results/exp3/exp3_loss_curves.png` | `results/exp3/exp3_confusion_matrix.png` | `results/exp3/complete_four_way_comparison.png` |

---

## 📂 Project Directory Structure

```text
pancreas-ct-abnormality-detection/
├── src/
│   ├── dataset.py                # In-memory cached PyTorch dataset loader
│   ├── preprocess.py             # HU windowing, CLAHE filters, and affine augmentations
│   ├── models/
│   │   └── multiscale_unet.py    # Multi-Scale U-Net architecture with multi-resolution branches
│   └── utils/
│       ├── loss.py               # DiceBCELoss, FocalTverskyLoss, and HybridTverskyBCELoss
│       └── metrics.py            # Vectorized DSC, IoU, Sensitivity, and Specificity metrics
├── results/
│   ├── baseline/                 # Exp 1 (Baseline) checkpoints, logs, and plots
│   ├── exp1/                     # Exp 2 (CLAHE clip=2.0) logs and comparison plots
│   ├── exp2/                     # Exp 3 (Pure FTL) logs and confusion matrices
│   └── exp3/                     # Exp 4 (Hybrid Tversky-BCE) recovery figures
├── modal_app_baseline.py         # Modal serverless training entrypoint for Exp 1 (Baseline)
├── modal_app_exp1.py             # Modal training entrypoint for Exp 2 (CLAHE + Aug)
├── modal_app_exp2.py             # Modal training entrypoint for Exp 3 (Pure FTL)
├── modal_app_exp3.py             # Modal training entrypoint for Exp 4 (Hybrid Tversky-BCE)
├── evaluate_baseline.py          # Baseline evaluation and overlay generation
├── evaluate_exp2.py              # Tri-model visual comparison script
├── evaluate_exp3.py              # Four-way side-by-side qualitative generator
├── pyproject.toml                # UV / Python environment dependencies
└── README.md                     # Comprehensive project documentation
```

---

## ⚙️ Environment Setup & Installation

This project is managed using modern Python packaging via `uv`.

### 1. Clone the Repository

```bash
git clone [https://github.com/razaasherazi/pancreas-ct-abnormality-detection.git](https://github.com/razaasherazi/pancreas-ct-abnormality-detection.git)
cd pancreas-ct-abnormality-detection
```

### 2. Set Up Virtual Environment with `uv`

```bash
# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate virtual environment (Linux / macOS)
source .venv/bin/activate

# Sync dependencies
uv pip install -e .
```

### 3. Configure Cloud Infrastructure (Modal Labs)

Training is dispatched to cloud-hosted NVIDIA Tesla T4 GPUs via Modal:

```bash
uv run modal setup
uv run modal secret create huggingface-secret HF_TOKEN="<YOUR_HF_TOKEN>"
```

---

## 🚀 Execution & Replication Workflow

### Phase 1: Train Models on Cloud GPU

To train any of the experimental regimes, execute the corresponding runner script:

```bash
# Run Experiment 1 (Baseline Replication)
uv run modal run modal_app_baseline.py

# Run Experiment 2 (CLAHE clip=2.0 + High-Degree Augmentation)
uv run modal run modal_app_exp1.py

# Run Experiment 3 (Pure Focal Tversky Loss)
uv run modal run modal_app_exp2.py

# Run Experiment 4 (Composite Hybrid Tversky-BCE Loss - Detached mode recommended)
uv run modal run --detach modal_app_exp3.py
```

### Phase 2: Run Comparative Evaluations

```bash
# Generate tri-ablation comparisons
uv run modal run evaluate_exp2.py

# Generate full 4-way qualitative and quantitative comparison panels
uv run modal run evaluate_exp3.py
```

### Phase 3: Sync Artifacts Locally

Download evaluation figures and logs directly from Hugging Face:

```bash
python -c "from huggingface_hub import hf_hub_download; import shutil; [shutil.copy(hf_hub_download('razaasherazi/pancreas-multiscale-unet-baseline', f), f'results/exp3/{f}') for f in ['exp3_loss_curves.png', 'exp3_training_log.json', 'exp3_confusion_matrix.png', 'complete_four_way_comparison.png']]; print('Exp 3 assets synced successfully!')"
```

---

## 🔮 Proposed Advanced Direction (Project Part 2)

Based on the empirical findings from Part 1, we identified key limitations in standard 2D multi-scale convolution networks: lack of spatial attention in direct skip connections, absence of 3D inter-slice volumetric context, and fixed local receptive fields.

For Part 2, we will implement and benchmark two advanced architectures:
1. **Multi-Scale Attention Gate U-Net (Att-UNet):** Integrates spatial attention gates within decoder skip connections to suppress non-pancreatic activations in retroperitoneal fat and eliminate the boundary dilation observed with CLAHE.
2. **Hybrid Swin-Transformer Feature Network (Swin-UNet / UNETR 2.5D):** Incorporates shifted-window self-attention mechanisms to model long-range anatomical dependencies across consecutive slices and better segment small, indistinct pancreatic lesions.

---

## 📜 License & Acknowledgments

* **Codebase:** Distributed under the MIT License.
* **Dataset:** The Medical Segmentation Decathlon (MSD) Task07 Pancreas dataset is licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
* **Base Paper Authors:** We acknowledge Qiu et al. (2024) for their foundational work on multi-scale feature fusion for pancreatic CT analysis.