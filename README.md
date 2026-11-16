# Automated Pancreatic Abnormality Detection and Segmentation in Abdominal CT Scans via Hybrid Deep and Handcrafted Feature Learning

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Modal](https://img.shields.io/badge/Cloud%20Compute-Modal-A10G%20%2F%20T4-000000.svg)](https://modal.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-ffcc00.svg)](https://huggingface.co/razaasherazi/pancreas-multiscale-unet-baseline)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

---

## 📌 Project Overview

This repository contains the complete implementation, cloud orchestration pipelines, evaluation suites, and empirical ablation studies for **Part 1** and **Part 2** of our Computer Vision (`AI4002`) research project: **Automated Pancreatic Abnormality Detection and Segmentation in Abdominal CT Scans**.

* **Part 1 (Baseline Replication & Loss/Contrast Ablation):** Replicates and systematically ablates the Multi-Scale U-Net framework proposed by **Qiu et al. (*Frontiers in Oncology*, 2024)** on the **Medical Segmentation Decathlon (MSD) Task07 Pancreas** dataset across four experimental regimes (Baseline, CLAHE + Augmentation, Pure Focal Tversky Loss, and Composite Hybrid Tversky–BCE Loss).
* **Part 2 (Hybrid Feature Learning — Handcrafted + Advanced Deep Architectures):** Investigates whether fusing dense, pixel-aligned classical computer vision feature maps (**LBP**, **Gabor Filter Bank**, **Dense Spatial HOG**, and **Multi-Scale LoG**) into two distinct advanced neural architectures (**Attention Residual U-Net** and **Hybrid CNN-Transformer `TransUNet-Lite`**) improves boundary delineation and surpasses the base paper benchmark (`0.7420` DSC) across six strictly controlled experimental runs.

> **Key Milestone:** **Run 3 (`Att-ResUNet + Multi-Orientation Gabor Bank` with SE Attention Fusion)** achieved **0.7707 Dice (DSC)** and **0.6314 IoU**, outperforming our Part 1 Baseline (`0.7113` DSC) by **+5.94%** and surpassing the Qiu *et al.* (2024) Q1 base paper (`0.7420` DSC) by **+2.87%**.

---

## 👥 Group & Academic Information

- **Institution:** FAST National University of Computer and Emerging Sciences (NUCES), Chiniot-Faisalabad Campus
- **Degree Program:** BS Artificial Intelligence
- **Course:** Computer Vision (`AI4002`)
- **Group Number:** 18
- **Team Members & Part 2 Contributions:**
  * **Syed Muhammad Raza** (`23F-0040`) — **Part 2 Model 1:** `AttentionResUNet` (`SEAttentionFusion`) | **Extractors:** Dense Local Binary Patterns (`LBP`), Multi-Orientation Gabor Filter Bank (`Gabor`)
  * **Muhammad Ahmad** (`23F-0022`) — **Part 2 Model 2:** `TransUNet-Lite` (`ProjectionConcatFusion`) | **Extractors:** Dense Spatial HOG (`HOG`), Multi-Scale Laplacian-of-Gaussian (`LoG`)

---

## 📄 Base Paper Reference

- **Title:** *Multi-scale U-Net with attention mechanism for pancreas segmentation in abdominal CT images*
- **Authors:** C. Qiu, Z. Liu, Y. Song, J. Yin, K. Han, Y. Zhu, Y. Liu, and V. S. Sheng
- **Year & Venue:** 2024 | *Frontiers in Oncology*, Vol. 14 (`DOI: 10.3389/fonc.2024.1348554`)
- **SJR Ranking:** **Q1** (Oncology / Cancer Research)
- **Reported Base Paper Benchmarks:** Dice (DSC) = **0.7420**, IoU = **0.5890**, Sensitivity = **0.8410**, Specificity = **0.9980**

---

## PART 1: Baseline Replication & Loss / Contrast Ablations

Part 1 investigated input-level adaptive contrast enhancement (CLAHE) and asymmetric region-based loss functions on a standard Multi-Scale U-Net across four sequential regimes:

1. **Experiment 1 — Baseline Replication (`baseline`):** Multi-Scale U-Net ($1\times1$, $3\times3$, $5\times5$ parallel branches), soft-tissue HU windowing ($[-100, +240]\text{ HU}$), symmetric composite loss $`\mathcal{L} = 0.5\,\mathcal{L}_{\text{Dice}} + 0.5\,\mathcal{L}_{\text{BCE}}`$, no augmentation $\rightarrow$ **0.7113 DSC / 0.5573 IoU**.
2. **Experiment 2 — Adaptive Contrast Enhancement via CLAHE (`exp1`):** Localized CLAHE ($`\alpha_{\text{clip}} = 2.0`$, $8\times8$ grid) + geometric flips/rotations ($\pm 10^\circ$) $\rightarrow$ Sensitivity jumped to **0.9331** (**+10.58%**), but symmetric BCE caused 1-pixel boundary dilation into retroperitoneal fat (**0.6731 DSC**).
3. **Experiment 3 — Standalone Pure Focal Tversky Loss (`exp2`):** Pure FTL ($`\alpha=0.6, \beta=0.4, \gamma=1.33`$) without BCE anchor $\rightarrow$ Optimization collapse (**0.4972 DSC**) due to vanishing regional scalar gradients under extreme foreground sparsity ($<1.5\%$ pancreas pixels).
4. **Experiment 4 — Composite Hybrid Tversky–BCE (`exp3`):** Anchored asymmetric Tversky penalty with point-wise BCE: $`\mathcal{L}_{\text{Hybrid}} = 0.5\,\mathcal{L}_{\text{BCE}} + 0.5\,\mathcal{L}_{\text{Tversky}}`$ ($`\alpha=0.7, \beta=0.3`$) + calibrated CLAHE ($`\alpha_{\text{clip}} = 1.2`$) $\rightarrow$ Stabilized recovery to **0.6680 DSC** and **0.9280 Sensitivity**.

### Part 1 Quantitative Summary (1,348 Validation Slices)

| Metric | Base Paper | Exp 1: Baseline (`baseline`) | Exp 2: CLAHE (`exp1`) | Exp 3: Pure FTL (`exp2`) | Exp 4: Hybrid Tversky–BCE (`exp3`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Loss Objective** | Hybrid Regional+BCE | Symmetric Dice+BCE | Symmetric Dice+BCE | Pure Focal Tversky | **Hybrid Tversky–BCE** |
| **Preprocessing** | Standard HU | $[-100, +240]$ HU | CLAHE ($`\alpha=2.0`$) | CLAHE ($`\alpha=1.2`$) | **CLAHE ($`\alpha=1.2`$)** |
| **Sensitivity (Recall)** | 0.8410 | 0.8273 | **0.9331** | 0.9081 | **0.9280** |
| **Specificity** | 0.9980 | **0.9976** | 0.9958 | 0.9873 | **0.9957** |
| **Dice (DSC)** | 0.7420 | **0.7113** | 0.6731 | 0.4972 | **0.6680** |
| **IoU (Jaccard Index)** | 0.5890 | **0.5573** | 0.5118 | 0.3570 | **0.5018** |
| **Final Validation Loss** | --- | 0.5102 | **0.4528** | 0.7111 | **0.5448** |

---

## PART 2: Hybrid Feature Learning (Handcrafted + Deep Architectures)

Building on Part 1's diagnostic insights (removing CLAHE boundary blurring while adding spatial attention and long-range context), Part 2 evaluates **6 strictly controlled runs** on **NVIDIA A10G GPUs** under locked hyperparameters (**30 epochs**, **Batch Size 32**, **AdamW `lr=1e-4`**, **`CosineAnnealingLR`**, **IEEE-754 FP32** with TF32 disabled, **`DiceBCELoss`**, deterministic seed `42`).

### 1. Advanced Deep Architectures & Handcrafted Feature Extractors

* **Student 1 (`AttentionResUNet` + `SEAttentionFusion`):**
  * **Architecture:** 5-level Residual U-Net with additive **Attention Gates** on all skip connections to suppress retroperitoneal fat activations.
  * **Handcrafted Extractors (GPU-Vectorized, Per-Slice Z-Score Normalized):**
    * **Dense LBP (1 channel):** 8-neighbor circular micro-texture pattern descriptor ($`\text{LBP}_{8,1}`$).
    * **Multi-Orientation Gabor Bank (4 channels):** Oriented sinusoidal-Gaussian filters at $`\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}`$ ($`\sigma=2.0, \lambda=5.0, \gamma=0.5`$).
  * **Fusion Mechanism:** **Squeeze-and-Excitation (SE) Channel Attention Fusion** at Encoder Stages 1 ($256\times256$) and 2 ($128\times128$), dynamically recalibrating channel weights between deep CNN features and projected handcrafted maps.

* **Student 2 (`TransUNet-Lite` + `ProjectionConcatFusion`):**
  * **Architecture:** 4-stage hierarchical CNN encoder coupled with a **3-layer Vision Transformer Bottleneck** (8 Multi-Head Self-Attention heads, GELU FFN, and learnable 2D spatial positional embeddings over $16\times16$ latent tokens) to capture global abdominal organ context.
  * **Handcrafted Extractors:**
    * **Dense Spatial HOG (4 channels):** 4-orientation Sobel gradient energy maps smoothed over $5\times5$ local neighborhoods and magnitude-normalized.
    * **Multi-Scale LoG (3 channels):** Lindberg scale-normalized Laplacian-of-Gaussian edge responses ($`\sigma^2 \nabla^2 G_\sigma * I`$) across fine ($`\sigma=1.0`$), medium ($`\sigma=2.0`$), and coarse ($`\sigma=4.0`$) scales.
  * **Fusion Mechanism:** **$1\times1$ Conv-BN-GELU Dimensionality Alignment + Channel Concatenation** followed by $3\times3$ spatial mixing at Encoder Stages 1 ($256\times256$) and 3 ($64\times64$).

### 2. Master 6-Run Ablation Results (1,348 Held-Out Validation Slices)

| Run / Benchmark | Student | Architecture | Handcrafted Feature | Best Ep. | Dice (DSC) | IoU (Jaccard) | Sensitivity | Specificity | Precision |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Paper (Qiu et al., 2024)** | *Ref* | Multi-Scale Att-UNet | --- | --- | 0.7420 | 0.5890 | 0.8410 | 0.9980 | --- |
| **Part 1 Baseline** | *Part 1* | Multi-Scale U-Net | None | 15 | 0.7113 | 0.5573 | 0.8273 | 0.9976 | --- |
| **Run 1 (`run1_att_resunet_base`)** | Student 1 | `Att-ResUNet` | None (Deep Baseline 1) | 22 | **0.7620** | **0.6216** | 0.8554 | 0.9981 | 0.6932 |
| **Run 2 (`run2_att_resunet_lbp`)** | Student 1 | `Att-ResUNet` | Dense LBP (1-ch) | 28 | **0.7637** | **0.6230** | 0.8531 | 0.9982 | 0.6963 |
| **Run 3 (`run3_att_resunet_gabor`)** | Student 1 | `Att-ResUNet` | **Gabor Bank (4-ch)** | **20** | **0.7707** | **0.6314** | 0.8430 | **0.9983** | **0.7152** |
| **Run 4 (`run4_transunet_base`)** | Student 2 | `TransUNet-Lite` | None (Deep Baseline 2) | 29 | 0.7081 | 0.5542 | 0.8841 | 0.9970 | 0.5945 |
| **Run 5 (`run5_transunet_hog`)** | Student 2 | `TransUNet-Lite` | **Dense HOG (4-ch)** | 27 | **0.7318** | **0.5817** | **0.8969** | 0.9973 | 0.6215 |
| **Run 6 (`run6_transunet_log`)** | Student 2 | `TransUNet-Lite` | Multi-Scale LoG (3-ch) | 29 | **0.7280** | **0.5768** | 0.8714 | 0.9975 | 0.6319 |

### 3. Key Empirical Findings

1. **Consistent Gains Across Both Architectures:** Fusing handcrafted descriptors improved every baseline metric in both model families—lifting `Att-ResUNet` from `0.7620` $\rightarrow$ **0.7707 DSC** (`+0.87%` with Gabor) and lifting `TransUNet-Lite` from `0.7081` $\rightarrow$ **0.7318 DSC** (`+2.37%` with Dense HOG).
2. **Restoring Spatial Inductive Bias in Transformers:** Trained from scratch on 5,390 medical slices without ImageNet pretraining, standalone `TransUNet-Lite` (`Run 4`) lacked local convolutional inductive bias (`0.7081` DSC, `0.5945` Precision). Injecting **Dense Spatial HOG** (`Run 5`) and **Multi-Scale LoG** (`Run 6`) restored explicit local contour gradients prior to Transformer tokenization, boosting Dice by **+2.37%** and achieving the highest clinical sensitivity across all runs (**0.8969**).
3. **Overall Group 18 Champion (`Run 3: Att-ResUNet + Gabor`):** Combining residual blocks, skip-connection Attention Gates, directional Gabor boundary filters, and SE channel recalibration yielded **0.7707 Dice**, **0.6314 IoU**, **0.9983 Specificity**, and **0.7152 Precision**, surpassing the base paper by **+2.87% DSC** and **+4.24% IoU**.

---

## 🖼️ Qualitative & Diagnostic Visualizations

### Part 2 Figures (`results/part2/`)
- **`results/part2/part2_ablation_bar_comparison.png`** — Unified bar chart comparing Dice, IoU, and Sensitivity across all 6 Part 2 runs against the Base Paper (`0.7420`) and Part 1 Baseline (`0.7113`).
- **`results/part2/part2_all_runs_convergence.png`** — 30-epoch validation Dice convergence curves for Student 1 (`Att-ResUNet`) and Student 2 (`TransUNet-Lite`).
- **`results/part2/part2_qualitative_overlays.png`** — Side-by-side CT segmentation overlays across all 6 models (green contour = ground truth; red/yellow mask = model prediction).
- **`results/part2/part2_master_summary.json`** — Complete JSON evaluation ledger including pixel-level confusion matrices (`TP`, `FP`, `TN`, `FN`) and precision/recall metrics.

### Part 1 Figures (`results/`)
- `results/baseline/` — Exp 1 loss curves, confusion matrix, and segmentation overlays.
- `results/exp1/` — Exp 2 CLAHE ablation curves and baseline comparison panel.
- `results/exp2/` — Exp 3 Pure FTL collapse diagnostics and tri-ablation comparison.
- `results/exp3/` — Exp 4 Hybrid Tversky–BCE recovery plots and four-way comparison panel.

---

## 📂 Project Directory Structure

```text
pancreas-ct-abnormality-detection/
├── src/
│   ├── dataset.py                    # In-memory cached PyTorch dataset loader
│   ├── preprocess.py                 # HU windowing, CLAHE filters, and affine augmentations
│   ├── handcrafted.py                # GPU-vectorized dense LBP, Gabor, HOG, and LoG extractors
│   ├── models/
│   │   ├── multiscale_unet.py        # Part 1: Multi-Scale U-Net baseline architecture
│   │   ├── att_resunet.py            # Part 2 (Student 1): Attention Residual U-Net + SE Fusion
│   │   └── transunet_lite.py         # Part 2 (Student 2): Hybrid CNN-Transformer + Concat Fusion
│   └── utils/
│       ├── loss.py                   # FP32-safe DiceBCELoss, FocalTverskyLoss, HybridTverskyBCELoss
│       └── metrics.py                # Vectorized DSC, IoU, Sensitivity, and Specificity metrics
├── notebooks/
│   └── Group18_CV_Project_Part2.ipynb # Unified executed Part 2 Jupyter Notebook with embedded charts
├── results/
│   ├── baseline/ ... exp3/           # Part 1 evaluation figures and JSON training logs
│   └── part2/                        # Part 2 6-run logs, curves, overlays, and master JSON summary
├── modal_app_baseline.py             # Part 1: Baseline Modal runner
├── modal_app_exp1.py                 # Part 1: CLAHE + Augmentation Modal runner
├── modal_app_exp2.py                 # Part 1: Pure Focal Tversky Modal runner
├── modal_app_exp3.py                 # Part 1: Hybrid Tversky-BCE Modal runner
├── modal_app_part2.py                # Part 2: Unified 6-run A10G FP32 Modal runner (auto-resume + HF sync)
├── evaluate_part2.py                 # Part 2: Unified 6-model test/val evaluation & figure generator
├── build_notebook.py                 # Reproducible generator script for Group18_CV_Project_Part2.ipynb
├── pyproject.toml                    # UV / Python environment dependencies
└── README.md                         # Complete Part 1 & Part 2 project documentation

```

---

## ⚙️ Environment Setup & Installation

This project uses `uv` for lightweight local orchestration while offloading all dataset storage and GPU training to **Modal Labs** and **Hugging Face Hub** (zero heavy local GPU footprint).

### 1. Clone the Repository & Initialize `uv`

```bash
git clone [https://github.com/razaasherazi/pancreas-ct-abnormality-detection.git](https://github.com/razaasherazi/pancreas-ct-abnormality-detection.git)
cd pancreas-ct-abnormality-detection

uv venv --python 3.11
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

uv pip install -e .

```

### 2. Configure Cloud Credentials (Modal & Hugging Face)

```bash
uv run modal setup
uv run modal secret create huggingface-secret HF_TOKEN="<YOUR_HF_TOKEN>"

```

---

## 🚀 Execution & Replication Workflow

### Part 2: Run All 6 Hybrid Ablation Experiments on Modal A10G GPUs

Each run executes in `--detach` mode on an isolated NVIDIA A10G container in strict IEEE-754 FP32 precision and automatically syncs checkpoints (`part2/run{id}_*_best.pth`), JSON logs, and curves to Hugging Face:

```bash
# Student 1 (Syed Muhammad Raza): Attention Residual U-Net + SE Attention Fusion
uv run modal run --detach modal_app_part2.py --run 1 --epochs 30 --batch-size 32
uv run modal run --detach modal_app_part2.py --run 2 --epochs 30 --batch-size 32
uv run modal run --detach modal_app_part2.py --run 3 --epochs 30 --batch-size 32

# Student 2 (Muhammad Ahmad): TransUNet-Lite + Projection Concatenation Fusion
uv run modal run --detach modal_app_part2.py --run 4 --epochs 30 --batch-size 32
uv run modal run --detach modal_app_part2.py --run 5 --epochs 30 --batch-size 32
uv run modal run --detach modal_app_part2.py --run 6 --epochs 30 --batch-size 32

```

### Part 2: Run Unified 6-Model Evaluation & Sync Artifacts Locally

```bash
# 1. Evaluate all 6 checkpoints and generate comparative figures + qualitative overlays
uv run modal run evaluate_part2.py

# 2. Download all Part 2 plots and JSON logs from Modal Volume to local results/part2/
uv run modal volume get pancreas-data-vol /results_part2 results/part2/ --force

```

### Part 1: Replicate Original Baseline & Loss Ablations

```bash
uv run modal run modal_app_baseline.py
uv run modal run modal_app_exp1.py
uv run modal run modal_app_exp2.py
uv run modal run --detach modal_app_exp3.py

```

---

## 📜 License & Acknowledgments

* **Codebase:** Distributed under the MIT License.
* **Dataset:** The Medical Segmentation Decathlon (MSD) Task07 Pancreas dataset is licensed under Creative Commons Attribution-ShareAlike 4.0 International (`CC BY-SA 4.0`).
* **Pretrained & Trained Checkpoints:** All Part 1 and Part 2 model weights (`.pth`) and evaluation logs are publicly archived on Hugging Face at [`razaasherazi/pancreas-multiscale-unet-baseline`](https://huggingface.co/razaasherazi/pancreas-multiscale-unet-baseline).
* **Base Paper:** We acknowledge Qiu *et al.* (2024) for their Q1 study on multi-scale attention architectures for pancreatic CT segmentation.
