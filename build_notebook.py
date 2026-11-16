import json
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return f"# File not found: {path}"

cells = [
    nbf.v4.new_markdown_cell(
        "# Computer Vision (AI4002) — Semester Project Part 2\n"
        "## Hybrid Feature Learning: Integration of Handcrafted and Deep Features for Pancreas CT Segmentation\n\n"
        "| Field | Details |\n"
        "| :--- | :--- |\n"
        "| **Group No.** | Group 18 |\n"
        "| **Student 1** | Syed Muhammad Raza (`23F-0040`) — **Model:** `AttentionResUNet` \\| **Extractors:** Dense LBP, Gabor Filter Bank |\n"
        "| **Student 2** | Muhammad Ahmad (`23F-0022`) — **Model:** `TransUNet-Lite` \\| **Extractors:** Dense Spatial HOG, Multi-Scale LoG |\n"
        "| **Base Paper** | Qiu *et al.* (2024), *Multi-scale U-Net with attention mechanism for pancreas segmentation*, Frontiers in Oncology (SJR Q1) |\n"
        "| **Dataset** | Medical Segmentation Decathlon (MSD) Task07 Pancreas (8,792 axial slices, 256×256, HU window [-100, +240]) |\n"
        "| **Cloud Infrastructure** | Modal Serverless NVIDIA A10G GPUs + Hugging Face Hub (`razaasherazi/pancreas-multiscale-unet-baseline`) |"
    ),
    nbf.v4.new_markdown_cell(
        "## 1. Environment & Experimental Consistency Protocol\n"
        "To ensure a strictly fair ablation across all 6 experimental runs, all training hyperparameters are locked:\n"
        "- **Epochs:** 30 | **Batch Size:** 32 | **Precision:** IEEE-754 FP32 (TF32 disabled)\n"
        "- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-5`) with `CosineAnnealingLR`\n"
        "- **Loss Function:** Composite `DiceBCELoss` (0.5 Dice + 0.5 FP32-clamped BCE)\n"
        "- **Augmentation:** Synchronous random horizontal/vertical flips and $\\pm 8^\\circ$ affine rotations"
    ),
    nbf.v4.new_markdown_cell(
        "## 2. Dense Spatial Handcrafted Feature Extractors (`src/handcrafted.py`)\n"
        "Implements four GPU-vectorized, per-slice Z-score normalized spatial extractors (`LBP`, `Gabor`, `Dense HOG`, `Multi-Scale LoG`) outputting $256 \\times 256$ pixel-aligned feature maps:\n\n"
        f"```python\n{read_file('src/handcrafted.py')}\n```"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Student 1 Architecture: Attention Residual U-Net + SE Attention Fusion (`src/models/att_resunet.py`)\n"
        "Integrates Residual Convolutional Blocks, Skip-Connection Attention Gates, and Squeeze-and-Excitation (SE) Channel Attention Fusion across Encoder Stages 1 and 2:\n\n"
        f"```python\n{read_file('src/models/att_resunet.py')}\n```"
    ),
    nbf.v4.new_markdown_cell(
        "## 4. Student 2 Architecture: Hybrid CNN-Transformer + Projection Concatenation Fusion (`src/models/transunet_lite.py`)\n"
        "Combines a 4-stage CNN encoder with a 3-layer Multi-Head Self-Attention (MHSA) Transformer Bottleneck ($16 \\times 16$ spatial tokens) and $1\\times1$ Projection + Channel Concatenation Fusion:\n\n"
        f"```python\n{read_file('src/models/transunet_lite.py')}\n```"
    ),
    nbf.v4.new_markdown_cell(
        "## 5. FP32-Safe Composite Loss & Modal Cloud Training Pipeline (`src/utils/loss.py` & `modal_app_part2.py`)\n\n"
        f"```python\n{read_file('src/utils/loss.py')}\n```\n\n"
        f"```python\n{read_file('modal_app_part2.py')}\n```"
    ),
    nbf.v4.new_markdown_cell(
        "## 6. Unified Cloud Evaluation Pipeline (`evaluate_part2.py`)\n\n"
        f"```python\n{read_file('evaluate_part2.py')}\n```"
    ),
    nbf.v4.new_markdown_cell(
        "## 7. Quantitative Results Across All 6 Runs\n\n"
        "| Model / Experiment | Student | Best Epoch | Dice (DSC) | IoU (Jaccard) | Sensitivity (Recall) | Specificity | Precision |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        "| Base Paper (Qiu et al., 2024) | Reference | — | 0.7420 | 0.5890 | 0.8410 | 0.9980 | — |\n"
        "| Part 1 Baseline (Multi-Scale UNet) | Part 1 | 15 | 0.7113 | 0.5573 | 0.8273 | 0.9976 | — |\n"
        "| **Run 1: Att-ResUNet (Base)** | Student 1 | 22 | **0.7620** | **0.6216** | 0.8554 | 0.9981 | 0.6932 |\n"
        "| **Run 2: Att-ResUNet + Dense LBP** | Student 1 | 28 | **0.7637** | **0.6230** | 0.8531 | 0.9982 | 0.6963 |\n"
        "| **Run 3: Att-ResUNet + Gabor Bank** | Student 1 | 20 | **0.7707** | **0.6314** | 0.8430 | **0.9983** | **0.7152** |\n"
        "| **Run 4: TransUNet-Lite (Base)** | Student 2 | 29 | 0.7081 | 0.5542 | 0.8841 | 0.9970 | 0.5945 |\n"
        "| **Run 5: TransUNet-Lite + Dense HOG** | Student 2 | 27 | **0.7318** | **0.5817** | **0.8969** | 0.9973 | 0.6215 |\n"
        "| **Run 6: TransUNet-Lite + Multi-Scale LoG** | Student 2 | 29 | **0.7280** | **0.5768** | 0.8714 | 0.9975 | 0.6319 |"
    ),
    nbf.v4.new_code_cell(
        "import json\n"
        "with open('../results/part2/part2_master_summary.json', 'r') as f:\n"
        "    master = json.load(f)\n\n"
        "print(f\"Evaluation Split: {master['split']} ({master['num_slices']} slices)\")\n"
        "print(f\"Base Paper Benchmark Dice: {master['base_paper_benchmark']['dice']:.4f}\")\n"
        "print(f\"Part 1 Baseline Dice:      {master['part1_baseline']['dice']:.4f}\\n\")\n"
        "for r in master['runs']:\n"
        "    print(f\"{r['label']:<30} | Dice: {r['dice']:.4f} | IoU: {r['iou']:.4f} | Sens: {r['sensitivity']:.4f} | Spec: {r['specificity']:.4f} | Prec: {r['precision']:.4f}\")"
    ),
    nbf.v4.new_markdown_cell(
        "## 8. Comparative Visualizations & Qualitative Segmentation Overlays"
    ),
    nbf.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "from PIL import Image\n\n"
        "fig, axes = plt.subplots(3, 1, figsize=(16, 22))\n"
        "imgs = [\n"
        "    ('../results/part2/part2_ablation_bar_comparison.png', 'Figure 1: 6-Run Quantitative Benchmark Comparison'),\n"
        "    ('../results/part2/part2_all_runs_convergence.png', 'Figure 2: Validation Dice Convergence Curves (Runs 1-6)'),\n"
        "    ('../results/part2/part2_qualitative_overlays.png', 'Figure 3: Qualitative CT Segmentation Overlays Across All 6 Models')\n"
        "]\n"
        "for ax, (path, title) in zip(axes, imgs):\n"
        "    ax.imshow(Image.open(path))\n"
        "    ax.set_title(title, fontsize=13, fontweight='bold')\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
]

nb["cells"] = cells
os.makedirs("notebooks", exist_ok=True)
out_nb = "notebooks/Group18_CV_Project_Part2.ipynb"
with open(out_nb, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {out_nb}")