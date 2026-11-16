import os
import json
from pathlib import Path
import modal

HF_REPO_ID = "razaasherazi/pancreas-multiscale-unet-baseline"

app = modal.App("pancreas-part2-evaluator")
volume = modal.Volume.from_name("pancreas-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "tqdm",
        "huggingface_hub",
    )
    .add_local_dir(str(Path(__file__).parent / "src"), remote_path="/root/src")
)

RUN_SPECS = [
    {
        "run_id": 1,
        "label": "Run 1: Att-ResUNet (Base)",
        "short": "R1: Att-ResUNet",
        "student": "Student 1 (Syed Muhammad Raza)",
        "name": "run1_att_resunet_base",
        "arch": "att_resunet",
        "feature_mode": "none",
    },
    {
        "run_id": 2,
        "label": "Run 2: Att-ResUNet + LBP",
        "short": "R2: +LBP",
        "student": "Student 1 (Syed Muhammad Raza)",
        "name": "run2_att_resunet_lbp",
        "arch": "att_resunet",
        "feature_mode": "lbp",
    },
    {
        "run_id": 3,
        "label": "Run 3: Att-ResUNet + Gabor",
        "short": "R3: +Gabor",
        "student": "Student 1 (Syed Muhammad Raza)",
        "name": "run3_att_resunet_gabor",
        "arch": "att_resunet",
        "feature_mode": "gabor",
    },
    {
        "run_id": 4,
        "label": "Run 4: TransUNet-Lite (Base)",
        "short": "R4: TransUNet",
        "student": "Student 2 (Muhammad Ahmad)",
        "name": "run4_transunet_base",
        "arch": "transunet_lite",
        "feature_mode": "none",
    },
    {
        "run_id": 5,
        "label": "Run 5: TransUNet-Lite + HOG",
        "short": "R5: +HOG",
        "student": "Student 2 (Muhammad Ahmad)",
        "name": "run5_transunet_hog",
        "arch": "transunet_lite",
        "feature_mode": "hog",
    },
    {
        "run_id": 6,
        "label": "Run 6: TransUNet-Lite + LoG",
        "short": "R6: +LoG",
        "student": "Student 2 (Muhammad Ahmad)",
        "name": "run6_transunet_log",
        "arch": "transunet_lite",
        "feature_mode": "log",
    },
]


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    cpu=4,
    memory=16384,
)
def run_full_part2_evaluation(batch_size: int = 32):
    import sys
    sys.path.append("/root")

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from huggingface_hub import HfApi

    from src.dataset import CachedPancreasDataset
    from src.models.att_resunet import AttentionResUNet
    from src.models.transunet_lite import TransUNetLite
    from src.utils.metrics import compute_segmentation_metrics

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluator] Running unified 6-run evaluation on {device} ({torch.cuda.get_device_name(0)})", flush=True)

    data = torch.load("/vol/pancreas_cache.pt", weights_only=True)
    # Use test split if present in cache; otherwise fall back cleanly to held-out val split
    if "test_images" in data and len(data["test_images"]) > 0:
        eval_images = data["test_images"]
        eval_masks = data["test_masks"]
        split_name = "Test Set"
    else:
        eval_images = data["val_images"]
        eval_masks = data["val_masks"]
        split_name = "Validation Set"

    print(f"[Dataset] Evaluating on {split_name}: {len(eval_images)} slices", flush=True)
    eval_dataset = CachedPancreasDataset(eval_images, eval_masks, transform_fn=None)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    ckpt_dir = "/vol/checkpoints_part2"
    res_dir = "/vol/results_part2"
    os.makedirs(res_dir, exist_ok=True)

    summary_results = []
    loaded_models = {}
    histories = {}

    for spec in RUN_SPECS:
        name = spec["name"]
        ckpt_path = os.path.join(ckpt_dir, f"{name}_best.pth")
        log_path = os.path.join(res_dir, f"{name}_log.json")

        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                histories[name] = json.load(f)

        if spec["arch"] == "att_resunet":
            model = AttentionResUNet(in_channels=1, out_channels=1, feature_mode=spec["feature_mode"]).float().to(device)
        else:
            model = TransUNetLite(in_channels=1, out_channels=1, feature_mode=spec["feature_mode"]).float().to(device)

        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        loaded_models[name] = model

        metrics_acc = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0}
        tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0

        with torch.no_grad():
            for imgs, msks in tqdm(eval_loader, desc=f"Eval [{spec['short']}]"):
                imgs = imgs.float().to(device, non_blocking=True)
                msks = msks.float().to(device, non_blocking=True)

                preds = model(imgs)
                batch_m = compute_segmentation_metrics(preds, msks)
                for k in metrics_acc:
                    metrics_acc[k] += batch_m[k] * imgs.size(0)

                pred_bin = (preds > 0.5).long()
                targ_bin = (msks > 0.5).long()
                tp_total += int(((pred_bin == 1) & (targ_bin == 1)).sum().item())
                fp_total += int(((pred_bin == 1) & (targ_bin == 0)).sum().item())
                tn_total += int(((pred_bin == 0) & (targ_bin == 0)).sum().item())
                fn_total += int(((pred_bin == 0) & (targ_bin == 1)).sum().item())

        for k in metrics_acc:
            metrics_acc[k] /= len(eval_dataset)

        precision = tp_total / (tp_total + fp_total + 1e-8)
        params = sum(p.numel() for p in model.parameters())

        entry = {
            "run_id": spec["run_id"],
            "label": spec["label"],
            "short": spec["short"],
            "student": spec["student"],
            "arch": spec["arch"],
            "feature_mode": spec["feature_mode"],
            "params": params,
            "split": split_name,
            "dice": round(metrics_acc["dice"], 4),
            "iou": round(metrics_acc["iou"], 4),
            "sensitivity": round(metrics_acc["sensitivity"], 4),
            "specificity": round(metrics_acc["specificity"], 4),
            "precision": round(precision, 4),
            "confusion_matrix": {"TP": tp_total, "FP": fp_total, "TN": tn_total, "FN": fn_total},
        }
        summary_results.append(entry)
        print(
            f">>> {spec['label']} | Dice: {entry['dice']:.4f} | IoU: {entry['iou']:.4f} | "
            f"Sens: {entry['sensitivity']:.4f} | Spec: {entry['specificity']:.4f} | Prec: {entry['precision']:.4f}",
            flush=True,
        )

    # Save master JSON summary
    summary_payload = {
        "split": split_name,
        "num_slices": len(eval_dataset),
        "base_paper_benchmark": {"dice": 0.7420, "iou": 0.5890, "sensitivity": 0.8410, "specificity": 0.9980},
        "part1_baseline": {"dice": 0.7113, "iou": 0.5573, "sensitivity": 0.8273, "specificity": 0.9976},
        "runs": summary_results,
    }
    summary_json_path = os.path.join(res_dir, "part2_master_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_payload, f, indent=4)

    # --- Figure 1: Master 6-Run Bar Comparison vs Base Paper & Part 1 Baseline ---
    labels = [r["short"] for r in summary_results]
    dices = [r["dice"] for r in summary_results]
    ious = [r["iou"] for r in summary_results]
    sens = [r["sensitivity"] for r in summary_results]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - width, dices, width, label="Dice (DSC)", color="#1f77b4")
    b2 = ax.bar(x, ious, width, label="IoU (Jaccard)", color="#ff7f0e")
    b3 = ax.bar(x + width, sens, width, label="Sensitivity (Recall)", color="#2ca02c")

    ax.axhline(0.7420, color="red", linestyle="--", linewidth=1.4, label="Base Paper Dice (0.7420)")
    ax.axhline(0.7113, color="purple", linestyle=":", linewidth=1.4, label="Part 1 Baseline Dice (0.7113)")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_title(f"Group 18 Part 2 Ablation Comparison Across All 6 Runs ({split_name})", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    bar_fig_path = os.path.join(res_dir, "part2_ablation_bar_comparison.png")
    plt.savefig(bar_fig_path, dpi=200)
    plt.close()

    # --- Figure 2: Combined Validation Dice Convergence Across All 6 Runs ---
    if len(histories) == 6:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        colors_s1 = ["#1b9e77", "#d95f02", "#7570b3"]
        colors_s2 = ["#e7298a", "#66a61e", "#e6ab02"]

        for idx, spec in enumerate(RUN_SPECS[:3]):
            h = histories[spec["name"]]
            ax1.plot(range(1, len(h["val_dice"]) + 1), h["val_dice"], label=spec["label"], color=colors_s1[idx], linewidth=2)
        ax1.axhline(0.7420, color="red", linestyle="--", label="Base Paper (0.7420)")
        ax1.set_title("Student 1: Att-ResUNet + SE Fusion Convergence", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Validation Dice (DSC)")
        ax1.legend(fontsize=8.5)
        ax1.grid(True, alpha=0.4)

        for idx, spec in enumerate(RUN_SPECS[3:]):
            h = histories[spec["name"]]
            ax2.plot(range(1, len(h["val_dice"]) + 1), h["val_dice"], label=spec["label"], color=colors_s2[idx], linewidth=2)
        ax2.axhline(0.7420, color="red", linestyle="--", label="Base Paper (0.7420)")
        ax2.set_title("Student 2: TransUNet-Lite + Concat Fusion Convergence", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Dice (DSC)")
        ax2.legend(fontsize=8.5)
        ax2.grid(True, alpha=0.4)

        plt.tight_layout()
        conv_fig_path = os.path.join(res_dir, "part2_all_runs_convergence.png")
        plt.savefig(conv_fig_path, dpi=200)
        plt.close()
    else:
        conv_fig_path = None

    # --- Figure 3: 6-Model Qualitative Segmentation Overlay Grid ---
    # Select 4 informative slices with highest ground-truth pancreas area
    areas = [int((eval_masks[i] > 0).sum()) for i in range(len(eval_masks))]
    top_indices = np.argsort(areas)[-4:][::-1]

    fig, axes = plt.subplots(len(top_indices), 8, figsize=(20, 10))
    col_titles = ["CT Input", "Ground Truth"] + [r["short"] for r in RUN_SPECS]

    for row_idx, slice_idx in enumerate(top_indices):
        img_t, msk_t = eval_dataset[int(slice_idx)]
        img_np = img_t.squeeze(0).numpy()
        msk_np = msk_t.squeeze(0).numpy()

        axes[row_idx, 0].imshow(img_np, cmap="gray")
        axes[row_idx, 1].imshow(img_np, cmap="gray")
        axes[row_idx, 1].imshow(np.ma.masked_where(msk_np < 0.5, msk_np), cmap="Greens", alpha=0.65, vmin=0, vmax=1)

        inp = img_t.unsqueeze(0).float().to(device)
        for col_offset, spec in enumerate(RUN_SPECS):
            with torch.no_grad():
                pred_prob = loaded_models[spec["name"]](inp).squeeze().cpu().numpy()
            pred_bin = (pred_prob > 0.5).astype(np.float32)

            ax_cell = axes[row_idx, col_offset + 2]
            ax_cell.imshow(img_np, cmap="gray")
            # Overlay Ground Truth contour in green and Prediction in red/yellow
            ax_cell.contour(msk_np, levels=[0.5], colors="lime", linewidths=1.0)
            ax_cell.imshow(np.ma.masked_where(pred_bin < 0.5, pred_bin), cmap="autumn", alpha=0.55, vmin=0, vmax=1)

        for c in range(8):
            axes[row_idx, c].axis("off")
            if row_idx == 0:
                axes[row_idx, c].set_title(col_titles[c], fontsize=9.5, fontweight="bold")

    plt.suptitle("Qualitative Comparison Across All 6 Part 2 Models (Green Contour = GT, Red/Yellow = Prediction)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    qual_fig_path = os.path.join(res_dir, "part2_qualitative_overlays.png")
    plt.savefig(qual_fig_path, dpi=200)
    plt.close()

    # Upload all evaluation artifacts to Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        api = HfApi(token=hf_token)
        artifacts = [summary_json_path, bar_fig_path, qual_fig_path]
        if conv_fig_path:
            artifacts.append(conv_fig_path)
        for path in artifacts:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"part2/{os.path.basename(path)}",
                repo_id=HF_REPO_ID,
                repo_type="model",
            )
        print("[HF Sync] Uploaded all Part 2 evaluation artifacts to Hugging Face.", flush=True)

    volume.commit()
    return summary_payload


@app.local_entrypoint()
def main():
    payload = run_full_part2_evaluation.remote(batch_size=32)
    os.makedirs("results/part2", exist_ok=True)
    with open("results/part2/part2_master_summary.json", "w") as f:
        json.dump(payload, f, indent=4)

    print("\n=== Final Part 2 Evaluation Table ===")
    for r in payload["runs"]:
        print(
            f"{r['label']:<32} | Dice: {r['dice']:.4f} | IoU: {r['iou']:.4f} | "
            f"Sens: {r['sensitivity']:.4f} | Spec: {r['specificity']:.4f} | Prec: {r['precision']:.4f}"
        )