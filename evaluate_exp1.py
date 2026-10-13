import os
from pathlib import Path
import modal

HF_REPO_ID = "razaasherazi/pancreas-multiscale-unet-baseline"

app = modal.App("pancreas-exp1-evaluator")
volume = modal.Volume.from_name("pancreas-data-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "opencv-python-headless", "numpy", "matplotlib", "huggingface_hub")
    .add_local_dir(str(Path(__file__).parent / "src"), remote_path="/root/src")
)

@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    cpu=4,
    memory=8192
)
def evaluate_exp1_and_compare(num_samples: int = 5):
    import sys
    sys.path.append("/root")

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from huggingface_hub import HfApi

    from src.dataset import CachedPancreasDataset
    from src.models.multiscale_unet import MultiScaleUNet
    from src.preprocess import val_transform_exp1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    data = torch.load("/vol/pancreas_cache.pt", weights_only=True)
    val_dataset_raw = CachedPancreasDataset(data["val_images"], data["val_masks"])
    val_dataset_clahe = CachedPancreasDataset(data["val_images"], data["val_masks"], transform_fn=val_transform_exp1)

    # 2. Load Models
    model_baseline = MultiScaleUNet().to(device)
    model_baseline.load_state_dict(torch.load("/vol/checkpoints/baseline_best_model.pth", map_location=device))
    model_baseline.eval()

    model_exp1 = MultiScaleUNet().to(device)
    model_exp1.load_state_dict(torch.load("/vol/checkpoints/exp1_best_model.pth", map_location=device))
    model_exp1.eval()

    results_dir = "/vol/results_exp1"
    os.makedirs(results_dir, exist_ok=True)

    # 3. Exp 1 Confusion Matrix
    val_loader = torch.utils.data.DataLoader(val_dataset_clahe, batch_size=32, shuffle=False)
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model_exp1(imgs)
            bin_preds = (preds > 0.5).float().view(-1)
            bin_targets = (masks > 0.5).float().view(-1)

            total_tp += (bin_preds * bin_targets).sum().item()
            total_fp += (bin_preds * (1.0 - bin_targets)).sum().item()
            total_fn += ((1.0 - bin_preds) * bin_targets).sum().item()
            total_tn += ((1.0 - bin_preds) * (1.0 - bin_targets)).sum().item()

    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap='Greens', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j]*100:.2f}%\n({int(cm[i, j]):,})",
                    ha="center", va="center", color="black" if cm_norm[i, j] < 0.5 else "white")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Background', 'Pancreas'])
    ax.set_yticklabels(['Background', 'Pancreas'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Exp 1 (CLAHE + Aug): Pixel Confusion Matrix')
    fig.colorbar(im)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "exp1_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # 4. Comparative Side-by-Side: Baseline vs Exp 1
    indices = np.linspace(0, len(val_dataset_raw) - 1, num_samples, dtype=int)
    fig, axes = plt.subplots(num_samples, 5, figsize=(18, 3.5 * num_samples))

    with torch.no_grad():
        for row, idx in enumerate(indices):
            raw_img_t, mask_t = val_dataset_raw[idx]
            clahe_img_t, _ = val_dataset_clahe[idx]

            pred_base_t = model_baseline(raw_img_t.unsqueeze(0).to(device)).squeeze().cpu()
            pred_exp1_t = model_exp1(clahe_img_t.unsqueeze(0).to(device)).squeeze().cpu()

            raw_img = raw_img_t.squeeze().numpy()
            gt_mask = mask_t.squeeze().numpy()
            pred_base = (pred_base_t.numpy() > 0.5).astype(np.float32)
            pred_exp1 = (pred_exp1_t.numpy() > 0.5).astype(np.float32)

            axes[row, 0].imshow(raw_img, cmap='gray')
            axes[row, 0].set_title(f"Slice #{idx}: Input CT")
            axes[row, 0].axis('off')

            axes[row, 1].imshow(gt_mask, cmap='viridis')
            axes[row, 1].set_title("Ground Truth Mask")
            axes[row, 1].axis('off')

            axes[row, 2].imshow(pred_base, cmap='Blues')
            axes[row, 2].set_title("Baseline Prediction")
            axes[row, 2].axis('off')

            axes[row, 3].imshow(pred_exp1, cmap='Greens')
            axes[row, 3].set_title("Exp 1 Prediction (CLAHE)")
            axes[row, 3].axis('off')

            # Delta comparison: Green = Exp 1 Fixed, Red = Baseline Disagreement
            diff_map = np.zeros((*raw_img.shape, 3), dtype=np.float32)
            diff_map[..., 0] = raw_img
            diff_map[..., 1] = raw_img
            diff_map[..., 2] = raw_img
            diff_map[(gt_mask == 1) & (pred_exp1 == 1)] = [0.0, 1.0, 0.0] # True Hit
            diff_map[(pred_base == 0) & (pred_exp1 == 1) & (gt_mask == 1)] = [1.0, 0.8, 0.0] # Improved by Exp1

            axes[row, 4].imshow(diff_map)
            axes[row, 4].set_title("Yellow: Gained by Exp1")
            axes[row, 4].axis('off')

    plt.tight_layout()
    comp_path = os.path.join(results_dir, "baseline_vs_exp1_comparison.png")
    plt.savefig(comp_path)
    plt.close()

    # 5. Sync to Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        api = HfApi(token=hf_token)
        api.upload_file(path_or_fileobj=cm_path, path_in_repo="exp1_confusion_matrix.png", repo_id=HF_REPO_ID, repo_type="model")
        api.upload_file(path_or_fileobj=comp_path, path_in_repo="baseline_vs_exp1_comparison.png", repo_id=HF_REPO_ID, repo_type="model")
        print("[Hugging Face] Uploaded Exp 1 evaluation assets successfully!", flush=True)

    volume.commit()
    print("[Evaluation] Exp 1 evaluation complete!", flush=True)

@app.local_entrypoint()
def main():
    evaluate_exp1_and_compare.remote()