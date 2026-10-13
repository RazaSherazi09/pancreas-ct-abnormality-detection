import os
from pathlib import Path
import modal

HF_REPO_ID = "razaasherazi/pancreas-multiscale-unet-baseline"

app = modal.App("pancreas-evaluator")
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
def evaluate_and_generate_visuals(num_visual_samples: int = 5):
    import sys
    sys.path.append("/root")

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from huggingface_hub import HfApi

    from src.dataset import CachedPancreasDataset
    from src.models.multiscale_unet import MultiScaleUNet
    from src.utils.metrics import compute_segmentation_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data and Checkpoint
    data = torch.load("/vol/pancreas_cache.pt", weights_only=True)
    val_dataset = CachedPancreasDataset(data["val_images"], data["val_masks"])
    
    model = MultiScaleUNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("/vol/checkpoints/baseline_best_model.pth", map_location=device))
    model.eval()

    results_dir = "/vol/results_baseline"
    os.makedirs(results_dir, exist_ok=True)

    print(f"[Evaluation] Evaluating on {len(val_dataset)} validation slices...", flush=True)

    # 2. Confusion Matrix Accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)

            bin_preds = (preds > 0.5).float().view(-1)
            bin_targets = (masks > 0.5).float().view(-1)

            total_tp += (bin_preds * bin_targets).sum().item()
            total_fp += (bin_preds * (1.0 - bin_targets)).sum().item()
            total_fn += ((1.0 - bin_preds) * bin_targets).sum().item()
            total_tn += ((1.0 - bin_preds) * (1.0 - bin_targets)).sum().item()

    # 3. Plot Pixel Confusion Matrix
    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
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
    ax.set_title('Baseline Multi-Scale U-Net: Pixel-Level Confusion Matrix')
    fig.colorbar(im)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "baseline_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # 4. Generate Qualitative Segmentation Overlays
    indices = np.linspace(0, len(val_dataset) - 1, num_visual_samples, dtype=int)
    fig, axes = plt.subplots(num_visual_samples, 4, figsize=(14, 3.5 * num_visual_samples))

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img_tensor, mask_tensor = val_dataset[idx]
            pred_tensor = model(img_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()

            raw_img = img_tensor.squeeze().numpy()
            gt_mask = mask_tensor.squeeze().numpy()
            pred_mask = (pred_tensor.squeeze().numpy() > 0.5).astype(np.float32)

            # Error overlay: Green = TP, Red = FP, Blue = FN
            tp = (gt_mask == 1) & (pred_mask == 1)
            fp = (gt_mask == 0) & (pred_mask == 1)
            fn = (gt_mask == 1) & (pred_mask == 0)

            overlay = np.zeros((*raw_img.shape, 3), dtype=np.float32)
            overlay[..., 0] = raw_img
            overlay[..., 1] = raw_img
            overlay[..., 2] = raw_img

            overlay[tp] = [0.0, 1.0, 0.0]  # Green (Hit)
            overlay[fp] = [1.0, 0.0, 0.0]  # Red (False Alarm)
            overlay[fn] = [0.0, 0.2, 1.0]  # Blue (Missed)

            axes[row, 0].imshow(raw_img, cmap='gray')
            axes[row, 0].set_title(f"Slice #{idx}: Input CT")
            axes[row, 0].axis('off')

            axes[row, 1].imshow(gt_mask, cmap='viridis')
            axes[row, 1].set_title("Ground Truth Mask")
            axes[row, 1].axis('off')

            axes[row, 2].imshow(pred_mask, cmap='viridis')
            axes[row, 2].set_title("Prediction (Threshold 0.5)")
            axes[row, 2].axis('off')

            axes[row, 3].imshow(overlay)
            axes[row, 3].set_title("Overlay (G:TP, R:FP, B:FN)")
            axes[row, 3].axis('off')

    plt.tight_layout()
    overlay_path = os.path.join(results_dir, "baseline_segmentation_overlays.png")
    plt.savefig(overlay_path)
    plt.close()

    # 5. Sync to Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        api = HfApi(token=hf_token)
        api.upload_file(path_or_fileobj=cm_path, path_in_repo="baseline_confusion_matrix.png", repo_id=HF_REPO_ID, repo_type="model")
        api.upload_file(path_or_fileobj=overlay_path, path_in_repo="baseline_segmentation_overlays.png", repo_id=HF_REPO_ID, repo_type="model")
        print("[Hugging Face] Uploaded confusion matrix and visual overlays successfully!", flush=True)

    volume.commit()
    print("[Evaluation] Completed successfully!", flush=True)

@app.local_entrypoint()
def main():
    evaluate_and_generate_visuals.remote()