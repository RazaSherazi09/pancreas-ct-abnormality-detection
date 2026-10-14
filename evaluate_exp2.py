import os
from pathlib import Path
import modal

HF_REPO_ID = "razaasherazi/pancreas-multiscale-unet-baseline"

app = modal.App("pancreas-exp2-evaluator")
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
def evaluate_exp2_tri_comparison(num_samples: int = 5):
    import sys
    sys.path.append("/root")

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from huggingface_hub import HfApi

    from src.dataset import CachedPancreasDataset
    from src.models.multiscale_unet import MultiScaleUNet
    from src.preprocess import val_transform_exp1, val_transform_exp2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load("/vol/pancreas_cache.pt", weights_only=True)
    val_dataset_raw = CachedPancreasDataset(data["val_images"], data["val_masks"])
    val_dataset_exp1 = CachedPancreasDataset(data["val_images"], data["val_masks"], transform_fn=val_transform_exp1)
    val_dataset_exp2 = CachedPancreasDataset(data["val_images"], data["val_masks"], transform_fn=val_transform_exp2)

    model_base = MultiScaleUNet().to(device)
    model_base.load_state_dict(torch.load("/vol/checkpoints/baseline_best_model.pth", map_location=device))
    model_base.eval()

    model_exp1 = MultiScaleUNet().to(device)
    model_exp1.load_state_dict(torch.load("/vol/checkpoints/exp1_best_model.pth", map_location=device))
    model_exp1.eval()

    model_exp2 = MultiScaleUNet().to(device)
    model_exp2.load_state_dict(torch.load("/vol/checkpoints/exp2_best_model.pth", map_location=device))
    model_exp2.eval()

    results_dir = "/vol/results_exp2"
    os.makedirs(results_dir, exist_ok=True)

    val_loader = torch.utils.data.DataLoader(val_dataset_exp2, batch_size=32, shuffle=False)
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model_exp2(imgs)
            bin_preds = (preds > 0.5).float().view(-1)
            bin_targets = (masks > 0.5).float().view(-1)

            total_tp += (bin_preds * bin_targets).sum().item()
            total_fp += (bin_preds * (1.0 - bin_targets)).sum().item()
            total_fn += ((1.0 - bin_preds) * bin_targets).sum().item()
            total_tn += ((1.0 - bin_preds) * (1.0 - bin_targets)).sum().item()

    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap='Purples', vmin=0, vmax=1)
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
    ax.set_title('Exp 2 (FTL + Calibrated CLAHE): Confusion Matrix')
    fig.colorbar(im)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "exp2_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    indices = np.linspace(0, len(val_dataset_raw) - 1, num_samples, dtype=int)
    fig, axes = plt.subplots(num_samples, 5, figsize=(18, 3.5 * num_samples))

    with torch.no_grad():
        for row, idx in enumerate(indices):
            raw_t, mask_t = val_dataset_raw[idx]
            exp1_t, _ = val_dataset_exp1[idx]
            exp2_t, _ = val_dataset_exp2[idx]

            p_base = (model_base(raw_t.unsqueeze(0).to(device)).squeeze().cpu().numpy() > 0.5).astype(np.float32)
            p_exp1 = (model_exp1(exp1_t.unsqueeze(0).to(device)).squeeze().cpu().numpy() > 0.5).astype(np.float32)
            p_exp2 = (model_exp2(exp2_t.unsqueeze(0).to(device)).squeeze().cpu().numpy() > 0.5).astype(np.float32)

            raw_np = raw_t.squeeze().numpy()
            gt_np = mask_t.squeeze().numpy()

            axes[row, 0].imshow(raw_np, cmap='gray')
            axes[row, 0].set_title(f"Slice #{idx}: Raw CT")
            axes[row, 0].axis('off')

            axes[row, 1].imshow(gt_np, cmap='viridis')
            axes[row, 1].set_title("Ground Truth")
            axes[row, 1].axis('off')

            axes[row, 2].imshow(p_base, cmap='Blues')
            axes[row, 2].set_title("Baseline (Under-segment)")
            axes[row, 2].axis('off')

            axes[row, 3].imshow(p_exp1, cmap='Reds')
            axes[row, 3].set_title("Exp 1 (Over-segment)")
            axes[row, 3].axis('off')

            axes[row, 4].imshow(p_exp2, cmap='Greens')
            axes[row, 4].set_title("Exp 2 (FTL Calibrated)")
            axes[row, 4].axis('off')

    plt.tight_layout()
    tri_path = os.path.join(results_dir, "complete_tri_ablation_comparison.png")
    plt.savefig(tri_path)
    plt.close()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        api = HfApi(token=hf_token)
        api.upload_file(path_or_fileobj=cm_path, path_in_repo="exp2_confusion_matrix.png", repo_id=HF_REPO_ID, repo_type="model")
        api.upload_file(path_or_fileobj=tri_path, path_in_repo="complete_tri_ablation_comparison.png", repo_id=HF_REPO_ID, repo_type="model")
        print("[Hugging Face] Uploaded Exp 2 evaluation figures successfully!", flush=True)

    volume.commit()
    print("[Evaluation] Exp 2 tri-comparison complete!", flush=True)

@app.local_entrypoint()
def main():
    evaluate_exp2_tri_comparison.remote()