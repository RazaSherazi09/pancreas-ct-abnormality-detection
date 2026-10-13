import os
from pathlib import Path
import modal

HF_REPO_ID = "razaasherazi/pancreas-multiscale-unet-baseline"

app = modal.App("pancreas-exp1-runner")
volume = modal.Volume.from_name("pancreas-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python-headless",
        "numpy",
        "matplotlib",
        "tqdm",
        "huggingface_hub"
    )
    .add_local_dir(str(Path(__file__).parent / "src"), remote_path="/root/src")
)

@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    cpu=4,
    memory=16384
)
def run_exp1_training(epochs: int = 15, batch_size: int = 32, lr: float = 1e-4):
    import sys
    sys.path.append("/root")

    import json
    import time
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from huggingface_hub import HfApi

    from src.dataset import CachedPancreasDataset
    from src.models.multiscale_unet import MultiScaleUNet
    from src.utils.loss import DiceBCELoss
    from src.utils.metrics import compute_segmentation_metrics
    from src.preprocess import train_transform_exp1, val_transform_exp1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[GPU Initialized] Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

    # 1. Load pre-packed binary cache from Modal volume
    cache_path = "/vol/pancreas_cache.pt"
    print(f"[Data Loader] Loading dataset cache from {cache_path}...", flush=True)
    t0 = time.time()
    data = torch.load(cache_path, weights_only=True)
    print(f"[Data Loader] Loaded into memory in {time.time() - t0:.2f}s!", flush=True)

    # 2. Dataset with Experiment 1 Transformations (CLAHE + Augmentation)
    train_dataset = CachedPancreasDataset(data["train_images"], data["train_masks"], transform_fn=train_transform_exp1)
    val_dataset = CachedPancreasDataset(data["val_images"], data["val_masks"], transform_fn=val_transform_exp1)

    print(f"[Dataset] Train slices: {len(train_dataset)} | Val slices: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 3. Model setup
    model = MultiScaleUNet(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "val_dice": [], "val_iou": [], "val_sensitivity": [], "val_specificity": []
    }

    best_dice = 0.0
    checkpoint_dir = "/vol/checkpoints"
    results_dir = "/vol/results_exp1"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, "exp1_best_model.pth")
    metrics_log_path = os.path.join(results_dir, "exp1_training_log.json")

    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token) if hf_token else None

    print("\n=======================================================", flush=True)
    print("   Starting Experiment 1 Training (CLAHE + Augment)   ", flush=True)
    print("=======================================================", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for images, masks in pbar_train:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            pbar_train.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_epoch_loss = running_train_loss / len(train_dataset)

        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        epoch_metrics = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0}

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]")
        with torch.no_grad():
            for images, masks in pbar_val:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                preds = model(images)
                loss = criterion(preds, masks)
                running_val_loss += loss.item() * images.size(0)

                batch_m = compute_segmentation_metrics(preds, masks)
                for k in epoch_metrics:
                    epoch_metrics[k] += batch_m[k] * images.size(0)

        val_epoch_loss = running_val_loss / len(val_dataset)
        for k in epoch_metrics:
            epoch_metrics[k] /= len(val_dataset)

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["val_dice"].append(epoch_metrics["dice"])
        history["val_iou"].append(epoch_metrics["iou"])
        history["val_sensitivity"].append(epoch_metrics["sensitivity"])
        history["val_specificity"].append(epoch_metrics["specificity"])

        summary = (
            f"\n>>> [Exp 1] Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | "
            f"Dice: {epoch_metrics['dice']:.4f} | "
            f"IoU: {epoch_metrics['iou']:.4f} | "
            f"Sens: {epoch_metrics['sensitivity']:.4f} | "
            f"Spec: {epoch_metrics['specificity']:.4f}\n"
        )
        print(summary, flush=True)

        with open(metrics_log_path, "w") as f:
            json.dump(history, f, indent=4)

        is_best = epoch_metrics["dice"] > best_dice
        if is_best:
            best_dice = epoch_metrics["dice"]
            torch.save(model.state_dict(), best_model_path)
            print(f"[*] New best Exp 1 Dice score: {best_dice:.4f}! Saved locally.", flush=True)

        # Sync checkpoint and log to Hugging Face
        if api:
            try:
                api.upload_file(
                    path_or_fileobj=metrics_log_path,
                    path_in_repo="exp1_training_log.json",
                    repo_id=HF_REPO_ID,
                    repo_type="model"
                )
                if is_best:
                    api.upload_file(
                        path_or_fileobj=best_model_path,
                        path_in_repo="exp1_best_model.pth",
                        repo_id=HF_REPO_ID,
                        repo_type="model"
                    )
                    print("[*] Uploaded new best Exp 1 weights to Hugging Face.", flush=True)
            except Exception as e:
                print(f"[HF Sync Note] {e}", flush=True)

    # Plot Exp 1 Loss & Metric Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", marker='o')
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Exp 1 (CLAHE + Aug): Loss Curves")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["val_dice"], label="Dice", color="green", marker='^')
    plt.plot(range(1, epochs + 1), history["val_iou"], label="IoU", color="orange", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Exp 1 (CLAHE + Aug): Validation Metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    loss_curve_path = os.path.join(results_dir, "exp1_loss_curves.png")
    plt.savefig(loss_curve_path)
    plt.close()

    if api:
        try:
            api.upload_file(
                path_or_fileobj=loss_curve_path,
                path_in_repo="exp1_loss_curves.png",
                repo_id=HF_REPO_ID,
                repo_type="model"
            )
            print("[*] Uploaded exp1_loss_curves.png to Hugging Face.", flush=True)
        except Exception as e:
            print(f"[HF Plot Note] {e}", flush=True)

    volume.commit()
    return {
        "best_dice": best_dice,
        "final_dice": history["val_dice"][-1],
        "final_iou": history["val_iou"][-1]
    }

@app.local_entrypoint()
def main():
    print("Launching Experiment 1 (CLAHE + Augmentation) on Cloud T4 GPU...", flush=True)
    results = run_exp1_training.remote(epochs=15, batch_size=32)
    print("\n--- Experiment 1 Completed Successfully ---", flush=True)
    for k, v in results.items():
        print(f" {k}: {v:.4f}")