import os
from pathlib import Path
import modal

HF_REPO_ID = "razaasherazi/pancreas-multiscale-unet-baseline"

app = modal.App("pancreas-part2-hybrid-runner")
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
        "huggingface_hub",
    )
    .add_local_dir(str(Path(__file__).parent / "src"), remote_path="/root/src")
)

RUN_CONFIGS = {
    1: {
        "student": "Student 1 (Syed Muhammad Raza)",
        "name": "run1_att_resunet_base",
        "arch": "att_resunet",
        "feature_mode": "none",
        "desc": "Run 1: Att-ResUNet Alone (Deep Model 1 Baseline)",
    },
    2: {
        "student": "Student 1 (Syed Muhammad Raza)",
        "name": "run2_att_resunet_lbp",
        "arch": "att_resunet",
        "feature_mode": "lbp",
        "desc": "Run 2: Att-ResUNet + Dense LBP (SE Attention Fusion)",
    },
    3: {
        "student": "Student 1 (Syed Muhammad Raza)",
        "name": "run3_att_resunet_gabor",
        "arch": "att_resunet",
        "feature_mode": "gabor",
        "desc": "Run 3: Att-ResUNet + Multi-Orientation Gabor Bank (SE Attention Fusion)",
    },
    4: {
        "student": "Student 2 (Muhammad Ahmad)",
        "name": "run4_transunet_base",
        "arch": "transunet_lite",
        "feature_mode": "none",
        "desc": "Run 4: TransUNet-Lite Alone (Deep Model 2 Baseline)",
    },
    5: {
        "student": "Student 2 (Muhammad Ahmad)",
        "name": "run5_transunet_hog",
        "arch": "transunet_lite",
        "feature_mode": "hog",
        "desc": "Run 5: TransUNet-Lite + Dense Spatial HOG (Projection Concat Fusion)",
    },
    6: {
        "student": "Student 2 (Muhammad Ahmad)",
        "name": "run6_transunet_log",
        "arch": "transunet_lite",
        "feature_mode": "log",
        "desc": "Run 6: TransUNet-Lite + Multi-Scale LoG Edges (Projection Concat Fusion)",
    },
}


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
    cpu=4,
    memory=16384,
)
def train_part2_run(run_id: int, epochs: int = 30, batch_size: int = 32, lr: float = 1e-4):
    import sys
    sys.path.append("/root")

    import json
    import time
    import cv2
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from huggingface_hub import HfApi

    from src.dataset import CachedPancreasDataset
    from src.models.att_resunet import AttentionResUNet
    from src.models.transunet_lite import TransUNetLite
    from src.utils.loss import DiceBCELoss
    from src.utils.metrics import compute_segmentation_metrics

    assert run_id in RUN_CONFIGS, f"Invalid run_id {run_id}. Must be 1..6."
    cfg = RUN_CONFIGS[run_id]

    # Enforce strict IEEE-754 FP32 precision on Ampere (A10G) to prevent TF32/autocast drift
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)

    # Deterministic seed for strict experimental consistency across all 6 runs
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[GPU Initialized] {cfg['desc']}", flush=True)
    print(f"[Hardware] Device: {device} ({torch.cuda.get_device_name(0)}) | Strict FP32 Enabled", flush=True)

    # Load pre-cached dataset from Modal Volume
    t0 = time.time()
    data = torch.load("/vol/pancreas_cache.pt", weights_only=True)
    print(f"[Data Loader] Loaded /vol/pancreas_cache.pt in {time.time() - t0:.2f}s", flush=True)

    def part2_train_transform(img_np: np.ndarray, mask_np: np.ndarray):
        if np.random.rand() > 0.5:
            img_np = np.fliplr(img_np).copy()
            mask_np = np.fliplr(mask_np).copy()
        if np.random.rand() > 0.5:
            img_np = np.flipud(img_np).copy()
            mask_np = np.flipud(mask_np).copy()
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-8.0, 8.0)
            h, w = img_np.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img_np = cv2.warpAffine(img_np, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask_np = cv2.warpAffine(mask_np, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img_np, mask_np

    train_dataset = CachedPancreasDataset(data["train_images"], data["train_masks"], transform_fn=part2_train_transform)
    val_dataset = CachedPancreasDataset(data["val_images"], data["val_masks"], transform_fn=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    if cfg["arch"] == "att_resunet":
        model = AttentionResUNet(in_channels=1, out_channels=1, feature_mode=cfg["feature_mode"]).float().to(device)
    else:
        model = TransUNetLite(in_channels=1, out_channels=1, feature_mode=cfg["feature_mode"]).float().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Architecture] {cfg['arch']} | Feature Mode: {cfg['feature_mode']} | Params: {total_params:,}", flush=True)

    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = "/vol/checkpoints_part2"
    res_dir = "/vol/results_part2"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    run_prefix = cfg["name"]
    best_model_path = os.path.join(ckpt_dir, f"{run_prefix}_best.pth")
    last_ckpt_path = os.path.join(ckpt_dir, f"{run_prefix}_last.pt")
    log_path = os.path.join(res_dir, f"{run_prefix}_log.json")

    start_epoch = 1
    best_dice = 0.0
    best_metrics = {}
    history = {
        "run_id": run_id,
        "student": cfg["student"],
        "arch": cfg["arch"],
        "feature_mode": cfg["feature_mode"],
        "params": total_params,
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_sensitivity": [],
        "val_specificity": [],
    }

    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_dice"]
        best_metrics = ckpt.get("best_metrics", {})
        history = ckpt["history"]
        print(f"[Resume] Resuming {run_prefix} from Epoch {start_epoch} (Best Dice so far: {best_dice:.4f})", flush=True)

    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token) if hf_token else None

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_train_loss = 0.0

        pbar_train = tqdm(train_loader, desc=f"[{run_prefix}] Ep {epoch:02d}/{epochs:02d} [Train]")
        for images, masks in pbar_train:
            images = images.float().to(device, non_blocking=True)
            masks = masks.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            pbar_train.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_epoch_loss = running_train_loss / len(train_dataset)

        # Validation
        model.eval()
        running_val_loss = 0.0
        epoch_metrics = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0}

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.float().to(device, non_blocking=True)
                masks = masks.float().to(device, non_blocking=True)

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

        print(
            f"\n>>> [{run_prefix}] Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | "
            f"Dice: {epoch_metrics['dice']:.4f} | IoU: {epoch_metrics['iou']:.4f} | "
            f"Sens: {epoch_metrics['sensitivity']:.4f} | Spec: {epoch_metrics['specificity']:.4f}",
            flush=True,
        )

        is_best = epoch_metrics["dice"] > best_dice
        if is_best:
            best_dice = epoch_metrics["dice"]
            best_metrics = {
                "best_epoch": epoch,
                "dice": epoch_metrics["dice"],
                "iou": epoch_metrics["iou"],
                "sensitivity": epoch_metrics["sensitivity"],
                "specificity": epoch_metrics["specificity"],
            }
            history["best_metrics"] = best_metrics
            torch.save(model.state_dict(), best_model_path)
            print(f"[*] New Best {run_prefix} Dice: {best_dice:.4f} (Epoch {epoch})", flush=True)

        with open(log_path, "w") as f:
            json.dump(history, f, indent=4)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_dice": best_dice,
                "best_metrics": best_metrics,
                "history": history,
            },
            last_ckpt_path,
        )
        volume.commit()

        if api and (is_best or epoch == epochs):
            try:
                api.upload_file(
                    path_or_fileobj=log_path,
                    path_in_repo=f"part2/{run_prefix}_log.json",
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                )
                if is_best:
                    api.upload_file(
                        path_or_fileobj=best_model_path,
                        path_in_repo=f"part2/{run_prefix}_best.pth",
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                    )
            except Exception as e:
                print(f"[HF Sync Note] {e}", flush=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", marker="o", markersize=3)
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Dice + BCE Loss")
    plt.title(f"{run_prefix}: Loss Curves")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["val_dice"], label="Val Dice", color="green", marker="^", markersize=3)
    plt.plot(range(1, epochs + 1), history["val_iou"], label="Val IoU", color="orange", marker="x", markersize=3)
    plt.axhline(y=0.7420, color="red", linestyle="--", label="Base Paper (0.7420)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{run_prefix}: Validation Metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    curve_path = os.path.join(res_dir, f"{run_prefix}_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()

    if api:
        try:
            api.upload_file(
                path_or_fileobj=curve_path,
                path_in_repo=f"part2/{run_prefix}_curves.png",
                repo_id=HF_REPO_ID,
                repo_type="model",
            )
        except Exception as e:
            print(f"[HF Plot Note] {e}", flush=True)

    volume.commit()
    return best_metrics


@app.local_entrypoint()
def main(run: int = 1, epochs: int = 30, batch_size: int = 32):
    print(f"Dispatching Part 2 Run {run} ({RUN_CONFIGS[run]['desc']}) on Modal A10G GPU...", flush=True)
    res = train_part2_run.remote(run_id=run, epochs=epochs, batch_size=batch_size)
    print(f"\n=== Completed Run {run} ===", flush=True)
    for k, v in res.items():
        print(f"  {k}: {v}")