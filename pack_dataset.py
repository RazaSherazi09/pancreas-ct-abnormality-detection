import os
import modal

app = modal.App("pancreas-data-packer")
volume = modal.Volume.from_name("pancreas-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "opencv-python-headless", "numpy", "tqdm")
)

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=1800,           # Increased to 30 mins to prevent timeout
    cpu=8,
    memory=16384
)
def pack_dataset_to_pt():
    import json
    import cv2
    import torch
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    data_dir = "/vol/dataset_2d"
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    with open(os.path.join(data_dir, "splits.json"), "r") as f:
        splits = json.load(f)

    train_slices = splits["train_slices"]
    val_split_idx = int(len(train_slices) * 0.8)
    train_files = train_slices[:val_split_idx]
    val_files = train_slices[val_split_idx:]

    def load_single_pair(slice_name):
        img_path = os.path.join(images_dir, slice_name)
        mask_path = os.path.join(masks_dir, slice_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            return None, None
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask_resized > 0).astype(np.uint8)
        return img_resized, mask_bin

    def load_set_multithreaded(file_list, name):
        print(f"\n[Packer] Loading {len(file_list)} {name} slices with 48 worker threads...", flush=True)
        imgs, msks = [], []
        with ThreadPoolExecutor(max_workers=48) as executor:
            results = list(tqdm(executor.map(load_single_pair, file_list), total=len(file_list)))
        for img, msk in results:
            if img is not None:
                imgs.append(img)
                msks.append(msk)
        
        imgs_tensor = torch.from_numpy(np.array(imgs, dtype=np.uint8)).unsqueeze(1)
        msks_tensor = torch.from_numpy(np.array(msks, dtype=np.uint8)).unsqueeze(1)
        return imgs_tensor, msks_tensor

    train_imgs, train_msks = load_set_multithreaded(train_files, "Train")
    val_imgs, val_msks = load_set_multithreaded(val_files, "Val")

    cache_payload = {
        "train_images": train_imgs,
        "train_masks": train_msks,
        "val_images": val_imgs,
        "val_masks": val_msks
    }

    cache_path = "/vol/pancreas_cache.pt"
    print(f"\n[Packer] Saving unified binary cache to {cache_path}...", flush=True)
    torch.save(cache_payload, cache_path)
    volume.commit()
    print("\n[Packer] SUCCESS: Dataset successfully packed and committed to Modal volume!", flush=True)

@app.local_entrypoint()
def main():
    pack_dataset_to_pt.remote()