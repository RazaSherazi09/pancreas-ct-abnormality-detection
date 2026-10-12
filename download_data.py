import os
import glob
import json
import tarfile
import urllib.request
import modal

# 1. Define App and persistent Cloud Volume
app = modal.App("pancreas-data-preparer")
volume = modal.Volume.from_name("pancreas-data-vol", create_if_missing=True)

# 2. Container image with medical image and computer vision tools
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "nibabel",
        "opencv-python-headless",
        "tqdm"
    )
)

HU_MIN, HU_MAX = -100, 240

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=3600,   # 1 hour maximum execution time
    cpu=4,
    memory=8192
)
def prepare_dataset():
    import numpy as np
    import nibabel as nib
    import cv2
    from tqdm import tqdm

    raw_archive = "/vol/Task07_Pancreas.tar"
    extracted_dir = "/vol/Task07_Pancreas"
    output_dir = "/vol/dataset_2d"

    # --- STEP A: Download Tar File if not already present ---
    if not os.path.exists(extracted_dir):
        if not os.path.exists(raw_archive):
            url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"
            print(f"[Modal Cloud] Downloading MSD Pancreas dataset from:\n{url}")
            urllib.request.urlretrieve(url, raw_archive)
            print("[Modal Cloud] Download complete.")

        print("[Modal Cloud] Extracting archive...")
        with tarfile.open(raw_archive, "r:*") as tar:
            tar.extractall(path="/vol")
        print("[Modal Cloud] Extraction complete.")

        if os.path.exists(raw_archive):
            os.remove(raw_archive)
    else:
        print("[Modal Cloud] Raw dataset already exists in Volume. Skipping download.")

    # --- STEP B: Slice Extraction & Preprocessing to 2D ---
    images_2d_dir = os.path.join(output_dir, "images")
    masks_2d_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_2d_dir, exist_ok=True)
    os.makedirs(masks_2d_dir, exist_ok=True)

    # Check if already processed
    existing_slices = glob.glob(os.path.join(images_2d_dir, "*.png"))
    if len(existing_slices) > 2000:
        print(f"[Modal Cloud] Found {len(existing_slices)} 2D slices already processed. Skipping extraction.")
        return

    images_tr_dir = os.path.join(extracted_dir, "imagesTr")
    labels_tr_dir = os.path.join(extracted_dir, "labelsTr")

    cases = sorted([
        f for f in os.listdir(images_tr_dir)
        if f.startswith("pancreas_") and f.endswith(".nii.gz") and not f.startswith("._")
    ])

    print(f"[Modal Cloud] Found {len(cases)} total cases. Converting to 2D axial slices...")

    # Train / Test split matching the base paper: 216 train, remaining ~65 test
    train_cases = cases[:216]
    test_cases = cases[216:]

    splits = {
        "train_cases": train_cases,
        "test_cases": test_cases,
        "train_slices": [],
        "test_slices": []
    }

    total_slices_saved = 0

    for case_file in tqdm(cases, desc="Processing 3D CT Volumes"):
        is_train = case_file in train_cases
        case_id = case_file.replace(".nii.gz", "")
        img_path = os.path.join(images_tr_dir, case_file)
        lbl_path = os.path.join(labels_tr_dir, case_file)

        if not os.path.exists(lbl_path):
            continue

        try:
            img_nii = nib.load(img_path).get_fdata()
            lbl_nii = nib.load(lbl_path).get_fdata()
        except Exception as e:
            print(f"Skipping {case_file} due to load error: {e}")
            continue

        num_slices = img_nii.shape[2]

        for s in range(num_slices):
            mask_slice = lbl_nii[:, :, s]

            # Only retain slices that have labeled anatomy (pancreas=1 or tumor=2)
            if np.sum(mask_slice > 0) > 0:
                img_slice = img_nii[:, :, s]

                # 1. Soft-tissue HU Windowing [-100, 240]
                img_clipped = np.clip(img_slice, HU_MIN, HU_MAX)

                # 2. Min-Max Normalization to 0-255 uint8
                img_norm = ((img_clipped - HU_MIN) / (HU_MAX - HU_MIN) * 255.0).astype(np.uint8)
                mask_uint8 = mask_slice.astype(np.uint8)

                slice_id = f"{case_id}_slice_{s:03d}.png"
                img_save_path = os.path.join(images_2d_dir, slice_id)
                mask_save_path = os.path.join(masks_2d_dir, slice_id)

                cv2.imwrite(img_save_path, img_norm)
                cv2.imwrite(mask_save_path, mask_uint8)

                if is_train:
                    splits["train_slices"].append(slice_id)
                else:
                    splits["test_slices"].append(slice_id)

                total_slices_saved += 1

    # Save split mappings for consistent training across all 3 experiments
    with open(os.path.join(output_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=4)

    print(f"\n[Modal Cloud] Processing complete!")
    print(f" - Total slices saved: {total_slices_saved}")
    print(f" - Train slices: {len(splits['train_slices'])}")
    print(f" - Test slices: {len(splits['test_slices'])}")

    # Commit persistent volume changes to Modal Cloud
    volume.commit()
    print("[Modal Cloud] Persistent Volume committed successfully.")

@app.local_entrypoint()
def main():
    print("Dispatching dataset preparation task to Modal Cloud...")
    prepare_dataset.remote()
    print("Finished! The dataset is permanently saved in your Modal cloud volume.")