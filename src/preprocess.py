import cv2
import numpy as np

# Contrast-Limited Adaptive Histogram Equalization
clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(img_np: np.ndarray) -> np.ndarray:
    """Applies CLAHE on normalized uint8 2D CT axial slice."""
    return clahe_filter.apply(img_np)

def train_transform_exp1(img_np: np.ndarray, mask_np: np.ndarray):
    """
    Joint transformation for Experiment 1 (Training):
    1. Random Horizontal Flip (50% probability)
    2. Random Vertical Flip (50% probability)
    3. Random Rotation (+-10 degrees with zero border padding for mask)
    4. CLAHE contrast enhancement on image
    """
    # 1. Random Horizontal Flip
    if np.random.rand() > 0.5:
        img_np = np.fliplr(img_np).copy()
        mask_np = np.fliplr(mask_np).copy()

    # 2. Random Vertical Flip
    if np.random.rand() > 0.5:
        img_np = np.flipud(img_np).copy()
        mask_np = np.flipud(mask_np).copy()

    # 3. Random Slight Rotation (-10 to +10 degrees)
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10.0, 10.0)
        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_np = cv2.warpAffine(img_np, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_np = cv2.warpAffine(mask_np, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 4. Apply CLAHE on the slice
    img_np = apply_clahe(img_np)

    return img_np, mask_np

def val_transform_exp1(img_np: np.ndarray, mask_np: np.ndarray):
    """
    Validation transformation for Experiment 1:
    Deterministic CLAHE contrast enhancement without spatial jitter.
    """
    img_np = apply_clahe(img_np)
    return img_np, mask_np