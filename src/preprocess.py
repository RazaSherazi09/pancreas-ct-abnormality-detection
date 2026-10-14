import cv2
import numpy as np

# Experiment 1: High clip limit (aggressive contrast expansion)
clahe_exp1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Experiment 2: Calibrated gentle clip limit (controls edge noise and boundary dilation)
clahe_exp2 = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))


def apply_clahe(img_np: np.ndarray) -> np.ndarray:
    """Applies default Experiment 1 CLAHE on a uint8 2D slice."""
    return clahe_exp1.apply(img_np)


# =====================================================================
# Experiment 1 Transformations (CLAHE clip=2.0 + High-degree Aug)
# =====================================================================

def train_transform_exp1(img_np: np.ndarray, mask_np: np.ndarray):
    """
    Experiment 1 Training:
    1. Random Horizontal & Vertical Flips (p=0.5)
    2. Random Affine Rotation (-10 to +10 degrees)
    3. CLAHE (clipLimit=2.0)
    """
    if np.random.rand() > 0.5:
        img_np = np.fliplr(img_np).copy()
        mask_np = np.fliplr(mask_np).copy()

    if np.random.rand() > 0.5:
        img_np = np.flipud(img_np).copy()
        mask_np = np.flipud(mask_np).copy()

    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10.0, 10.0)
        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_np = cv2.warpAffine(img_np, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_np = cv2.warpAffine(mask_np, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    img_np = clahe_exp1.apply(img_np)
    return img_np, mask_np


def val_transform_exp1(img_np: np.ndarray, mask_np: np.ndarray):
    """Experiment 1 Validation: Deterministic CLAHE (clip=2.0)."""
    return clahe_exp1.apply(img_np), mask_np


# =====================================================================
# Experiment 2 Transformations (Calibrated CLAHE clip=1.2 + Regularized Aug)
# =====================================================================

def train_transform_exp2(img_np: np.ndarray, mask_np: np.ndarray):
    """
    Experiment 2 Training:
    1. Random Horizontal & Vertical Flips (p=0.5)
    2. Subtle Regularized Affine Rotation (-8 to +8 degrees)
    3. Calibrated CLAHE (clipLimit=1.2) to mitigate boundary over-dilation
    """
    if np.random.rand() > 0.5:
        img_np = np.fliplr(img_np).copy()
        mask_np = np.fliplr(mask_np).copy()

    if np.random.rand() > 0.5:
        img_np = np.flipud(img_np).copy()
        mask_np = np.flipud(mask_np).copy()

    if np.random.rand() > 0.5:
        angle = np.random.uniform(-8.0, 8.0)
        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_np = cv2.warpAffine(img_np, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_np = cv2.warpAffine(mask_np, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    img_np = clahe_exp2.apply(img_np)
    return img_np, mask_np


def val_transform_exp2(img_np: np.ndarray, mask_np: np.ndarray):
    """Experiment 2 Validation: Deterministic Calibrated CLAHE (clip=1.2)."""
    return clahe_exp2.apply(img_np), mask_np