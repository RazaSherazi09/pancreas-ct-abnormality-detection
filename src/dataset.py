import torch
from torch.utils.data import Dataset

class CachedPancreasDataset(Dataset):
    """
    Zero-disk I/O dataset. Slices are served directly from RAM tensors.
    """
    def __init__(self, images_tensor, masks_tensor, transform_fn=None):
        self.images = images_tensor  # (N, 1, 256, 256) uint8
        self.masks = masks_tensor    # (N, 1, 256, 256) uint8
        self.transform_fn = transform_fn

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        # Apply transformation if specified (for CLAHE / Sharpening experiments)
        if self.transform_fn is not None:
            img_np = img.squeeze(0).numpy()
            img_transformed = self.transform_fn(img_np)
            img = torch.from_numpy(img_transformed).unsqueeze(0)

        # Normalize to float32 [0.0, 1.0]
        img_float = img.float() / 255.0
        mask_float = mask.float()

        return img_float, mask_float