import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseHandcraftedExtractor(nn.Module):
    """
    GPU-accelerated, FP32-enforced dense spatial handcrafted feature extractors.
    Outputs pixel-aligned (B, C, H, W) feature maps normalized per slice.
    """

    def __init__(self, mode: str = "none"):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in {"none", "lbp", "gabor", "hog", "log"}, f"Unsupported mode: {mode}"

        if self.mode == "lbp":
            self.out_channels = 1
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, 1), (1, 1), (1, 0),
                (1, -1), (0, -1)
            ]
            kernels = torch.zeros((8, 1, 3, 3), dtype=torch.float32)
            for idx, (dy, dx) in enumerate(offsets):
                kernels[idx, 0, 1 + dy, 1 + dx] = 1.0
                kernels[idx, 0, 1, 1] = -1.0
            weights = torch.tensor([2 ** i for i in range(8)], dtype=torch.float32).view(1, 8, 1, 1)
            self.register_buffer("lbp_kernels", kernels)
            self.register_buffer("lbp_weights", weights)

        elif self.mode == "gabor":
            self.out_channels = 4
            kernels = self._build_gabor_bank(
                ksize=9,
                sigma=2.0,
                lambd=5.0,
                gamma=0.5,
                thetas=[0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
            )
            self.register_buffer("gabor_kernels", kernels)

        elif self.mode == "hog":
            self.out_channels = 4
            sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                    [-2.0, 0.0, 2.0],
                                    [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                    [0.0,  0.0,  0.0],
                                    [1.0,  2.0,  1.0]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)

        elif self.mode == "log":
            self.out_channels = 3
            sigmas = [1.0, 2.0, 4.0]
            self.sigmas = sigmas
            for idx, s in enumerate(sigmas):
                k = self._build_log_kernel(sigma=s)
                self.register_buffer(f"log_kernel_{idx}", k)

        else:
            self.out_channels = 0

    @staticmethod
    def _build_gabor_bank(ksize: int, sigma: float, lambd: float, gamma: float, thetas: list):
        half = ksize // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing="ij"
        )
        filters = []
        for theta in thetas:
            x_theta = x * math.cos(theta) + y * math.sin(theta)
            y_theta = -x * math.sin(theta) + y * math.cos(theta)
            gb = torch.exp(-0.5 * (x_theta ** 2 + (gamma ** 2) * (y_theta ** 2)) / (sigma ** 2))
            gb = gb * torch.cos(2.0 * math.pi * x_theta / lambd)
            gb = gb - gb.mean()
            gb = gb / (gb.norm() + 1e-6)
            filters.append(gb.view(1, 1, ksize, ksize))
        return torch.cat(filters, dim=0)

    @staticmethod
    def _build_log_kernel(sigma: float):
        ksize = int(2 * math.ceil(3 * sigma) + 1)
        half = ksize // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing="ij"
        )
        r2 = x ** 2 + y ** 2
        sigma2 = sigma ** 2
        norm = -1.0 / (math.pi * (sigma2 ** 2))
        log_k = norm * (1.0 - r2 / (2.0 * sigma2)) * torch.exp(-r2 / (2.0 * sigma2))
        log_k = log_k - log_k.mean()
        log_k = log_k * sigma2
        return log_k.view(1, 1, ksize, ksize)

    @staticmethod
    def _normalize_per_slice(feat: torch.Tensor) -> torch.Tensor:
        feat = feat.float()
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = torch.clamp(feat.std(dim=(2, 3), keepdim=True), min=1e-5)
        return (feat - mean) / std

    def forward(self, x: torch.Tensor):
        if self.mode == "none":
            return None

        x = x.float()

        if self.mode == "lbp":
            diff = F.conv2d(x, self.lbp_kernels, padding=1)
            bits = (diff >= 0).float()
            lbp_map = (bits * self.lbp_weights).sum(dim=1, keepdim=True) / 255.0
            return self._normalize_per_slice(lbp_map)

        if self.mode == "gabor":
            pad = self.gabor_kernels.shape[-1] // 2
            resp = torch.abs(F.conv2d(x, self.gabor_kernels, padding=pad))
            return self._normalize_per_slice(resp)

        if self.mode == "hog":
            gx = F.conv2d(x, self.sobel_x, padding=1)
            gy = F.conv2d(x, self.sobel_y, padding=1)
            mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
            c0 = torch.abs(gx)
            c45 = torch.abs(gx + gy) / math.sqrt(2.0)
            c90 = torch.abs(gy)
            c135 = torch.abs(gx - gy) / math.sqrt(2.0)
            hog_raw = torch.cat([c0, c45, c90, c135], dim=1)
            hog_smooth = F.avg_pool2d(hog_raw, kernel_size=5, stride=1, padding=2)
            hog_norm = hog_smooth / (mag + 1e-4)
            return self._normalize_per_slice(hog_norm)

        if self.mode == "log":
            outputs = []
            for idx in range(len(self.sigmas)):
                k = getattr(self, f"log_kernel_{idx}")
                pad = k.shape[-1] // 2
                outputs.append(F.conv2d(x, k, padding=pad))
            log_map = torch.cat(outputs, dim=1)
            return self._normalize_per_slice(log_map)