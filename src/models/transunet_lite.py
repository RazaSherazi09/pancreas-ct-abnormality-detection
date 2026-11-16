import torch
import torch.nn as nn
import torch.nn.functional as F
from src.handcrafted import DenseHandcraftedExtractor


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ProjectionConcatFusion(nn.Module):
    """
    Projection & Dimensionality Alignment followed by Channel Concatenation Fusion:
    Aligns dense handcrafted features via 1x1 projection and fuses with CNN encoder features.
    """
    def __init__(self, deep_ch: int, hc_ch: int):
        super().__init__()
        proj_ch = deep_ch // 2
        self.hc_proj = nn.Sequential(
            nn.Conv2d(hc_ch, proj_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_ch),
            nn.GELU(),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(deep_ch + proj_ch, deep_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(deep_ch),
            nn.GELU(),
        )

    def forward(self, deep_feat, hc_feat):
        hc_aligned = self.hc_proj(hc_feat)
        concat = torch.cat([deep_feat, hc_aligned], dim=1)
        return self.fusion_conv(concat)


class TransformerBottleneck(nn.Module):
    """
    2D Spatial Vision Transformer Bottleneck:
    Models global long-range dependencies across abdominal organs at 16x16 resolution.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 3, grid_size: int = 16):
        super().__init__()
        num_tokens = grid_size * grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = tokens + self.pos_embed
        out_tokens = self.transformer(tokens)
        return out_tokens.transpose(1, 2).reshape(B, C, H, W)


class TransUNetLite(nn.Module):
    """
    Student 2 Advanced Architecture: Hybrid CNN-Transformer Segmentation Network
    Supports feature_mode in {'none', 'hog', 'log'}.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, feature_mode: str = "none"):
        super().__init__()
        self.feature_mode = feature_mode.lower()
        self.extractor = DenseHandcraftedExtractor(mode=self.feature_mode)
        self.use_fusion = self.extractor.out_channels > 0

        dims = [32, 64, 128, 256]

        # CNN Encoder (256 -> 128 -> 64 -> 32 -> 16)
        self.enc1 = ConvBNReLU(in_channels, dims[0])
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBNReLU(dims[0], dims[1])
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBNReLU(dims[1], dims[2])
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBNReLU(dims[2], dims[3])
        self.pool4 = nn.MaxPool2d(2)

        if self.use_fusion:
            hc_c = self.extractor.out_channels
            self.fuse1 = ProjectionConcatFusion(deep_ch=dims[0], hc_ch=hc_c)
            self.fuse3 = ProjectionConcatFusion(deep_ch=dims[2], hc_ch=hc_c)

        # Global Self-Attention Bottleneck at 16x16
        self.vit_bottleneck = TransformerBottleneck(embed_dim=dims[3], num_heads=8, num_layers=3, grid_size=16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(dims[3], dims[3], kernel_size=2, stride=2)
        self.dec4 = ConvBNReLU(dims[3] + dims[3], dims[2])

        self.up3 = nn.ConvTranspose2d(dims[2], dims[2], kernel_size=2, stride=2)
        self.dec3 = ConvBNReLU(dims[2] + dims[2], dims[1])

        self.up2 = nn.ConvTranspose2d(dims[1], dims[1], kernel_size=2, stride=2)
        self.dec2 = ConvBNReLU(dims[1] + dims[1], dims[0])

        self.up1 = nn.ConvTranspose2d(dims[0], dims[0], kernel_size=2, stride=2)
        self.dec1 = ConvBNReLU(dims[0] + dims[0], dims[0])

        self.head = nn.Conv2d(dims[0], out_channels, kernel_size=1)

    def forward(self, x):
        hc_map = self.extractor(x) if self.use_fusion else None

        e1 = self.enc1(x)
        if self.use_fusion:
            e1 = self.fuse1(e1, hc_map)

        e2 = self.enc2(self.pool1(e1))

        e3 = self.enc3(self.pool2(e2))
        if self.use_fusion:
            hc_quarter = F.avg_pool2d(hc_map, kernel_size=4, stride=4)
            e3 = self.fuse3(e3, hc_quarter)

        e4 = self.enc4(self.pool3(e3))
        b = self.vit_bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))