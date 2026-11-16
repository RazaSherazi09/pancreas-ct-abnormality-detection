import torch
import torch.nn as nn
import torch.nn.functional as F
from src.handcrafted import DenseHandcraftedExtractor


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class AttentionGate(nn.Module):
    """Suppresses background retroperitoneal fat responses on skip connections."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        alpha = self.psi(psi)
        return x * alpha


class SEAttentionFusion(nn.Module):
    """
    Squeeze-and-Excitation (SE) Attention-Based Feature Fusion:
    Projects handcrafted maps, concatenates with deep encoder features,
    and dynamically recalibrates channel importance.
    """
    def __init__(self, deep_ch: int, hc_ch: int, reduction: int = 8):
        super().__init__()
        self.hc_stem = nn.Sequential(
            nn.Conv2d(hc_ch, deep_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(deep_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(deep_ch // 2, deep_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(deep_ch // 2),
            nn.ReLU(inplace=True),
        )
        total_ch = deep_ch + (deep_ch // 2)
        self.se_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total_ch, total_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(total_ch // reduction, total_ch, bias=False),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(total_ch, deep_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(deep_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, deep_feat, hc_feat):
        hc_proj = self.hc_stem(hc_feat)
        fused = torch.cat([deep_feat, hc_proj], dim=1)
        weights = self.se_fc(fused).unsqueeze(-1).unsqueeze(-1)
        recalibrated = fused * weights
        return self.out_proj(recalibrated)


class AttentionResUNet(nn.Module):
    """
    Student 1 Advanced Architecture: Attention Residual U-Net
    Supports feature_mode in {'none', 'lbp', 'gabor'}.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, feature_mode: str = "none"):
        super().__init__()
        self.feature_mode = feature_mode.lower()
        self.extractor = DenseHandcraftedExtractor(mode=self.feature_mode)
        self.use_fusion = self.extractor.out_channels > 0

        filters = [32, 64, 128, 256, 512]

        # Encoder
        self.enc1 = ResidualConvBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ResidualConvBlock(filters[3], filters[4])

        # Multi-scale SE Attention Fusion at Stage 1 & Stage 2
        if self.use_fusion:
            hc_c = self.extractor.out_channels
            self.fuse1 = SEAttentionFusion(deep_ch=filters[0], hc_ch=hc_c)
            self.fuse2 = SEAttentionFusion(deep_ch=filters[1], hc_ch=hc_c)

        # Decoder with Attention Gates
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.dec4 = ResidualConvBlock(filters[4], filters[3])

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.dec3 = ResidualConvBlock(filters[3], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.dec2 = ResidualConvBlock(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=16)
        self.dec1 = ResidualConvBlock(filters[1], filters[0])

        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        hc_map = self.extractor(x) if self.use_fusion else None

        e1 = self.enc1(x)
        if self.use_fusion:
            e1 = self.fuse1(e1, hc_map)

        e2 = self.enc2(self.pool1(e1))
        if self.use_fusion:
            hc_half = F.avg_pool2d(hc_map, kernel_size=2, stride=2)
            e2 = self.fuse2(e2, hc_half)

        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([e4_att, d4], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))

        return torch.sigmoid(self.final_conv(d1))