import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MultiScaleEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        mid_c = out_c // 2
        # Branch 1
        self.b1_conv1 = ConvBlock(in_c, mid_c, kernel_size=3, padding=1)
        self.b1_conv2 = ConvBlock(mid_c, mid_c, kernel_size=3, padding=1)
        # Branch 2
        self.b2_conv1x1 = ConvBlock(in_c, mid_c // 2, kernel_size=1, padding=0)
        self.b2_conv5x5 = ConvBlock(in_c, mid_c // 2, kernel_size=5, padding=2)
        self.b2_fusion = ConvBlock(mid_c, mid_c, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        c1 = self.b1_conv2(self.b1_conv1(x))
        c2_1 = self.b2_conv1x1(x)
        c2_2 = self.b2_conv5x5(x)
        c2 = self.b2_fusion(torch.cat([c2_1, c2_2], dim=1))
        pooled = self.pool(torch.cat([c1, c2], dim=1))
        return pooled, c1, c2

class MultiScaleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = MultiScaleEncoderBlock(in_channels, 64)
        self.enc2 = MultiScaleEncoderBlock(64, 128)
        self.enc3 = MultiScaleEncoderBlock(128, 256)
        self.enc4 = MultiScaleEncoderBlock(256, 512)

        self.bottleneck = nn.Sequential(
            ConvBlock(512, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1)
        )

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            ConvBlock(256 + 256 + 256, 256),
            ConvBlock(256, 256)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(128 + 128 + 128, 128),
            ConvBlock(128, 128)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(64 + 64 + 64, 64),
            ConvBlock(64, 64)
        )

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(32 + 32 + 32, 32),
            ConvBlock(32, 32)
        )

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p1, c1_1, c1_2 = self.enc1(x)
        p2, c2_1, c2_2 = self.enc2(p1)
        p3, c3_1, c3_2 = self.enc3(p2)
        p4, c4_1, c4_2 = self.enc4(p3)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, c4_1, c4_2], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, c3_1, c3_2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, c2_1, c2_2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, c1_1, c1_2], dim=1))

        return self.sigmoid(self.final_conv(d1))