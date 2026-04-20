"""
U-Net architecture for multi-view joint heatmap prediction.

Input:  (B, 7, H, W) — RGB(3) + WorldNormal(3) + Silhouette(1)
Output: (B, J, H, W) — per-joint confidence heatmaps in [0, 1]

The encoder progressively down-samples the spatial resolution while
increasing feature depth.  The decoder up-samples with skip connections
from the encoder, preserving fine spatial detail needed for precise
joint localisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock → MaxPool 2x2.  Returns both the feature map (for skip)
    and the pooled output (for the next level)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Upsample 2x → concat skip → ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd-sized feature maps.
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetHeatmap(nn.Module):
    """U-Net for dense joint heatmap regression.

    Parameters
    ----------
    num_joints : int
        Number of output heatmap channels (one per joint).
    in_channels : int
        Number of input channels (default 7: RGB + Normal + Silhouette).
    base_channels : int
        Feature depth of the first encoder level.  Subsequent levels
        double: base → 2×base → 4×base → 8×base.
    """

    def __init__(self, num_joints: int = 19, in_channels: int = 7,
                 base_channels: int = 64, dropout: float = 0.15):
        super().__init__()
        c = base_channels

        # Encoder path.
        self.enc1 = EncoderBlock(in_channels, c)       # → c
        self.enc2 = EncoderBlock(c, c * 2)              # → 2c
        self.enc3 = EncoderBlock(c * 2, c * 4)          # → 4c
        self.enc4 = EncoderBlock(c * 4, c * 8)          # → 8c

        # Bottleneck.
        self.bottleneck = ConvBlock(c * 8, c * 16)       # → 16c

        # Dropout in bottleneck + deeper decoder layers to prevent overfitting.
        self.drop_bottleneck = nn.Dropout2d(dropout)
        self.drop_dec4 = nn.Dropout2d(dropout)
        self.drop_dec3 = nn.Dropout2d(dropout * 0.5)

        # Decoder path (with skip connections).
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8,  c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4,  c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2,  c,     c)

        # Final 1x1 conv → J heatmaps.
        self.head = nn.Conv2d(c, num_joints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        heatmaps : (B, num_joints, H, W) in [0, 1]
        """
        # Encoder.
        s1, x = self.enc1(x)   # s1: (B, c, H, W)
        s2, x = self.enc2(x)   # s2: (B, 2c, H/2, W/2)
        s3, x = self.enc3(x)   # s3: (B, 4c, H/4, W/4)
        s4, x = self.enc4(x)   # s4: (B, 8c, H/8, W/8)

        # Bottleneck.
        x = self.drop_bottleneck(self.bottleneck(x))  # (B, 16c, H/16, W/16)

        # Decoder.
        x = self.drop_dec4(self.dec4(x, s4))    # (B, 8c, H/8, W/8)
        x = self.drop_dec3(self.dec3(x, s3))    # (B, 4c, H/4, W/4)
        x = self.dec2(x, s2)    # (B, 2c, H/2, W/2)
        x = self.dec1(x, s1)    # (B, c, H, W)

        # Head.
        return torch.sigmoid(self.head(x))


# ---------------------------------------------------------------------------
#  Convenience helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(num_joints: int = 19, in_channels: int = 7,
                base_channels: int = 64,
                device: str = "cpu") -> UNetHeatmap:
    """Build and return a UNetHeatmap, optionally on a specific device."""
    model = UNetHeatmap(num_joints=num_joints,
                        in_channels=in_channels,
                        base_channels=base_channels)
    model = model.to(device)
    print(f"[UNetHeatmap] {count_parameters(model):,} trainable parameters  "
          f"(base_channels={base_channels}, joints={num_joints})")
    return model


# ---------------------------------------------------------------------------
#  Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_model(num_joints=19, base_channels=64, device="cpu")
    dummy = torch.randn(2, 7, 256, 256)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}  range=[{out.min():.4f}, {out.max():.4f}]")
    assert out.shape == (2, 19, 256, 256)
    print("OK")
