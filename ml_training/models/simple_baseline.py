"""
SimpleBaseline (ResNet + Deconv) for multi-view joint heatmap prediction.

Input:  (B, C, H, W) — e.g. RGB(3) + Normal(3) + Silhouette(1) = 7
Output: (B, J, H, W) — per-joint confidence heatmaps in [0, 1]

A deep ResNet-style encoder extracts global features (much deeper than
U-Net's 4-level encoder), followed by deconvolution layers that upsample
back to the input resolution.  The deeper encoder gives a much larger
receptive field and better feature hierarchy for understanding body
structure.

Reference: Xiao et al., "Simple Baselines for Human Pose Estimation
and Tracking", ECCV 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet basic block (two 3×3 convs with skip connection)."""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet bottleneck block (1×1 → 3×3 → 1×1 with skip)."""
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1,
                 downsample: nn.Module = None):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)


class SimpleBaseline(nn.Module):
    """SimpleBaseline: ResNet encoder + deconvolution head for heatmaps.

    Parameters
    ----------
    num_joints : int
        Number of output heatmap channels (one per joint).
    in_channels : int
        Number of input channels.
    layers : list[int]
        Number of blocks per ResNet stage.
        [2, 2, 2, 2] ≈ ResNet-18 depth.
        [3, 4, 6, 3] ≈ ResNet-50 depth.
    base_width : int
        Base channel width (64 for standard ResNet).
    use_bottleneck : bool
        If True, use Bottleneck blocks (4× expansion); else BasicBlock.
    num_deconv : int
        Number of deconvolution layers in the head (each upsamples 2×).
    deconv_channels : int
        Channel width of deconvolution layers.
    dropout : float
        Dropout rate applied before the final head.
    """

    def __init__(self, num_joints: int = 19, in_channels: int = 7,
                 layers: list = None, base_width: int = 64,
                 use_bottleneck: bool = False, num_deconv: int = 3,
                 deconv_channels: int = 256, dropout: float = 0.15):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]  # ResNet-18 style

        Block = Bottleneck if use_bottleneck else BasicBlock
        self.expansion = Block.expansion
        self.in_ch = base_width

        # Stem: adapt to arbitrary input channels.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # ResNet stages (each stage doubles channels, halves spatial).
        self.layer1 = self._make_layer(Block, base_width, layers[0])
        self.layer2 = self._make_layer(Block, base_width * 2, layers[1],
                                       stride=2)
        self.layer3 = self._make_layer(Block, base_width * 4, layers[2],
                                       stride=2)
        self.layer4 = self._make_layer(Block, base_width * 8, layers[3],
                                       stride=2)

        # Deconvolution head (upsamples back toward input resolution).
        encoder_out = base_width * 8 * self.expansion
        deconv_layers = []
        in_c = encoder_out
        for i in range(num_deconv):
            out_c = deconv_channels
            deconv_layers.extend([
                nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1,
                                   bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ])
            in_c = out_c
        self.deconv = nn.Sequential(*deconv_layers)

        self.drop = nn.Dropout2d(dropout)

        # Final 1×1 conv → J heatmaps.
        self.head = nn.Conv2d(deconv_channels, num_joints, 1)

    def _make_layer(self, Block, channels: int, num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Build a ResNet stage."""
        out_ch = channels * Block.expansion
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        layers = []
        if Block == Bottleneck:
            layers.append(Block(self.in_ch, channels, stride, downsample))
        else:
            layers.append(Block(self.in_ch, out_ch, stride, downsample))
        self.in_ch = out_ch

        for _ in range(1, num_blocks):
            if Block == Bottleneck:
                layers.append(Block(self.in_ch, channels))
            else:
                layers.append(Block(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        heatmaps : (B, num_joints, H, W) in [0, 1]
        """
        input_size = x.shape[2:]

        # Encoder.
        x = self.stem(x)       # stride 4
        x = self.layer1(x)     # stride 4
        x = self.layer2(x)     # stride 8
        x = self.layer3(x)     # stride 16
        x = self.layer4(x)     # stride 32

        # Decoder (deconv head).
        x = self.deconv(x)     # 3 deconvs: stride 32 → 16 → 8 → 4
        x = self.drop(x)

        # Head.
        x = self.head(x)

        # Upsample to original input resolution if needed.
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear',
                              align_corners=False)

        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
#  Convenience helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(num_joints: int = 19, in_channels: int = 7,
                base_width: int = 64, layers: list = None,
                use_bottleneck: bool = False,
                device: str = "cpu") -> SimpleBaseline:
    """Build and return a SimpleBaseline, optionally on a specific device."""
    if layers is None:
        layers = [2, 2, 2, 2]
    model = SimpleBaseline(
        num_joints=num_joints,
        in_channels=in_channels,
        layers=layers,
        base_width=base_width,
        use_bottleneck=use_bottleneck,
    )
    model = model.to(device)
    block_name = "Bottleneck" if use_bottleneck else "Basic"
    print(f"[SimpleBaseline] {count_parameters(model):,} trainable parameters  "
          f"(layers={layers}, base_width={base_width}, block={block_name}, "
          f"joints={num_joints})")
    return model


# ---------------------------------------------------------------------------
#  Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== BasicBlock (ResNet-18 style) ===")
    model = build_model(num_joints=19, in_channels=7, device="cpu")
    dummy = torch.randn(2, 7, 256, 256)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}  range=[{out.min():.4f}, {out.max():.4f}]")
    assert out.shape == (2, 19, 256, 256)
    print("OK\n")

    print("=== Bottleneck (ResNet-50 style) ===")
    model2 = build_model(num_joints=19, in_channels=7,
                         layers=[3, 4, 6, 3], use_bottleneck=True,
                         device="cpu")
    out2 = model2(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out2.shape}  range=[{out2.min():.4f}, {out2.max():.4f}]")
    assert out2.shape == (2, 19, 256, 256)
    print("OK")
