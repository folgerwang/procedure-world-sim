"""
Stacked Hourglass Network for multi-view joint heatmap prediction.

Input:  (B, C, H, W) — e.g. RGB(3) + Normal(3) + Silhouette(1) = 7
Output: (B, J, H, W) — per-joint confidence heatmaps in [0, 1]

The hourglass architecture processes the image through multiple stacked
hourglass modules, each producing an intermediate heatmap prediction.
Intermediate supervision at every stage gives stronger gradients with
small datasets.  The iterative refinement means each hourglass corrects
the errors of the previous one.

Reference: Newell et al., "Stacked Hourglass Networks for Human Pose
Estimation", ECCV 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """Bottleneck residual block (pre-activation style)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch // 2
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1, bias=False)

        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = F.relu(self.bn3(out))
        out = self.conv3(out)
        return out + identity


class Hourglass(nn.Module):
    """Single hourglass module with recursive encoder-decoder structure.

    depth=4 gives 4 levels of downsampling (256→128→64→32→16 at default res).
    """

    def __init__(self, depth: int, num_features: int):
        super().__init__()
        self.depth = depth
        self.nf = num_features

        # Upper branch (skip connection).
        self.up1 = Residual(num_features, num_features)

        # Lower branch: pool → residual → recurse/bottleneck → residual → upsample.
        self.pool = nn.MaxPool2d(2)
        self.low1 = Residual(num_features, num_features)

        if depth > 1:
            self.low2 = Hourglass(depth - 1, num_features)
        else:
            self.low2 = Residual(num_features, num_features)

        self.low3 = Residual(num_features, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up1 = self.up1(x)
        low = self.pool(x)
        low = self.low1(low)
        low = self.low2(low)
        low = self.low3(low)
        low = F.interpolate(low, size=up1.shape[2:], mode='bilinear',
                            align_corners=False)
        return up1 + low


class StackedHourglass(nn.Module):
    """Stacked Hourglass Network for dense joint heatmap regression.

    Parameters
    ----------
    num_joints : int
        Number of output heatmap channels (one per joint).
    in_channels : int
        Number of input channels.
    num_stacks : int
        Number of hourglass modules stacked.  More stacks = more
        refinement passes, but heavier.  2 is standard.
    num_features : int
        Feature depth inside each hourglass.
    depth : int
        Recursion depth of each hourglass (number of downsampling levels).
    """

    def __init__(self, num_joints: int = 19, in_channels: int = 7,
                 num_stacks: int = 2, num_features: int = 128,
                 depth: int = 4, dropout: float = 0.15):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_joints = num_joints

        # Initial feature extraction (brings input channels → num_features).
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual(64, 128),
            nn.MaxPool2d(2),
            Residual(128, 128),
            Residual(128, num_features),
        )

        # Stacked hourglass modules.
        self.hourglasses = nn.ModuleList()
        self.post_residuals = nn.ModuleList()
        self.head_convs = nn.ModuleList()       # 1×1 conv → heatmaps
        self.remap_convs = nn.ModuleList()      # remap heatmaps back to features
        self.merge_convs = nn.ModuleList()       # merge features for next stack
        self.dropouts = nn.ModuleList()

        for i in range(num_stacks):
            self.hourglasses.append(Hourglass(depth, num_features))
            self.post_residuals.append(nn.Sequential(
                Residual(num_features, num_features),
                nn.Conv2d(num_features, num_features, 1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ))
            self.head_convs.append(nn.Conv2d(num_features, num_joints, 1))
            self.dropouts.append(nn.Dropout2d(dropout if i < num_stacks - 1 else 0))

            if i < num_stacks - 1:
                self.remap_convs.append(
                    nn.Conv2d(num_joints, num_features, 1, bias=False))
                self.merge_convs.append(
                    nn.Conv2d(num_features, num_features, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        heatmaps : (B, num_joints, H, W) in [0, 1]
            Output from the last stack, upsampled to input resolution.
        """
        # Initial feature extraction (reduces spatial by 4×).
        feat = self.pre(x)
        input_size = x.shape[2:]  # original H, W

        outputs = []
        for i in range(self.num_stacks):
            hg_out = self.hourglasses[i](feat)
            hg_out = self.post_residuals[i](hg_out)
            hg_out = self.dropouts[i](hg_out)

            heatmaps = self.head_convs[i](hg_out)
            outputs.append(heatmaps)

            if i < self.num_stacks - 1:
                # Feed back into next stack.
                feat = feat + self.merge_convs[i](hg_out) + \
                       self.remap_convs[i](heatmaps)

        # Upsample all stack outputs to input resolution.
        for i in range(len(outputs)):
            if outputs[i].shape[2:] != input_size:
                outputs[i] = F.interpolate(outputs[i], size=input_size,
                                           mode='bilinear', align_corners=False)

        if self.training and len(outputs) > 1:
            # Return all stacks for intermediate supervision during training.
            return [torch.sigmoid(o) for o in outputs]

        # Inference: return only the final (most refined) stack.
        return torch.sigmoid(outputs[-1])


# ---------------------------------------------------------------------------
#  Convenience helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(num_joints: int = 19, in_channels: int = 7,
                num_stacks: int = 3, num_features: int = 256,
                depth: int = 4,
                device: str = "cpu") -> StackedHourglass:
    """Build and return a StackedHourglass, optionally on a specific device."""
    model = StackedHourglass(
        num_joints=num_joints,
        in_channels=in_channels,
        num_stacks=num_stacks,
        num_features=num_features,
        depth=depth,
    )
    model = model.to(device)
    print(f"[StackedHourglass] {count_parameters(model):,} trainable parameters  "
          f"(stacks={num_stacks}, features={num_features}, depth={depth}, "
          f"joints={num_joints})")
    return model


# ---------------------------------------------------------------------------
#  Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_model(num_joints=19, in_channels=7, device="cpu")
    dummy = torch.randn(2, 7, 256, 256)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}  range=[{out.min():.4f}, {out.max():.4f}]")
    assert out.shape == (2, 19, 256, 256)
    print("OK")
