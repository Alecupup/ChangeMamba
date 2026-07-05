import torch
import torch.nn as nn


class ChangeGuidedSparseSpatialGate(nn.Module):
    """Soft spatial gate generated from bi-temporal feature differences."""

    def __init__(self, channels, reduction=4, alpha_init=0.5, learnable_alpha=True):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.mask_generator = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        alpha = torch.tensor(float(alpha_init))
        if learnable_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

    def forward(self, feat_t1, feat_t2):
        diff = torch.abs(feat_t1 - feat_t2)
        change_mask = torch.sigmoid(self.mask_generator(diff))
        scale = 1.0 + self.alpha * change_mask
        return feat_t1 * scale, feat_t2 * scale, change_mask
