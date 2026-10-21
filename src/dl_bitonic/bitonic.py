from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableBitonicPrior(nn.Module):
    """
    Differentiable local adaptive bitonic-like filter.

    It computes a robust weighted local average where weights depend on:
    1) spatial Gaussian distance,
    2) intensity distance to center pixel with adaptive threshold tau.
    """

    def __init__(self, kernel_size: int = 5, softness: float = 20.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.softness = softness

        coords = torch.arange(kernel_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        center = (kernel_size - 1) / 2.0
        dist2 = (xx - center) ** 2 + (yy - center) ** 2
        spatial = torch.exp(-dist2 / (2.0 * (0.35 * kernel_size) ** 2))
        spatial = spatial / spatial.sum()
        self.register_buffer("spatial_weights", spatial.view(1, 1, -1, 1), persistent=False)

    def forward(self, x: torch.Tensor, tau_map: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] in [0, 1]
        tau_map: [B, 1, H, W] adaptive threshold in [0, 1]
        """
        b, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.pad)
        patches = patches.view(b, c, self.kernel_size * self.kernel_size, h * w)

        center_idx = (self.kernel_size * self.kernel_size) // 2
        center = patches[:, :, center_idx : center_idx + 1, :]
        abs_diff = (patches - center).abs()

        tau = tau_map.view(b, 1, 1, h * w).clamp(1e-4, 1.0)
        range_w = torch.sigmoid((tau - abs_diff) * self.softness)
        spatial_w = self.spatial_weights
        weights = (range_w * spatial_w).clamp_min(1e-8)

        weighted = (patches * weights).sum(dim=2)
        norm = weights.sum(dim=2)
        out = weighted / norm
        return out.view(b, c, h, w)


class TauPredictor(nn.Module):
    """Predict adaptive bitonic threshold from image and sigma map."""

    def __init__(self, in_channels: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, sigma_map: torch.Tensor) -> torch.Tensor:
        tau = self.net(torch.cat([x, sigma_map], dim=1))
        # Keep useful dynamic range, avoid too small tau.
        return 0.02 + 0.25 * tau
