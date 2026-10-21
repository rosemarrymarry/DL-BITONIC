from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitonic import DifferentiableBitonicPrior, TauPredictor


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int = 10, base: int = 32) -> None:
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(base * 2, base * 4)

        self.u2 = UpBlock(base * 4, base * 2, base * 2)
        self.u1 = UpBlock(base * 2, base, base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        s2 = self.e2(self.p1(s1))
        b = self.e3(self.p2(s2))
        x = self.u2(b, s2)
        x = self.u1(x, s1)
        return x


class BitonicGuidedDenoiser(nn.Module):
    """
    Output:
      x_hat: final denoised image
      aux: dict for losses and analysis
    """

    def __init__(self, use_sigma_aware_tau: bool = True, base_channels: int = 32) -> None:
        super().__init__()
        self.use_sigma_aware_tau = use_sigma_aware_tau

        self.bitonic = DifferentiableBitonicPrior(kernel_size=5, softness=20.0)
        self.tau_predictor = TauPredictor(in_channels=4, hidden=32)

        self.backbone = TinyUNet(in_ch=10, base=base_channels)
        self.head_denoise = nn.Conv2d(base_channels, 3, 3, padding=1)
        self.head_gate = nn.Sequential(nn.Conv2d(base_channels, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.use_sigma_aware_tau:
            tau = self.tau_predictor(y, sigma_map)
        else:
            tau = torch.full_like(sigma_map, 0.08)

        xb = self.bitonic(y, tau)
        r = y - xb
        z = torch.cat([y, xb, r, sigma_map], dim=1)

        feat = self.backbone(z)
        xd = torch.sigmoid(self.head_denoise(feat))
        gate = self.head_gate(feat)

        x_hat = gate * xb + (1.0 - gate) * xd
        x_hat = x_hat.clamp(0.0, 1.0)
        aux = {"xb": xb, "xd": xd, "gate": gate, "tau": tau}
        return x_hat, aux
