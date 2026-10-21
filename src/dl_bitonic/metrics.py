from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .losses import ssim_loss


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    mse = max(mse, 1e-12)
    return 10.0 * math.log10(1.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(1.0 - ssim_loss(pred, target).item())
