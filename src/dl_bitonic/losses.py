from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(channels: int, kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords = coords - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window2d = torch.outer(g, g)
    return window2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def ssim_loss(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11) -> torch.Tensor:
    c = x.shape[1]
    w = _gaussian_window(c, kernel_size=kernel_size).to(x.device, x.dtype)
    padding = kernel_size // 2

    mu_x = F.conv2d(x, w, padding=padding, groups=c)
    mu_y = F.conv2d(y, w, padding=padding, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=padding, groups=c) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    return 1.0 - ssim_map.mean()


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    return kx, ky


def edge_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, c, _, _ = x.shape
    kx, ky = _sobel_kernels(x.device, x.dtype)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)

    gx_x = F.conv2d(x, kx, padding=1, groups=c)
    gy_x = F.conv2d(x, ky, padding=1, groups=c)
    gx_y = F.conv2d(y, kx, padding=1, groups=c)
    gy_y = F.conv2d(y, ky, padding=1, groups=c)

    grad_x = torch.sqrt(gx_x * gx_x + gy_x * gy_x + 1e-6)
    grad_y = torch.sqrt(gx_y * gx_y + gy_y * gy_y + 1e-6)
    return F.l1_loss(grad_x, grad_y)


class CompositeLoss(nn.Module):
    def __init__(self, w_ssim: float = 0.2, w_edge: float = 0.05, w_cons: float = 0.05) -> None:
        super().__init__()
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.w_cons = w_cons

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        consistency_ref: torch.Tensor,
        consistency_out: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l1 = F.l1_loss(pred, target)
        lssim = ssim_loss(pred, target)
        ledge = edge_loss(pred, target)
        lcons = F.l1_loss(consistency_out, consistency_ref)

        total = l1 + self.w_ssim * lssim + self.w_edge * ledge + self.w_cons * lcons
        stat = {
            "l1": float(l1.detach().item()),
            "ssim": float(lssim.detach().item()),
            "edge": float(ledge.detach().item()),
            "cons": float(lcons.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, stat
