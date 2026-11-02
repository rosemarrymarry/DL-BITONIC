from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.dl_bitonic.dataset import GaussianNoisyPairDataset
from src.dl_bitonic.metrics import psnr, ssim
from src.dl_bitonic.model import BitonicGuidedDenoiser


class SimpleCNNBaseline(nn.Module):
    """Pure CNN denoiser without the bitonic prior branch."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(4, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat([y, sigma_map], dim=1)
        pred = self.backbone(x)
        pred = pred.clamp(0.0, 1.0)
        aux = {
            "xb": y,
            "xd": pred,
            "gate": torch.zeros((y.size(0), 1, y.size(2), y.size(3)), device=y.device, dtype=y.dtype),
            "tau": sigma_map.mean(dim=1, keepdim=True),
        }
        return pred, aux


class HybridNoGate(nn.Module):
    """Ablation: keep the hybrid structure but disable gate fusion."""

    def __init__(self, base_channels: int = 32, use_sigma_aware_tau: bool = True) -> None:
        super().__init__()
        self.model = BitonicGuidedDenoiser(use_sigma_aware_tau=use_sigma_aware_tau, base_channels=base_channels)

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_hat, aux = self.model(y, sigma_map)
        aux["gate"] = torch.zeros_like(aux["gate"])
        x_no_gate = aux["xd"]
        aux["xb"] = x_no_gate
        aux["tau"] = torch.full_like(aux["tau"], 0.08)
        return x_no_gate, aux


class BitonicFilterReference(nn.Module):
    """Non-learned traditional bitonic-like smoothing baseline.

    This is not a trainable model. It is only used as a reference baseline to compare
    with the learned hybrid model and ablations.
    """

    def __init__(self, kernel_size: int = 5, sigma_spatial: float = 1.5, sigma_intensity: float = 0.08) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_spatial = sigma_spatial
        self.sigma_intensity = sigma_intensity

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = y
        pad = self.kernel_size // 2
        y_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        win = F.unfold(y_pad, kernel_size=self.kernel_size, padding=0)
        win = win.view(x.size(0), x.size(1), self.kernel_size * self.kernel_size, x.size(2), x.size(3))

        center = x.unsqueeze(2)
        diff = win - center
        intensity = torch.exp(-(diff.abs() ** 2) / (2.0 * self.sigma_intensity ** 2))

        yy, xx = torch.meshgrid(
            torch.arange(self.kernel_size, device=x.device, dtype=x.dtype),
            torch.arange(self.kernel_size, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        spatial = ((xx - (self.kernel_size - 1) / 2.0) ** 2 + (yy - (self.kernel_size - 1) / 2.0) ** 2)
        spatial = torch.exp(-(spatial / (2.0 * self.sigma_spatial ** 2)))
        spatial = spatial.view(1, 1, self.kernel_size * self.kernel_size, 1, 1)

        weights = spatial * intensity
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
        out = (weights * win).sum(dim=2, keepdim=False)
        out = out.clamp(0.0, 1.0)

        aux = {
            "xb": out,
            "xd": out,
            "gate": torch.zeros((y.size(0), 1, y.size(2), y.size(3)), device=y.device, dtype=y.dtype),
            "tau": sigma_map.mean(dim=1, keepdim=True),
        }
        return out, aux


def build_model(kind: str, base_ch: int = 32) -> nn.Module:
    if kind == "hybrid":
        return BitonicGuidedDenoiser(use_sigma_aware_tau=True, base_channels=base_ch)
    if kind == "cnn_only":
        return SimpleCNNBaseline(base_channels=base_ch)
    if kind == "hybrid_no_gate":
        return HybridNoGate(base_channels=base_ch, use_sigma_aware_tau=True)
    if kind == "traditional":
        return BitonicFilterReference(kernel_size=5, sigma_spatial=1.5, sigma_intensity=0.08)
    raise ValueError(f"Unsupported model kind: {kind}")


def load_checkpoint(model: nn.Module, ckpt_path: str | None) -> nn.Module:
    if ckpt_path is None:
        return model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    if isinstance(state_dict, dict) and any(k.startswith("backbone") or k.startswith("model") for k in state_dict.keys()):
        model.load_state_dict(state_dict, strict=False)
    elif isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    return model


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnr_vals, ssim_vals = [], []
    for batch in loader:
        clean = batch["clean"].to(device)
        noisy = batch["noisy"].to(device)
        sigma_map = batch["sigma_map"].to(device)

        pred, _ = model(noisy, sigma_map)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        clean = clean.clamp(0.0, 1.0)
        psnr_vals.append(psnr(pred, clean))
        ssim_vals.append(ssim(pred, clean))

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def build_loader(data_root: str, val_patch_size: int, val_batch_size: int, max_val_samples: int, workers: int, seed: int) -> DataLoader:
    val_set = GaussianNoisyPairDataset(
        root=data_root,
        split="val",
        patch_size=val_patch_size,
        fixed_sigma=25.0,
        train=False,
        seed=seed,
        eval_patch_size=val_patch_size,
    )
    if max_val_samples is not None and max_val_samples > 0:
        val_set = torch.utils.data.Subset(val_set, list(range(min(len(val_set), max_val_samples))))
    return DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=max(1, workers // 2),
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hybrid CNN, hybrid_no_gate, and traditional reference baseline")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--val-patch-size", type=int, default=32)
    parser.add_argument("--max-val-samples", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hybrid-ckpt", type=str, default=None)
    parser.add_argument("--cnn-only-ckpt", type=str, default=None)
    parser.add_argument("--hybrid-no-gate-ckpt", type=str, default=None)
    parser.add_argument("--models", type=str, default="hybrid,cnn_only,hybrid_no_gate,traditional")
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(
        data_root=args.data_root,
        val_patch_size=args.val_patch_size,
        val_batch_size=args.val_batch_size,
        max_val_samples=args.max_val_samples,
        workers=args.workers,
        seed=args.seed,
    )

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    results: dict[str, dict[str, float | str]] = {}

    for model_name in model_names:
        model = build_model(model_name, base_ch=args.base_ch).to(device)
        if model_name == "hybrid":
            model = load_checkpoint(model, args.hybrid_ckpt)
        elif model_name == "cnn_only":
            model = load_checkpoint(model, args.cnn_only_ckpt)
        elif model_name == "hybrid_no_gate":
            model = load_checkpoint(model, args.hybrid_no_gate_ckpt)
        else:
            # Traditional reference is non-learned, so no checkpoint is required.
            pass

        psnr_val, ssim_val = evaluate_model(model, loader, device)
        results[model_name] = {
            "best_psnr": float(psnr_val),
            "best_ssim": float(ssim_val),
            "loss": float("nan") if model_name == "traditional" else float("nan"),
        }

    print("Comparison summary:")
    for model_name in model_names:
        metrics = results[model_name]
        print(
            f"{model_name}: best_psnr={metrics['best_psnr']}, "
            f"best_ssim={metrics['best_ssim']}, loss={metrics['loss']}"
        )

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved comparison summary to {save_path}")


if __name__ == "__main__":
    main()
