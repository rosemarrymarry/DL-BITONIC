from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
if str(PROJ / "src") not in sys.path:
    sys.path.insert(0, str(PROJ / "src"))

from src.dl_bitonic.dataset import GaussianNoisyPairDataset
from src.dl_bitonic.losses import CompositeLoss
from src.dl_bitonic.metrics import psnr, ssim
from src.dl_bitonic.model import BitonicGuidedDenoiser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnr_vals, ssim_vals = [], []
    for batch in loader:
        clean = batch["clean"].to(device)
        noisy = batch["noisy"].to(device)
        sigma_map = batch["sigma_map"].to(device)
        pred, _ = model(noisy, sigma_map)
        psnr_vals.append(psnr(pred, clean))
        ssim_vals.append(ssim(pred, clean))
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


class SimpleCNNBaseline(nn.Module):
    """A pure CNN baseline without the bitonic prior branch."""

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
    """Ablation: keep hybrid structure but disable gate fusion."""

    def __init__(self, base_channels: int = 32, use_sigma_aware_tau: bool = True) -> None:
        super().__init__()
        self.model = BitonicGuidedDenoiser(use_sigma_aware_tau=use_sigma_aware_tau, base_channels=base_channels)

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_hat, aux = self.model(y, sigma_map)
        aux["gate"] = torch.zeros_like(aux["gate"])
        # Force the branch to reduce to the CNN output for a true no-gate ablation.
        x_no_gate = aux["xd"]
        aux["xb"] = x_no_gate
        aux["tau"] = torch.full_like(aux["tau"], 0.08)
        return x_no_gate, aux


def make_model(kind: str, device: torch.device, base_ch: int = 32) -> nn.Module:
    if kind == "hybrid":
        return BitonicGuidedDenoiser(use_sigma_aware_tau=True, base_channels=base_ch).to(device)
    if kind == "cnn_only":
        return SimpleCNNBaseline(base_channels=base_ch).to(device)
    if kind == "hybrid_no_gate":
        return HybridNoGate(base_channels=base_ch, use_sigma_aware_tau=True).to(device)
    if kind == "traditional":
        return TraditionalBaseline().to(device)
    raise ValueError(f"Unsupported model type: {kind}")


class TraditionalBaseline(nn.Module):
    """A simple, non-learned smoothing baseline. This is a lightweight traditional baseline for comparison."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self._make_gaussian_kernel()

    def _make_gaussian_kernel(self) -> torch.Tensor:
        k = self.kernel_size
        coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        dist2 = xx ** 2 + yy ** 2
        kernel = torch.exp(-dist2 / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, k, k)

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        kernel = self.kernel.to(y.device, y.dtype)
        pad = self.kernel_size // 2
        x = y
        x = F.conv2d(x, kernel.expand(3, 1, self.kernel_size, self.kernel_size), padding=pad, groups=3)
        x = x.clamp(0.0, 1.0)
        aux = {
            "xb": x,
            "xd": x,
            "gate": torch.zeros((y.size(0), 1, y.size(2), y.size(3)), device=y.device, dtype=y.dtype),
            "tau": sigma_map.mean(dim=1, keepdim=True),
        }
        return x, aux


def run_training(
    model_kind: str,
    data_root: str,
    save_root: str,
    epochs: int,
    batch_size: int,
    patch_size: int,
    workers: int,
    lr: float,
    sigma_values: list[float],
    seed: int,
    base_ch: int,
    device: torch.device,
    val_batch_size: int = 1,
    max_val_samples: int | None = None,
    val_patch_size: int | None = None,
) -> dict[str, float]:
    set_seed(seed)
    out = {
        "model": model_kind,
        "epochs": epochs,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "lr": lr,
        "seed": seed,
        "sigma_values": sigma_values,
    }

    if model_kind == "traditional":
        model = make_model(model_kind, device=device, base_ch=base_ch)
        val_patch_size = patch_size if val_patch_size is None else val_patch_size
        val_set = GaussianNoisyPairDataset(
            root=data_root,
            split="val",
            patch_size=val_patch_size,
            fixed_sigma=25.0,
            train=False,
            seed=seed,
            eval_patch_size=val_patch_size,
        )
        if max_val_samples is not None:
            keep = list(range(min(len(val_set), max_val_samples)))
            val_set = torch.utils.data.Subset(val_set, keep)
        val_loader = DataLoader(
            val_set,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=max(workers // 2, 1),
            pin_memory=torch.cuda.is_available(),
        )
        val_psnr, val_ssim = validate(model, val_loader, device)
        out["val_psnr"] = float(val_psnr)
        out["val_ssim"] = float(val_ssim)
        out["loss"] = float("nan")
        save_dir = Path(save_root) / model_kind
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    model = make_model(model_kind, device=device, base_ch=base_ch)
    criterion = CompositeLoss(w_ssim=0.2, w_edge=0.05, w_cons=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_set = GaussianNoisyPairDataset(
        root=data_root,
        split="train",
        patch_size=patch_size,
        sigma_min=min(sigma_values),
        sigma_max=max(sigma_values),
        train=True,
        seed=seed,
    )
    val_patch_size = patch_size if val_patch_size is None else val_patch_size
    val_set = GaussianNoisyPairDataset(
        root=data_root,
        split="val",
        patch_size=val_patch_size,
        fixed_sigma=25.0,
        train=False,
        seed=seed,
        eval_patch_size=val_patch_size,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if max_val_samples is not None:
        keep = list(range(min(len(val_set), max_val_samples)))
        val_set = torch.utils.data.Subset(val_set, keep)

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=max(workers // 2, 1),
        pin_memory=torch.cuda.is_available(),
    )

    best_psnr = -1.0
    best_ssim = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = []
        for batch in tqdm(train_loader, desc=f"{model_kind} epoch {epoch}/{epochs}", leave=False):
            clean = batch["clean"].to(device, non_blocking=True)
            noisy = batch["noisy"].to(device, non_blocking=True)
            sigma_map = batch["sigma_map"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred, aux = model(noisy, sigma_map)
                if hasattr(model, "bitonic") and callable(model.bitonic):
                    if model_kind in {"hybrid", "hybrid_no_gate"}:
                        consistency_out = model.bitonic(pred, aux["tau"])
                    else:
                        consistency_out = pred
                else:
                    consistency_out = pred
                consistency_ref = aux.get("xb", pred)
                loss, stat = criterion(pred, clean, consistency_ref, consistency_out)

            loss.backward()
            optimizer.step()
            running_loss.append(stat["total"])

        val_psnr, val_ssim = validate(model, val_loader, device)
        best_psnr = max(best_psnr, val_psnr)
        best_ssim = max(best_ssim, val_ssim)
        epoch_loss = float(np.mean(running_loss))
        out["val_psnr"] = best_psnr
        out["val_ssim"] = best_ssim
        out["loss"] = epoch_loss
        print(f"[{model_kind}] epoch={epoch:02d} train_loss={epoch_loss:.4f} val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f}")

    save_dir = Path(save_root) / model_kind
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare denoising baselines and hybrid model")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--save-root", type=str, default="experiments")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-values", type=str, default="10,25,50,75")
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--val-patch-size", type=int, default=32, help="Crop validation images to this size to reduce bitonic memory usage")
    parser.add_argument("--max-val-samples", type=int, default=4, help="Limit eval samples to reduce VRAM spikes on Kaggle")
    parser.add_argument("--models", type=str, default="hybrid,cnn_only,hybrid_no_gate,traditional", help="Comma-separated model list")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma_values = [float(v.strip()) for v in args.sigma_values.split(",") if v.strip()]
    model_types = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {}

    for model_type in model_types:
        print(f"\n=== Running model: {model_type} ===")
        result = run_training(
            model_kind=model_type,
            data_root=args.data_root,
            save_root=args.save_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            workers=args.workers,
            lr=args.lr,
            sigma_values=sigma_values,
            seed=args.seed,
            base_ch=args.base_ch,
            device=device,
            val_batch_size=args.val_batch_size,
            max_val_samples=args.max_val_samples,
            val_patch_size=args.val_patch_size,
        )
        results[model_type] = result

    summary_path = Path(args.save_root) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")

    print("\nComparison summary:")
    for model_name, metrics in results.items():
        print(f"{model_name}: best_psnr={metrics.get('val_psnr', 'n/a')}, best_ssim={metrics.get('val_ssim', 'n/a')}, loss={metrics.get('loss', 'n/a')}")


if __name__ == "__main__":
    main()
