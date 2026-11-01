from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dl_bitonic.dataset import GaussianNoisyPairDataset
from src.dl_bitonic.losses import CompositeLoss
from src.dl_bitonic.metrics import psnr, ssim
from src.dl_bitonic.model import BitonicGuidedDenoiser


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train bitonic-guided denoiser")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="runs/exp")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--sigma-min", type=float, default=5.0)
    p.add_argument("--sigma-max", type=float, default=75.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--disable-sigma-aware-tau", action="store_true")
    p.add_argument("--w-ssim", type=float, default=0.2)
    p.add_argument("--w-edge", type=float, default=0.05)
    p.add_argument("--w-cons", type=float, default=0.05)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--val-patch-size", type=int, default=32, help="Reduce validation crop size to control bitonic memory use")
    p.add_argument("--max-val-samples", type=int, default=4, help="Limit validation set size to avoid VRAM spikes on Kaggle")
    p.add_argument("--kaggle-safe", action="store_true", help="Use a conservative batch/patch configuration for low-memory Kaggle runs")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
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


def main() -> None:
    args = parse_args()
    if args.kaggle_safe:
        args.batch_size = 1
        args.patch_size = min(args.patch_size, 32)
        args.val_batch_size = 1
        args.val_patch_size = min(args.val_patch_size, 32)
        args.max_val_samples = min(args.max_val_samples, 4)
        args.workers = max(1, min(args.workers, 2))
        args.epochs = max(1, min(args.epochs, 20))

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = GaussianNoisyPairDataset(
        root=args.data_root,
        split="train",
        patch_size=args.patch_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        train=True,
        seed=args.seed,
    )
    val_set = GaussianNoisyPairDataset(
        root=args.data_root,
        split="val",
        patch_size=args.val_patch_size,
        fixed_sigma=25.0,
        train=False,
        seed=args.seed,
        eval_patch_size=args.val_patch_size,
    )
    if args.max_val_samples is not None and args.max_val_samples > 0:
        val_set = torch.utils.data.Subset(val_set, list(range(min(len(val_set), args.max_val_samples))))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=max(args.workers // 2, 1),
        pin_memory=torch.cuda.is_available(),
    )

    model = BitonicGuidedDenoiser(
        use_sigma_aware_tau=not args.disable_sigma_aware_tau,
        base_channels=args.base_ch,
    ).to(device)

    criterion = CompositeLoss(w_ssim=args.w_ssim, w_edge=args.w_edge, w_cons=args.w_cons)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_psnr = -1.0
    log_path = save_dir / "train_log.jsonl"

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_psnr = float(ckpt.get("best_psnr", -1.0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
        for batch in pbar:
            clean = batch["clean"].to(device, non_blocking=True)
            noisy = batch["noisy"].to(device, non_blocking=True)
            sigma_map = batch["sigma_map"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                pred, aux = model(noisy, sigma_map)
                consistency_out = model.bitonic(pred, aux["tau"])
                # Use the bitonic prior output as the consistency reference rather than the
                # network prediction itself. This avoids collapsing the hybrid branch toward
                # its own smoothed output and preserves a meaningful structural constraint.
                loss, stat = criterion(pred, clean, aux["xb"], consistency_out)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running.append(stat["total"])
            pbar.set_postfix(loss=f"{np.mean(running):.4f}")

        scheduler.step()
        val_psnr, val_ssim = validate(model, val_loader, device)

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"[Epoch {epoch}] train_loss={rec['train_loss']:.4f} val_psnr={val_psnr:.3f} val_ssim={val_ssim:.4f}")

        latest_path = save_dir / "latest.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_psnr": best_psnr,
                "args": vars(args),
            },
            latest_path,
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = save_dir / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved best checkpoint to {best_path} (PSNR={best_psnr:.3f})")


if __name__ == "__main__":
    main()
