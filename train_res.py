from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dl_bitonic.dataset import GaussianNoisyPairDataset
from src.dl_bitonic.losses import CompositeLoss
from src.dl_bitonic.metrics import psnr, ssim
from src.dl_bitonic.model import BitonicGuidedDenoiser


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Resume training from the best checkpoint")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="runs/exp_resume")
    p.add_argument("--resume-checkpoint", type=str, default=None, help="Path to checkpoint to resume from. If omitted, script will auto-use runs/exp/best.pt or parse train_log.jsonl.")
    p.add_argument("--epochs", type=int, default=15, help="Number of extra epochs to train")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
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
    p.add_argument("--metric", type=str, default="psnr", choices=["psnr", "ssim"], help="Metric used to choose best checkpoint")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def find_best_metric_epoch(save_dir: Path, metric: str = "psnr") -> tuple[int, float, Path]:
    log_path = save_dir / "train_log.jsonl"
    if log_path.exists():
        best_epoch = -1
        best_value = float("-inf")
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                value = rec.get(f"val_{metric}", None)
                if value is None:
                    continue
                if value > best_value:
                    best_value = float(value)
                    best_epoch = int(rec["epoch"])
        if best_epoch >= 0:
            return best_epoch, best_value, log_path

    best_pt = save_dir / "best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location="cpu")
        epoch = int(ckpt.get("epoch", 0))
        val_psnr = float(ckpt.get("best_psnr", -1.0))
        return epoch, val_psnr, best_pt

    raise FileNotFoundError(f"Could not locate a valid checkpoint or train_log.jsonl under {save_dir}")


def load_best_checkpoint(model: torch.nn.Module, save_dir: Path, metric: str = "psnr") -> tuple[Path, int, dict]:
    best_pt = save_dir / "best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        epoch = int(ckpt.get("epoch", 0))
        best_metric = float(ckpt.get("best_psnr", -1.0)) if metric == "psnr" else float(ckpt.get("best_ssim", -1.0))
        print(f"Loaded best checkpoint from {best_pt}: epoch={epoch}, val_{metric}={best_metric:.6f}")
        return best_pt, epoch, ckpt

    epoch, value, _ = find_best_metric_epoch(save_dir, metric)
    ckpt = torch.load(save_dir / "best.pt", map_location="cpu") if (save_dir / "best.pt").exists() else None
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint exists in {save_dir}")
    model.load_state_dict(ckpt["model"])
    return save_dir / "best.pt", epoch, ckpt


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume_checkpoint is None:
        resume_checkpoint = save_dir / "best.pt"
        if not resume_checkpoint.exists():
            try:
                epoch, _, _ = find_best_metric_epoch(save_dir, args.metric)
                print(f"Best epoch by {args.metric} is {epoch}. Using {save_dir / 'best.pt'} if present, otherwise you need to specify --resume-checkpoint.")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No checkpoint found in {save_dir}. "
                    "Please train once with train.py first, or pass --resume-checkpoint explicitly."
                )
    else:
        resume_checkpoint = Path(args.resume_checkpoint)

    if not resume_checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    print(f"Resume from: {resume_checkpoint}")

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
        patch_size=args.patch_size,
        fixed_sigma=25.0,
        train=False,
        seed=args.seed,
    )

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
        batch_size=1,
        shuffle=False,
        num_workers=max(args.workers // 2, 1),
        pin_memory=torch.cuda.is_available(),
    )

    model = BitonicGuidedDenoiser(
        use_sigma_aware_tau=not args.disable_sigma_aware_tau,
        base_channels=args.base_ch,
    ).to(device)

    ckpt = torch.load(resume_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    criterion = CompositeLoss(w_ssim=args.w_ssim, w_edge=args.w_edge, w_cons=args.w_cons)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_psnr = float(ckpt.get("best_psnr", -1.0))
    best_ssim = float(ckpt.get("best_ssim", -1.0)) if "best_ssim" in ckpt else -1.0

    if args.metric == "ssim":
        best_metric_for_resume = best_ssim
    else:
        best_metric_for_resume = best_psnr

    print(f"Resume epoch: {start_epoch}")
    print(f"Resume best metric ({args.metric}): {best_metric_for_resume:.6f}")

    log_path = save_dir / "train_log.jsonl"
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        running = []
        pbar = tqdm(train_loader, desc=f"Resume epoch {epoch}/{start_epoch + args.epochs - 1}", ncols=120)

        for batch in pbar:
            clean = batch["clean"].to(device, non_blocking=True)
            noisy = batch["noisy"].to(device, non_blocking=True)
            sigma_map = batch["sigma_map"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                pred, aux = model(noisy, sigma_map)
                consistency_out = model.bitonic(pred, aux["tau"])
                loss, stat = criterion(pred, clean, pred, consistency_out)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running.append(stat["total"])
            pbar.set_postfix(loss=f"{np.mean(running):.4f}")

        val_psnr, val_ssim = validate(model, val_loader, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_ssim if args.metric == "ssim" else val_psnr)

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "lr": current_lr,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"[epoch {epoch}] train_loss={rec['train_loss']:.4f} val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f} lr={current_lr:.8f}")

        latest_path = save_dir / "latest_resume.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_psnr": max(best_psnr, val_psnr),
                "best_ssim": max(best_ssim, val_ssim),
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
                    "best_ssim": max(best_ssim, val_ssim),
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved improved best checkpoint to {best_path} (PSNR={best_psnr:.4f})")

        if val_ssim > best_ssim:
            best_ssim = val_ssim

    print(f"Finished resume training. Best checkpoint is in {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
