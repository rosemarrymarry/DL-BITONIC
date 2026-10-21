from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dl_bitonic.dataset import GaussianNoisyPairDataset
from src.dl_bitonic.metrics import psnr, ssim
from src.dl_bitonic.model import BitonicGuidedDenoiser


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate bitonic-guided denoiser")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sigma", type=float, default=25.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--save-json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model = BitonicGuidedDenoiser(
        use_sigma_aware_tau=not bool(ckpt_args.get("disable_sigma_aware_tau", False)),
        base_channels=int(ckpt_args.get("base_ch", 32)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    ds = GaussianNoisyPairDataset(
        root=args.data_root,
        split="val",
        patch_size=128,
        fixed_sigma=args.sigma,
        train=False,
        seed=123,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers)

    psnr_vals, ssim_vals = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval sigma={args.sigma}", ncols=120):
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)
            sigma_map = batch["sigma_map"].to(device)

            pred, _ = model(noisy, sigma_map)
            psnr_vals.append(psnr(pred, clean))
            ssim_vals.append(ssim(pred, clean))

    result = {
        "sigma": args.sigma,
        "psnr_mean": float(np.mean(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "num_samples": len(psnr_vals),
    }

    print(json.dumps(result, indent=2))
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
