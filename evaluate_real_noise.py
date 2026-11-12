from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dl_bitonic.metrics import psnr, ssim
from src.dl_bitonic.model import BitonicGuidedDenoiser

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_rgb(path: str | Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def center_crop_pair(noisy: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = min(noisy.shape[1], clean.shape[1])
    w = min(noisy.shape[2], clean.shape[2])
    if h <= 0 or w <= 0:
        raise ValueError("Image size is invalid after crop.")

    noisy = noisy[:, :h, :w]
    clean = clean[:, :h, :w]
    return noisy, clean


def resize_to_max_side(img: torch.Tensor, max_side: int = 512) -> torch.Tensor:
    h, w = img.shape[1], img.shape[2]
    if max(h, w) <= max_side:
        return img

    scale = max_side / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def normalize_name(name: str) -> str:
    name = Path(name).name.lower()

    # Strip the RGB / sRGB marker before the noisy/gt prefix so names like
    # NOISY_SRGB_010.PNG and GT_SRGB_010.PNG collapse to the same key.
    for marker in ("_srgb_", "_rgb_", "_srgb", "_rgb"):
        if marker in name:
            name = name.split(marker, 1)[1]
            break

    for prefix in ("noisy_", "gt_", "noisy", "gt"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    if Path(name).suffix.lower() in VALID_SUFFIXES:
        name = Path(name).stem

    return name.replace(" ", "")


def _is_noisy_like(path: Path) -> bool:
    low = path.name.lower()
    return low.startswith("noisy") or "_noisy" in low or "noisy_" in low


def _is_clean_like(path: Path) -> bool:
    low = path.name.lower()
    return low.startswith("gt") or low.startswith("clean") or "_gt" in low or "gt_" in low or "_clean" in low or "clean_" in low


class RealNoisyPairDataset(Dataset):
    def __init__(self, noisy_root: str | Path, clean_root: str | Path, sigma: float = 25.0, max_samples: int | None = None, max_side: int = 512):
        self.noisy_root = Path(noisy_root)
        self.clean_root = Path(clean_root)
        self.sigma = float(sigma)
        self.max_side = int(max_side)

        noisy_files = sorted([
            p for p in self.noisy_root.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_SUFFIXES and (_is_noisy_like(p) or not _is_clean_like(p))
        ])
        clean_files = sorted([
            p for p in self.clean_root.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_SUFFIXES and (_is_clean_like(p) or not _is_noisy_like(p))
        ])

        clean_map: dict[str, Path] = {}
        for p in clean_files:
            key = normalize_name(p.name)
            if key not in clean_map or _is_clean_like(p) and not _is_noisy_like(clean_map[key]):
                clean_map[key] = p

        self.pairs: list[tuple[Path, Path]] = []
        for noisy_path in noisy_files:
            key = normalize_name(noisy_path.name)
            clean_path = clean_map.get(key)
            if clean_path is not None and clean_path.resolve() != noisy_path.resolve():
                self.pairs.append((noisy_path, clean_path))

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No matched noisy-clean pairs found between {self.noisy_root} and {self.clean_root}. "
                "Please ensure the two folders contain matching file names or the same relative structure."
            )

        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        noisy_path, clean_path = self.pairs[index]
        noisy = load_rgb(noisy_path)
        clean = load_rgb(clean_path)
        noisy, clean = center_crop_pair(noisy, clean)
        noisy = resize_to_max_side(noisy, self.max_side)
        clean = resize_to_max_side(clean, self.max_side)

        sigma_map = torch.full((1, noisy.shape[1], noisy.shape[2]), self.sigma / 75.0, dtype=noisy.dtype)
        return {
            "clean": clean,
            "noisy": noisy,
            "sigma_map": sigma_map,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a denoiser on real noisy-clean pairs.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint .pt file.")
    parser.add_argument("--noisy-root", type=str, default="", help="Folder containing noisy images.")
    parser.add_argument("--clean-root", type=str, default="", help="Folder containing clean references.")
    parser.add_argument("--noisy-img", type=str, default="", help="Single noisy image path for one-pair evaluation.")
    parser.add_argument("--clean-img", type=str, default="", help="Single clean image path for one-pair evaluation.")
    parser.add_argument("--sigma", type=float, default=25.0, help="Approximate noise level used to build sigma_map.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-side", type=int, default=512, help="Reduce large images to this max side to avoid bitonic OOM on Kaggle.")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save evaluation JSON.")
    return parser.parse_args()


def evaluate_single_pair(model: torch.nn.Module, noisy_img: str | Path, clean_img: str | Path, sigma: float, device: torch.device, max_side: int = 512) -> dict[str, float]:
    noisy = load_rgb(noisy_img)
    clean = load_rgb(clean_img)
    noisy, clean = center_crop_pair(noisy, clean)
    noisy = resize_to_max_side(noisy, max_side)
    clean = resize_to_max_side(clean, max_side)

    sigma_map = torch.full((1, noisy.shape[1], noisy.shape[2]), sigma / 75.0, dtype=noisy.dtype)
    noisy = noisy.unsqueeze(0).to(device)
    clean = clean.unsqueeze(0).to(device)
    sigma_map = sigma_map.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred, _ = model(noisy, sigma_map)

    return {
        "psnr": float(psnr(pred, clean)),
        "ssim": float(ssim(pred, clean)),
    }


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

    if args.noisy_img and args.clean_img:
        result = evaluate_single_pair(model, args.noisy_img, args.clean_img, args.sigma, device, max_side=args.max_side)
        print(json.dumps({"single_pair": True, **result}, indent=2))
        if args.save_json:
            Path(args.save_json).write_text(json.dumps({"single_pair": True, **result}, indent=2), encoding="utf-8")
        return

    if not args.noisy_root or not args.clean_root:
        raise ValueError("Please provide either --noisy-img/--clean-img or both --noisy-root/--clean-root.")

    dataset = RealNoisyPairDataset(
        noisy_root=args.noisy_root,
        clean_root=args.clean_root,
        sigma=args.sigma,
        max_samples=args.max_samples,
        max_side=args.max_side,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    all_psnr = []
    all_ssim = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating real-noise pairs", ncols=120):
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)
            sigma_map = batch["sigma_map"].to(device)

            pred, _ = model(noisy, sigma_map)
            all_psnr.append(psnr(pred, clean))
            all_ssim.append(ssim(pred, clean))

    result = {
        "sigma": args.sigma,
        "num_samples": len(all_psnr),
        "psnr_mean": float(np.mean(all_psnr)),
        "ssim_mean": float(np.mean(all_ssim)),
    }
    print(json.dumps(result, indent=2))
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
