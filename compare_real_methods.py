from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from src.dl_bitonic.metrics import psnr, ssim
from src.dl_bitonic.model import BitonicGuidedDenoiser

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_rgb(path: str | Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


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


def center_crop_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = min(a.shape[1], b.shape[1])
    w = min(a.shape[2], b.shape[2])
    if h <= 0 or w <= 0:
        raise ValueError("Image size is invalid after crop.")
    return a[:, :h, :w], b[:, :h, :w]


def normalize_name(name: str) -> str:
    s = Path(name).name.lower()
    for marker in ("_srgb_", "_rgb_", "_srgb", "_rgb"):
        if marker in s:
            s = s.split(marker, 1)[1]
            break
    for prefix in ("noisy_", "gt_", "clean_", "noisy", "gt", "clean"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if Path(s).suffix.lower() in VALID_SUFFIXES:
        s = Path(s).stem
    return s.replace(" ", "")


class SimpleCNNBaseline(nn.Module):
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
        pred = self.backbone(torch.cat([y, sigma_map], dim=1)).clamp(0.0, 1.0)
        aux = {
            "xb": y,
            "xd": pred,
            "gate": torch.zeros((y.size(0), 1, y.size(2), y.size(3)), device=y.device, dtype=y.dtype),
            "tau": sigma_map.mean(dim=1, keepdim=True),
        }
        return pred, aux


class HybridNoGate(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.model = BitonicGuidedDenoiser(use_sigma_aware_tau=True, base_channels=base_channels)

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, aux = self.model(y, sigma_map)
        aux["gate"] = torch.zeros_like(aux["gate"])
        x_no_gate = aux["xd"].clamp(0.0, 1.0)
        aux["xb"] = x_no_gate
        return x_no_gate, aux


def load_model(model_kind: str, checkpoint: Path, base_ch: int, device: torch.device) -> nn.Module:
    if model_kind == "hybrid":
        model = BitonicGuidedDenoiser(use_sigma_aware_tau=True, base_channels=base_ch)
    elif model_kind == "cnn_only":
        model = SimpleCNNBaseline(base_channels=base_ch)
    elif model_kind == "hybrid_no_gate":
        model = HybridNoGate(base_channels=base_ch)
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


@dataclass
class PairSample:
    key: str
    noisy_path: Path
    clean_path: Path


def build_pair_list(noisy_root: Path, clean_root: Path, max_samples: int | None = None) -> list[PairSample]:
    noisy_files = [p for p in noisy_root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]
    clean_files = [p for p in clean_root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]

    noisy_map = {normalize_name(p.name): p for p in noisy_files}
    clean_map = {normalize_name(p.name): p for p in clean_files}

    common_keys = sorted(set(noisy_map.keys()).intersection(clean_map.keys()))
    pairs: list[PairSample] = []
    for key in common_keys:
        noisy_path = noisy_map[key]
        clean_path = clean_map[key]
        if noisy_path.resolve() == clean_path.resolve():
            continue
        pairs.append(PairSample(key=key, noisy_path=noisy_path, clean_path=clean_path))

    if max_samples is not None:
        pairs = pairs[:max_samples]
    if not pairs:
        raise FileNotFoundError("No matched noisy-clean pairs were found.")
    return pairs


def parse_method(spec: str) -> tuple[str, str, str]:
    # Format: name=type:path
    # type in {pred, hybrid_ckpt, cnn_only_ckpt, hybrid_no_gate_ckpt}
    if "=" not in spec or ":" not in spec:
        raise ValueError(
            "Invalid --method format. Use name=type:path, e.g. "
            "restormer=pred:/kaggle/working/restormer_outputs"
        )
    name, rhs = spec.split("=", 1)
    mtype, payload = rhs.split(":", 1)
    name = name.strip()
    mtype = mtype.strip()
    payload = payload.strip()
    if not name or not mtype or not payload:
        raise ValueError(f"Invalid --method spec: {spec}")
    return name, mtype, payload


@torch.no_grad()
def infer_with_model(
    model: nn.Module,
    noisy: torch.Tensor,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    sigma_map = torch.full((1, 1, noisy.shape[1], noisy.shape[2]), sigma / 75.0, dtype=noisy.dtype, device=device)
    pred, _ = model(noisy.unsqueeze(0).to(device), sigma_map)
    return pred.squeeze(0).detach().cpu().clamp(0.0, 1.0)


def evaluate_method_on_pairs(
    name: str,
    mtype: str,
    payload: str,
    pairs: list[PairSample],
    sigma: float,
    max_side: int,
    base_ch: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    pred_map: dict[str, Path] = {}
    model = None

    if mtype == "pred":
        pred_root = Path(payload)
        pred_files = [p for p in pred_root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]
        pred_map = {normalize_name(p.name): p for p in pred_files}
    elif mtype in {"hybrid_ckpt", "cnn_only_ckpt", "hybrid_no_gate_ckpt"}:
        kind = mtype.replace("_ckpt", "")
        model = load_model(kind, Path(payload), base_ch=base_ch, device=device)
    else:
        raise ValueError(f"Unsupported method type: {mtype}")

    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    matched = 0

    for sample in tqdm(pairs, desc=f"{name}", ncols=120):
        clean = load_rgb(sample.clean_path)
        clean = resize_to_max_side(clean, max_side=max_side)

        if mtype == "pred":
            pred_path = pred_map.get(sample.key)
            if pred_path is None:
                continue
            pred = load_rgb(pred_path)
            pred = resize_to_max_side(pred, max_side=max_side)
        else:
            noisy = load_rgb(sample.noisy_path)
            noisy = resize_to_max_side(noisy, max_side=max_side)
            pred = infer_with_model(model, noisy, sigma=sigma, device=device)

        pred, clean = center_crop_pair(pred, clean)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        clean = clean.clamp(0.0, 1.0)

        psnr_vals.append(float(psnr(pred.unsqueeze(0), clean.unsqueeze(0))))
        ssim_vals.append(float(ssim(pred.unsqueeze(0), clean.unsqueeze(0))))
        matched += 1

    if matched == 0:
        raise RuntimeError(f"Method {name} has no matched predictions for evaluation.")

    return {
        "method": name,
        "type": mtype,
        "num_samples": matched,
        "psnr_mean": float(np.mean(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare denoising methods on real noisy-clean pairs.")
    parser.add_argument("--noisy-root", type=str, required=True, help="Folder containing noisy images.")
    parser.add_argument("--clean-root", type=str, required=True, help="Folder containing clean references.")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help=(
            "Method spec, repeatable. Format: name=type:path. "
            "type: pred | hybrid_ckpt | cnn_only_ckpt | hybrid_no_gate_ckpt"
        ),
    )
    parser.add_argument("--sigma", type=float, default=25.0, help="Used only when a method is loaded from checkpoint.")
    parser.add_argument("--max-side", type=int, default=512, help="Resize max image side to avoid OOM.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for quick benchmark.")
    parser.add_argument("--base-ch", type=int, default=32, help="Base channels for in-house model loading.")
    parser.add_argument("--save-json", type=str, default="", help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.method:
        raise ValueError("At least one --method is required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = build_pair_list(Path(args.noisy_root), Path(args.clean_root), max_samples=args.max_samples)

    results: list[dict[str, float | int | str]] = []
    for spec in args.method:
        name, mtype, payload = parse_method(spec)
        result = evaluate_method_on_pairs(
            name=name,
            mtype=mtype,
            payload=payload,
            pairs=pairs,
            sigma=args.sigma,
            max_side=args.max_side,
            base_ch=args.base_ch,
            device=device,
        )
        results.append(result)

    results.sort(key=lambda r: float(r["psnr_mean"]), reverse=True)
    summary = {
        "num_pairs": len(pairs),
        "sigma_for_ckpt_methods": args.sigma,
        "results": results,
    }
    print(json.dumps(summary, indent=2))

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()