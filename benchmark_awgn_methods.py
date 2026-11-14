from __future__ import annotations

import argparse
import csv
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


def load_img(path: Path) -> torch.Tensor:
    img = Image.open(path)
    if img.mode == "L":
        arr = np.asarray(img, dtype=np.float32)[..., None] / 255.0
    else:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def to_rgb3(x: torch.Tensor) -> torch.Tensor:
    if x.size(0) == 3:
        return x
    return x.repeat(3, 1, 1)


def align_pred_to_clean(pred: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = min(pred.shape[1], clean.shape[1])
    w = min(pred.shape[2], clean.shape[2])
    pred = pred[:, :h, :w]
    clean = clean[:, :h, :w]

    if clean.size(0) == 1 and pred.size(0) == 3:
        pred = pred.mean(dim=0, keepdim=True)
    elif clean.size(0) == 3 and pred.size(0) == 1:
        pred = pred.repeat(3, 1, 1)

    return pred, clean


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


def build_model(kind: str, ckpt_path: Path, base_ch: int, device: torch.device) -> nn.Module:
    if kind == "hybrid_ckpt":
        model = BitonicGuidedDenoiser(use_sigma_aware_tau=True, base_channels=base_ch)
    elif kind == "cnn_only_ckpt":
        model = SimpleCNNBaseline(base_channels=base_ch)
    elif kind == "hybrid_no_gate_ckpt":
        model = HybridNoGate(base_channels=base_ch)
    else:
        raise ValueError(f"Unsupported checkpoint method type: {kind}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def infer_model(model: nn.Module, noisy: torch.Tensor, sigma: int, device: torch.device) -> torch.Tensor:
    noisy_rgb = to_rgb3(noisy).unsqueeze(0).to(device)
    sigma_map = torch.full((1, 1, noisy_rgb.shape[2], noisy_rgb.shape[3]), float(sigma) / 75.0, device=device)
    with torch.no_grad():
        pred, _ = model(noisy_rgb, sigma_map)
    pred = pred.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    if noisy.size(0) == 1:
        pred = pred.mean(dim=0, keepdim=True)
    return pred


@dataclass
class MethodSpec:
    name: str
    mtype: str
    payload: Path


def parse_method(spec: str) -> MethodSpec:
    if "=" not in spec or ":" not in spec:
        raise ValueError(
            f"Invalid --method: {spec}. Use name=type:path, type in pred|hybrid_ckpt|cnn_only_ckpt|hybrid_no_gate_ckpt"
        )
    name, rhs = spec.split("=", 1)
    mtype, payload = rhs.split(":", 1)
    return MethodSpec(name=name.strip(), mtype=mtype.strip(), payload=Path(payload.strip()))


def list_datasets(bench_root: Path) -> list[str]:
    names = [p.name for p in bench_root.iterdir() if p.is_dir()]
    names.sort()
    return names


def parse_sigmas(text: str) -> list[int]:
    vals = []
    for token in text.split(","):
        token = token.strip()
        if token:
            vals.append(int(float(token)))
    if not vals:
        raise ValueError("No valid sigma provided.")
    return vals


def build_rel_to_path_map(root: Path) -> dict[str, Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]
    return {p.relative_to(root).as_posix(): p for p in files}


def rank_average(rows: list[dict[str, object]]) -> dict[str, float]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for r in rows:
        key = (str(r["dataset"]), int(r["sigma"]))
        groups.setdefault(key, []).append(r)

    ranks: dict[str, list[float]] = {}
    for _, items in groups.items():
        sorted_items = sorted(items, key=lambda x: float(x["psnr"]), reverse=True)
        for idx, item in enumerate(sorted_items, start=1):
            ranks.setdefault(str(item["method"]), []).append(float(idx))

    return {k: float(np.mean(v)) for k, v in ranks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple denoising methods on AWGN datasets.")
    parser.add_argument("--bench-root", type=str, required=True, help="Root produced by export_awgn_noisy_sets.py")
    parser.add_argument("--sigmas", type=str, default="15,25,50")
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Repeatable: name=type:path, type in pred|hybrid_ckpt|cnn_only_ckpt|hybrid_no_gate_ckpt",
    )
    parser.add_argument("--datasets", type=str, default="", help="Optional comma separated dataset names. Empty means all.")
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--save-csv", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench_root = Path(args.bench_root)
    sigmas = parse_sigmas(args.sigmas)
    method_specs = [parse_method(s) for s in args.method]

    datasets = list_datasets(bench_root)
    if args.datasets.strip():
        selected = [x.strip() for x in args.datasets.split(",") if x.strip()]
        datasets = [d for d in datasets if d in selected]
    if not datasets:
        raise ValueError("No dataset found to evaluate.")

    models: dict[str, nn.Module] = {}
    for m in method_specs:
        if m.mtype != "pred":
            models[m.name] = build_model(m.mtype, m.payload, base_ch=args.base_ch, device=device)

    rows: list[dict[str, object]] = []

    for dataset in datasets:
        for sigma in sigmas:
            clean_root = bench_root / dataset / f"sigma{sigma}" / "clean"
            noisy_root = bench_root / dataset / f"sigma{sigma}" / "noisy"
            if not clean_root.exists() or not noisy_root.exists():
                raise FileNotFoundError(f"Missing benchmark folders for {dataset} sigma={sigma}")

            clean_map = build_rel_to_path_map(clean_root)
            noisy_map = build_rel_to_path_map(noisy_root)
            keys = sorted(set(clean_map.keys()).intersection(noisy_map.keys()))
            if not keys:
                raise RuntimeError(f"No clean/noisy matches for {dataset} sigma={sigma}")

            for method in method_specs:
                pred_map = None
                if method.mtype == "pred":
                    pred_root = method.payload / dataset / f"sigma{sigma}"
                    if not pred_root.exists():
                        raise FileNotFoundError(f"Prediction folder missing: {pred_root}")
                    pred_map = build_rel_to_path_map(pred_root)

                pvals = []
                svals = []
                matched = 0
                for key in tqdm(keys, desc=f"{dataset} sigma{sigma} {method.name}", ncols=120, leave=False):
                    clean = load_img(clean_map[key])

                    if method.mtype == "pred":
                        pred_path = pred_map.get(key) if pred_map is not None else None
                        if pred_path is None:
                            continue
                        pred = load_img(pred_path)
                    else:
                        noisy = load_img(noisy_map[key])
                        model = models[method.name]
                        pred = infer_model(model, noisy, sigma=sigma, device=device)

                    pred, clean = align_pred_to_clean(pred, clean)
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                    clean = clean.clamp(0.0, 1.0)

                    pvals.append(float(psnr(pred.unsqueeze(0), clean.unsqueeze(0))))
                    svals.append(float(ssim(pred.unsqueeze(0), clean.unsqueeze(0))))
                    matched += 1

                if matched == 0:
                    raise RuntimeError(f"No matched samples for method={method.name}, dataset={dataset}, sigma={sigma}")

                rows.append(
                    {
                        "dataset": dataset,
                        "sigma": sigma,
                        "method": method.name,
                        "psnr": float(np.mean(pvals)),
                        "ssim": float(np.mean(svals)),
                        "num_samples": matched,
                    }
                )

    avg_rank = rank_average(rows)
    for r in rows:
        r["avg_rank"] = avg_rank[str(r["method"])]

    summary = {
        "datasets": datasets,
        "sigmas": sigmas,
        "rows": rows,
        "avg_rank_by_method": avg_rank,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON to {out}")

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["dataset", "sigma", "method", "psnr", "ssim", "num_samples", "avg_rank"],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Saved CSV to {out}")


if __name__ == "__main__":
    main()