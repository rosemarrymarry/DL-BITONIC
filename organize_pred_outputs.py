from __future__ import annotations

import argparse
import shutil
from pathlib import Path

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
KNOWN_METHOD_SUFFIXES = (
    "_swinir", "_dncnn", "_ffdnet", "_drunet", "_scunet", "_restormer", "_bm3d",
    "_restormer_denoising", "_scunet_color", "_scunet_gray",
)


def normalize_stem(path: Path, extra_suffixes: tuple[str, ...] = ()) -> str:
    stem = path.stem.lower()
    for suf in (*extra_suffixes, *KNOWN_METHOD_SUFFIXES):
        suf = suf.lower()
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem.replace(" ", "").replace("_", "")


def collect_pred_files(pred_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for p in sorted(pred_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VALID_SUFFIXES:
            continue
        key = normalize_stem(p)
        if key in mapping:
            print(f"[warn] pred 重名冲突,覆盖: {mapping[key]} -> {p}")
        mapping[key] = p
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description="整理外部方法输出到 benchmark 的 pred 约定目录")
    ap.add_argument("--noisy-root", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sigma", required=True, type=int)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--strip-suffix", action="append", default=[])
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    extra = tuple(s.strip() for s in args.strip_suffix if s.strip())

    noisy_dir = Path(args.noisy_root) / args.dataset / f"sigma{args.sigma}" / "noisy"
    if not noisy_dir.is_dir():
        raise FileNotFoundError(f"noisy 目录不存在: {noisy_dir}")

    pred_root = Path(args.pred_root)
    if not pred_root.is_dir():
        raise FileNotFoundError(f"pred 目录不存在: {pred_root}")

    pred_map = collect_pred_files(pred_root)
    if not pred_map:
        raise RuntimeError(f"pred 目录里没找到图片: {pred_root}")

    out_dir = Path(args.out_root) / args.method / args.dataset / f"sigma{args.sigma}"

    matched, missing = 0, 0
    noisy_files = sorted(p for p in noisy_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES)
    for noisy in noisy_files:
        rel = noisy.relative_to(noisy_dir)
        key = normalize_stem(noisy, extra)
        pred = pred_map.get(key)
        if pred is None:
            missing += 1
            print(f"[miss ] {rel}")
            continue
        matched += 1
        dst = out_dir / rel
        if args.list_only:
            print(f"[match] {rel}  <-  {pred}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pred, dst)

    print(f"\n匹配 {matched} 张,缺失 {missing} 张。")
    if not args.list_only and matched:
        print(f"已写入: {out_dir}")

    if missing:
        raise SystemExit(f"有 {missing} 张 noisy 没匹配到 pred,请检查 --strip-suffix 或 pred 目录。")


if __name__ == "__main__":
    main()
