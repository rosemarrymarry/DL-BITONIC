from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_dataset_specs(items: list[str]) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset spec: {item}. Use name=/path/to/clean_images")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --dataset spec: {item}")
        pairs.append((name, Path(path)))
    return pairs


def parse_sigmas(text: str) -> list[int]:
    vals = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(float(token)))
    if not vals:
        raise ValueError("No valid sigma provided.")
    return vals


def list_images(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES])


def to_numpy_float(img: Image.Image) -> np.ndarray:
    if img.mode == "L":
        arr = np.asarray(img, dtype=np.float32)[..., None]
    else:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return arr / 255.0


def from_numpy_float(arr: np.ndarray, mode: str) -> Image.Image:
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    if mode == "L":
        if arr.ndim == 3:
            arr = arr[..., 0]
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return Image.fromarray(arr, mode="RGB")


def seed_from_key(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2 ** 32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic AWGN noisy sets for multiple datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeatable: name=/path/to/clean_images",
    )
    parser.add_argument("--sigmas", type=str, default="15,25,50")
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_specs = parse_dataset_specs(args.dataset)
    sigmas = parse_sigmas(args.sigmas)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for dataset_name, clean_root in dataset_specs:
        files = list_images(clean_root)
        if not files:
            raise FileNotFoundError(f"No images found in dataset: {clean_root}")

        for sigma in sigmas:
            sigma_dir = out_root / dataset_name / f"sigma{sigma}"
            clean_out = sigma_dir / "clean"
            noisy_out = sigma_dir / "noisy"
            clean_out.mkdir(parents=True, exist_ok=True)
            noisy_out.mkdir(parents=True, exist_ok=True)

            for img_path in files:
                rel = img_path.relative_to(clean_root)
                clean_target = clean_out / rel
                noisy_target = noisy_out / rel
                clean_target.parent.mkdir(parents=True, exist_ok=True)
                noisy_target.parent.mkdir(parents=True, exist_ok=True)

                if (not args.overwrite) and clean_target.exists() and noisy_target.exists():
                    continue

                img = Image.open(img_path)
                mode = "L" if img.mode == "L" else "RGB"
                arr = to_numpy_float(img)

                key = f"{dataset_name}/{rel.as_posix()}/sigma{sigma}"
                rng = np.random.default_rng(seed_from_key(args.seed, key))
                noise = rng.normal(loc=0.0, scale=float(sigma) / 255.0, size=arr.shape).astype(np.float32)
                noisy = np.clip(arr + noise, 0.0, 1.0)

                clean_img = from_numpy_float(arr, mode=mode)
                noisy_img = from_numpy_float(noisy, mode=mode)

                clean_img.save(clean_target)
                noisy_img.save(noisy_target)
                total_written += 1

    print(f"Done. Exported {total_written} clean/noisy files under: {out_root}")


if __name__ == "__main__":
    main()