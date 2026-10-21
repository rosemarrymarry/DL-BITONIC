from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return ten


def random_crop(img: torch.Tensor, patch: int, rng: np.random.Generator) -> torch.Tensor:
    _, h, w = img.shape
    if h < patch or w < patch:
        scale = max(patch / h, patch / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
        pil = pil.resize((nw, nh), Image.BICUBIC)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        _, h, w = img.shape

    top = int(rng.integers(0, h - patch + 1))
    left = int(rng.integers(0, w - patch + 1))
    return img[:, top : top + patch, left : left + patch]


def augment(img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    if rng.random() < 0.5:
        img = torch.flip(img, dims=[2])
    if rng.random() < 0.5:
        img = torch.flip(img, dims=[1])
    k = int(rng.integers(0, 4))
    if k > 0:
        img = torch.rot90(img, k=k, dims=[1, 2])
    return img


class GaussianNoisyPairDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        patch_size: int = 128,
        sigma_min: float = 5.0,
        sigma_max: float = 75.0,
        fixed_sigma: float | None = None,
        train: bool = True,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.fixed_sigma = fixed_sigma
        self.train = train
        self.rng = np.random.default_rng(seed)

        folder = self.root / split
        if not folder.exists():
            raise FileNotFoundError(f"Dataset split folder not found: {folder}")
        self.files = sorted([p for p in folder.rglob("*") if p.is_file() and _is_image(p)])
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def _sample_sigma(self) -> float:
        if self.fixed_sigma is not None:
            return float(self.fixed_sigma)
        return float(self.rng.uniform(self.sigma_min, self.sigma_max))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        clean = _load_rgb(self.files[index])
        if self.train:
            clean = random_crop(clean, self.patch_size, self.rng)
            clean = augment(clean, self.rng)

        sigma = self._sample_sigma()
        sigma_norm = sigma / 255.0
        noise = torch.randn_like(clean) * sigma_norm
        noisy = (clean + noise).clamp(0.0, 1.0)

        sigma_map = torch.full((1, clean.shape[1], clean.shape[2]), sigma / 75.0, dtype=clean.dtype)
        return {
            "clean": clean,
            "noisy": noisy,
            "sigma_map": sigma_map,
            "sigma": torch.tensor([sigma], dtype=torch.float32),
        }
