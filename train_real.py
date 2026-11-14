import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dl_bitonic.model import BitonicGuidedDenoiser
from src.dl_bitonic.losses import CompositeLoss
from src.dl_bitonic.metrics import psnr, ssim

# ---------------------
# Config
# ---------------------
SEED = 42
TRAIN_ROOT = "/kaggle/working/sidd_split/train"
VAL_ROOT = "/kaggle/working/sidd_split/val"
TEST_ROOT = "/kaggle/working/sidd_split/test"
CKPT = "/kaggle/working/DL-BITONIC/src/dl_bitonic/best-120.pt"
SAVE_DIR = "/kaggle/working/real_finetune_valid_10"
EPOCHS = 10
MAX_TRAIN_PAIRS = 30
MAX_VAL_PAIRS = 10
BATCH_SIZE = 1
PATCH = 64
LR = 5e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------------
# Helpers
# ---------------------
def normalize_name(name: str) -> str:
    s = Path(name).name.lower()
    for marker in ("_srgb_", "_rgb_", "_srgb", "_rgb"):
        if marker in s:
            s = s.split(marker, 1)[1]
            break
    for prefix in ("noisy_", "gt_", "noisy", "gt"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith(".png"):
        s = s[:-4]
    return s.replace(" ", "")

def load_rgb(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

class RealPairDataset(Dataset):
    def __init__(self, root, max_pairs=None, patch=64):
        self.root = Path(root)
        self.patch = patch
        self.pairs = []

        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            files = {normalize_name(p.name): p for p in folder.iterdir() if p.is_file()}
            noisy = files.get(normalize_name("NOISY_SRGB_010.PNG"))
            gt = files.get(normalize_name("GT_SRGB_010.PNG"))
            if noisy is not None and gt is not None:
                self.pairs.append((noisy, gt))

        if max_pairs is not None:
            self.pairs = self.pairs[:max_pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]
        noisy = load_rgb(noisy_path)
        clean = load_rgb(gt_path)

        h, w = noisy.shape[1], noisy.shape[2]
        top = random.randint(0, max(0, h - self.patch))
        left = random.randint(0, max(0, w - self.patch))
        noisy = noisy[:, top:top+self.patch, left:left+self.patch]
        clean = clean[:, top:top+self.patch, left:left+self.patch]

        sigma_map = torch.full((1, noisy.shape[1], noisy.shape[2]), 25.0 / 75.0, dtype=noisy.dtype)
        return {"clean": clean, "noisy": noisy, "sigma_map": sigma_map}

# ---------------------
# Train loader
# ---------------------
train_ds = RealPairDataset(TRAIN_ROOT, max_pairs=MAX_TRAIN_PAIRS, patch=PATCH)
val_ds = RealPairDataset(VAL_ROOT, max_pairs=MAX_VAL_PAIRS, patch=PATCH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

# ---------------------
# Model
# ---------------------
model = BitonicGuidedDenoiser(use_sigma_aware_tau=True, base_channels=32).to(DEVICE)
ckpt = torch.load(CKPT, map_location="cpu")
model.load_state_dict(ckpt["model"], strict=True)

criterion = CompositeLoss(w_ssim=0.2, w_edge=0.05, w_cons=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

best_val_psnr = -1e9
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for batch in train_loader:
        clean = batch["clean"].to(DEVICE)
        noisy = batch["noisy"].to(DEVICE)
        sigma_map = batch["sigma_map"].to(DEVICE)

        optimizer.zero_grad()
        pred, aux = model(noisy, sigma_map)
        consistency_out = model.bitonic(pred, aux["tau"])
        loss, _ = criterion(pred, clean, aux["xb"], consistency_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())

    # validation on independent val set
    model.eval()
    val_psnr_vals = []
    val_ssim_vals = []
    with torch.no_grad():
        for batch in val_loader:
            clean = batch["clean"].to(DEVICE)
            noisy = batch["noisy"].to(DEVICE)
            sigma_map = batch["sigma_map"].to(DEVICE)

            pred, _ = model(noisy, sigma_map)
            val_psnr_vals.append(psnr(pred, clean))
            val_ssim_vals.append(ssim(pred, clean))

    val_psnr = float(np.mean(val_psnr_vals))
    val_ssim = float(np.mean(val_ssim_vals))

    print(f"epoch {epoch} train_loss={total_loss/len(train_loader):.6f} val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f}")

    if val_psnr > best_val_psnr:
        best_val_psnr = val_psnr
        best_state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim
        }
        torch.save(best_state, os.path.join(SAVE_DIR, "best_real_ft.pt"))

print("best_val_psnr =", best_val_psnr)