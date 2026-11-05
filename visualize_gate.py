from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dl_bitonic.dataset import GaussianNoisyPairDataset
from src.dl_bitonic.model import BitonicGuidedDenoiser


def main() -> None:
    ckpt_path = Path("/kaggle/working/runs/exp1/best.pt")
    data_root = Path("/kaggle/working/div2k")
    save_dir = Path("/kaggle/working/gate_vis")
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model = BitonicGuidedDenoiser(
        use_sigma_aware_tau=not bool(ckpt_args.get("disable_sigma_aware_tau", False)),
        base_channels=int(ckpt_args.get("base_ch", 32)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = GaussianNoisyPairDataset(
        root=data_root,
        split="val",
        patch_size=128,
        fixed_sigma=25.0,
        train=False,
        seed=123,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= 3:
                break

            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)
            sigma_map = batch["sigma_map"].to(device)

            pred, aux = model(noisy, sigma_map)
            gate = aux["gate"][0, 0].detach().cpu().numpy()

            clean_img = clean[0].permute(1, 2, 0).cpu().numpy()
            noisy_img = noisy[0].permute(1, 2, 0).cpu().numpy()
            pred_img = pred[0].permute(1, 2, 0).cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(clean_img)
            axes[0].set_title("clean")
            axes[0].axis("off")

            axes[1].imshow(noisy_img)
            axes[1].set_title("noisy")
            axes[1].axis("off")

            axes[2].imshow(pred_img)
            axes[2].set_title("pred")
            axes[2].axis("off")

            heat = axes[3].imshow(gate, cmap="jet", vmin=0, vmax=1)
            axes[3].set_title("gate")
            axes[3].axis("off")
            fig.colorbar(heat, ax=axes[3], fraction=0.046, pad=0.04)

            fig.tight_layout()
            fig.savefig(save_dir / f"gate_vis_{count}.png", dpi=200)
            plt.close(fig)

            count += 1

    print(f"Saved gate visualizations to: {save_dir}")


if __name__ == "__main__":
    main()
