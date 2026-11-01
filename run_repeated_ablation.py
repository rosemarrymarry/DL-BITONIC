from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run repeated ablation experiments across seeds.")
    p.add_argument("--data-root", type=str, required=True, help="Dataset root, e.g. /kaggle/input/... or local data root")
    p.add_argument("--save-root", type=str, default="experiments/repeated_ablation", help="Root folder for all runs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--val-patch-size", type=int, default=32)
    p.add_argument("--max-val-samples", type=int, default=4)
    p.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated random seeds")
    p.add_argument(
        "--models",
        type=str,
        default="hybrid,cnn_only,hybrid_no_gate",
        help="Comma-separated learned models to compare; use traditional only for sanity check.",
    )
    p.add_argument("--sigma-values", type=str, default="10,25,50,75")
    p.add_argument("--include-traditional", action="store_true", help="Also run the non-learned traditional baseline")
    return p.parse_args()


def run_once(cmd: list[str], cwd: Path) -> None:
    print(f"\n=== Running ===\n{' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    script_path = project_root / "src" / "dl_bitonic" / "compare_denoising_experiments.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find comparison script: {script_path}")

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.include_traditional:
        models = list(dict.fromkeys(models + ["traditional"]))

    aggregate = {}

    for seed in seeds:
        seed_dir = save_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--data-root",
            args.data_root,
            "--save-root",
            str(seed_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--patch-size",
            str(args.patch_size),
            "--workers",
            str(args.workers),
            "--lr",
            str(args.lr),
            "--seed",
            str(seed),
            "--sigma-values",
            args.sigma_values,
            "--base-ch",
            str(args.base_ch),
            "--val-batch-size",
            str(args.val_batch_size),
            "--val-patch-size",
            str(args.val_patch_size),
            "--max-val-samples",
            str(args.max_val_samples),
            "--models",
            ",".join(models),
        ]

        run_once(cmd, project_root)

        summary_path = seed_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file for seed {seed}: {summary_path}")

        with summary_path.open("r", encoding="utf-8") as f:
            aggregate[f"seed_{seed}"] = json.load(f)

    report_path = save_root / "aggregate_summary.json"
    report_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("\n=== Aggregated summary ===")
    for seed_name, result_map in aggregate.items():
        print(f"\n{seed_name}")
        for model_name, metrics in result_map.items():
            print(
                f"  {model_name}: best_psnr={metrics.get('val_psnr', 'n/a')}, "
                f"best_ssim={metrics.get('val_ssim', 'n/a')}, loss={metrics.get('loss', 'n/a')}"
            )

    print(f"\nSaved aggregate report to: {report_path}")


if __name__ == "__main__":
    main()
