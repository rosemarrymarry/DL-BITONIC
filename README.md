# dl-bitonic

Bitonic-guided deep denoising baseline for additive white Gaussian noise (AWGN).

This project implements:

- A differentiable bitonic-like local adaptive prior module.
- A dual-path denoiser that fuses bitonic prior and deep prediction.
- Sigma-aware conditioning for noise-level adaptation.
- Training and evaluation scripts for reproducible ablations.

## 1. Project structure

```text
dl-bitonic/
  requirements.txt
  train.py
  evaluate.py
  src/
    dl_bitonic/
      __init__.py
      bitonic.py
      dataset.py
      losses.py
      model.py
      metrics.py
```

## 2. Prepare data

Expected clean image layout:

```text
your_dataset_root/
  train/
    *.png|jpg|bmp
  val/
    *.png|jpg|bmp
```

Noise is synthesized on-the-fly as Gaussian noise.

## 3. Install

```bash
pip install -r requirements.txt
```

## 4. Train

```bash
python train.py \
  --data-root path/to/your_dataset_root \
  --save-dir runs/exp1 \
  --epochs 300 \
  --batch-size 16 \
  --patch-size 128 \
  --sigma-min 5 \
  --sigma-max 75
```

## 5. Evaluate

```bash
python evaluate.py \
  --data-root path/to/your_dataset_root \
  --checkpoint runs/exp1/best.pt \
  --sigma 15
```

Run evaluation multiple times with sigma 15, 25, 50.

## 6. Recommended paper ablations

- Deep-only: disable bitonic branch and gate fusion.
- + Bitonic prior input: enable xb and residual input.
- + Gate fusion: enable per-pixel gate output.
- + Sigma-aware bitonic: enable adaptive threshold prediction.

## 7. Notes

- Input range is normalized to [0, 1].
- Default model is lightweight and suitable for a single GPU.
- Use fixed random seed and identical test sets for fair comparison.
