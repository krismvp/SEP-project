# SEP: Emotion Recognition (CelebA, RAF-DB, FER2013)

Self-supervised pretraining with SimCLR on CelebA, followed by fine-tuning on
RAF-DB or FER2013 using a ResNet-18 backbone.

## Setup

```bash
conda create -n fer python=3.10 -y
conda activate fer
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data Layout

FER2013 (ImageFolder):
```
data/FER13/
  train/
    <class_name>/*.png
  val/               # optional; if missing, a random split is used
    <class_name>/*.png
```

FER+ (ImageFolder):
```
data/ferplus/
  fer2013plus/fer2013/
    train/
      <class_name>/*.png
    test/             # optional
      <class_name>/*.png
```
If no `val/` folder exists, a random split is used.
Use `--drop-neutral` and/or `--drop-contempt` to reduce to 6 or 7 classes.

RAF-DB (CSV + images):
```
data/RAF-DB/
  train_labels.csv
  test_labels.csv
  DATASET/
    train/
      <label>/*.jpg
    # val/ and test/ are optional; if missing, val is a random split and test is skipped
```
By default, the neutral class is dropped. Use `--include-neutral` to keep it.

## Train/Fine-tune RAF-DB

From scratch:
```bash
python scripts/train/train_raf.py --data-dir data/RAF-DB --epochs 25
```

Fine-tune from CelebA pretrain:
```bash
python scripts/train/train_raf.py \
  --data-dir data/RAF-DB \
  --pretrained-path outputs/pretrained_backbone.pth
```

## Train/Fine-tune FER+

From scratch:
```bash
python scripts/train/train_ferplus.py --data-dir data/ferplus --epochs 20
```

Fine-tune from CelebA pretrain:
```bash
python scripts/train/train_ferplus.py \
  --data-dir data/ferplus \
  --pretrained-path outputs/pretrained_backbone.pth
```
Use `--confusion-matrix` to save a test-set confusion matrix in the output dir.
For class imbalance, try `--no-weighted-loss` to disable weights or
`--class-weight-power 0.5` to soften them. Use `--weighted-sampler` to oversample
minority classes.
Use `--augmentation strong` for stronger FER-style training augmentations.

## Outputs

- RAF-DB: `outputs/finetune/` (best/last checkpoints + `training_curves.png`)
- FER2013: `outputs/` (`training_curves.png` and best-epoch log)
- FER+: `outputs/ferplus/` (best/last checkpoints + `training_curves.png`)
- Pretrain: `outputs/pretrained_backbone.pth`

Use `--output-dir` to avoid overwriting previous runs.

## Notes

- `pillow` is required for image loading.
- `gdown` is only needed if you want torchvision to download CelebA automatically.
- All experiments use grayscale 1-channel inputs for cross-dataset consistency; RGB datasets are converted to grayscale at preprocessing time.
