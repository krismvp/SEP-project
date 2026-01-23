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

CelebA (for SSL pretraining):
```
data/celebA/
  img_align_celeba/
  list_bbox_celeba.csv  # or .txt
  list_eval_partition.csv  # optional
```
The loader also accepts a nested `img_align_celeba/img_align_celeba` folder and
alternate roots like `data/celebA/celeba`.

## Pretrain (SimCLR on CelebA)

```bash
python scripts/pretrain_selfsupervised.py --data-dir data/celebA --epochs 100
```
Outputs: `outputs/pretrained_backbone.pth`

## Train/Fine-tune RAF-DB

From scratch:
```bash
python scripts/train_raf.py --data-dir data/RAF-DB --epochs 25
```

Fine-tune from CelebA pretrain:
```bash
python scripts/train_raf.py \
  --data-dir data/RAF-DB \
  --pretrained-path outputs/pretrained_backbone.pth \
  --freeze-epochs 5
```

## Train/Fine-tune FER2013

From scratch:
```bash
python scripts/train_fer2013.py --data-path data/FER13 --epochs 20
```

Fine-tune from CelebA pretrain:
```bash
python scripts/train_fer2013.py \
  --data-path data/FER13 \
  --pretrained-path outputs/pretrained_backbone.pth \
  --freeze-epochs 5
```

## Train/Fine-tune FER+

From scratch:
```bash
python scripts/train_ferplus.py --data-dir data/ferplus --epochs 20
```

Fine-tune from CelebA pretrain:
```bash
python scripts/train_ferplus.py \
  --data-dir data/ferplus \
  --pretrained-path outputs/pretrained_backbone.pth \
  --freeze-epochs 5
```

## Outputs

- RAF-DB: `outputs/finetune/` (best/last checkpoints + `training_curves.png`)
- FER2013: `outputs/` (`training_curves.png` and best-epoch log)
- FER+: `outputs/ferplus/` (best/last checkpoints + `training_curves.png`)
- Pretrain: `outputs/pretrained_backbone.pth`

Use `--output-dir` to avoid overwriting previous runs.

## Notes

- `pillow` is required for image loading.
- `gdown` is only needed if you want torchvision to download CelebA automatically.
