# SEP: Emotion Recognition (FER2013, FER+, RAF-DB, AffectNet)

Emotion recognition training on FER2013, FER+, RAF-DB, and AffectNet datasets
using ResNet backbones. Supports mixed-domain training for improved generalization.

## Setup

```bash
conda create -n sep_py311 python=3.11 -y
conda activate sep_py311
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure (Code Overview)

- scripts/train/: training entry points (FER2013, FER+, RAF-DB, AffectNet, mixed-domain).
- scripts/eval/: evaluation and inference utilities (including folder inference).
- src/data/: dataset loaders and dataset-specific preprocessing logic.
- src/models/: model architectures and factory to build models.
- src/preprocessing/: optional MTCNN face detection and cropping utilities.
- src/training/: training loops, metrics, and shared utilities.
- outputs/: checkpoints, logs, plots, and inference CSVs.

## Quick Inference (Folder)

Run emotion prediction on a folder of images and save scores to CSV:

```bash
python scripts/eval/predict_folder.py /path/to/images \
  --weights /path/to/checkpoint.pth \
  --output-csv outputs/preds/folder_predictions.csv
```

Notes:
- A checkpoint is required; pass it via `--weights`.
- All models use 6-class emotion labels following the canonical order in `src/constants/emotions.py`.
- Use `--no-mtcnn` to disable face detection and cropping.

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

## Data Download and Preprocessing

Datasets are not included in this repository. Download them from official sources
and place them under the `data/` folder as shown below.

### FER2013 (ImageFolder)

1) Download FER2013 https://www.kaggle.com/datasets/msambare/fer2013
2) Convert the CSV to images and arrange them as ImageFolder:
  - data/FER13/train/<class_name>/*.png
  - data/FER13/val/<class_name>/*.png  (optional; if missing, a random split is used)
3) Class names should match the expected FER labels used in the CSV.

### FER+ (ImageFolder)

1) Download FER+ https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset 
2) Use either of the following layouts:
  - data/ferplus/train/<class_name>/*.png
    data/ferplus/val/<class_name>/*.png (optional)
    data/ferplus/test/<class_name>/*.png (optional)
  - data/ferplus/fer2013plus/fer2013/train/<class_name>/*.png
3) Neutral and contempt are automatically dropped to map into 6 classes.

### RAF-DB (CSV + Images)

1) Download RAF-DB https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset 
2) Ensure the following files exist under data/RAF-DB/ (or inside DATASET/):
  - train_labels.csv
  - test_labels.csv
  - DATASET/train/<label>/*.jpg
  - DATASET/test/<label>/*.jpg (optional)
3) The loader expects CSVs with RAF label ids; neutral is dropped by default.

### AffectNet (ImageFolder)

1) Download AffectNet: https://www.kaggle.com/datasets/mstjebashazida/affectnet 
2) Arrange as ImageFolder structure:
  - data/AffectNet/train/<emotion_class>/*.jpg
  - data/AffectNet/val/<emotion_class>/*.jpg (optional; random split if missing)
3) The loader automatically maps AffectNet labels to the 6-class canonical format.

## Training

### Single-Dataset Training

**RAF-DB:**
```bash
python scripts/train/train_raf.py --data-dir data/RAF-DB --epochs 25
```

**FER+:**
```bash
python scripts/train/train_ferplus.py --data-dir data/ferplus --epochs 20
```

For class imbalance, use `--weighted-sampler` to balance classes or `--class-weight-power 0.5`
to soften class weights. Use `--augmentation strong` for stronger augmentations.

### Mixed-Domain Training (FER+ + RAF + AffectNet)

Train on multiple datasets simultaneously with balanced sampling:
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
- Folder inference: `outputs/preds/folder_predictions.csv` (recommended)

Use `--output-dir` to avoid overwriting previous runs.

## Notes

- `pillow` is required for image loading.
- `gdown` is only needed if you want torchvision to download CelebA automatically.
- All experiments use grayscale 1-channel inputs for cross-dataset consistency; RGB datasets are converted to grayscale at preprocessing time.
