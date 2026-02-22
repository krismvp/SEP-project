# SEP Emotion Recognition

Emotion recognition project with ResNet backbones on a shared 6-class label space:
`anger`, `disgust`, `fear`, `happy`, `sad`, `surprise`.

## For Correctors (Run This First)

Please test these two scripts:

1. `process_video` (required)
```bash
python3 scripts/demo/process_video.py
```
- Put test videos in `inputdata/` (file picker opens there by default).
- Put checkpoint at `inference/resnet34_best.pth`.
- Output video is saved to `inference/processed_<original_filename>`.

2. `predict_folder` (required)
```bash
python3 scripts/eval/predict_folder.py
```
- Default input folder: `inputdata/`
- Default checkpoint: `inference/resnet34_best.pth`
- Default CSV output: `inference/folder_predictions.csv`
- Shows a live `tqdm` progress bar while running.

## Setup

```bash
conda create -n sep_py311 python=3.11 -y
conda activate sep_py311
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Layout

- `scripts/train/`: training entrypoints.
- `scripts/eval/`: evaluation and folder inference scripts.
- `scripts/demo/`: webcam and video demo scripts.
- `src/data/`: dataset loaders and preprocessing pipelines.
- `src/models/`: model implementations and factory.
- `src/training/`: training loops and utilities.
- `inputdata/`: default inputs for inference demos.
- `inference/`: default checkpoint and inference outputs.
- `outputs/`: training/evaluation artifacts.

## Data Layout

### AffectNet (ImageFolder)
```text
data/AffectNet/
  train/  (or Train/ or Training/)
    <class_name>/*.{jpg,png,...}
  test/   (optional; or Test/ or Testing/)
    <class_name>/*.{jpg,png,...}
```

### FER+ (ImageFolder)
Supported layouts:
```text
data/ferplus/
  train/
  val/   (optional)
  test/  (optional)
```
or
```text
data/ferplus/fer2013plus/fer2013/
  train/
  val/   (optional)
  test/  (optional)
```

### RAF-DB (CSV + images)
```text
data/RAF-DB/
  train_labels.csv
  test_labels.csv
  DATASET/
    train/*.jpg
    test/*.jpg
```

Notes:
- Loaders map labels to canonical 6 classes and drop unsupported labels (for example neutral/contempt where applicable).
- If an explicit validation split is missing, loaders can create one with `--val-split`.

## Training

### Mixed AffectNet + FER+ + RAF
```bash
python3 scripts/train/train_mixed_affectnet_ferplus_raf.py \
  --affectnet-dir data/AffectNet \
  --fer-data-dir data/ferplus \
  --raf-data-dir data/RAF-DB \
  --arch resnet34 \
  --epochs 25 \
  --output-dir outputs/mixed/affectnet_ferplus_raf_resnet34_mtcnn
```

### Mixed FER+ + RAF
```bash
python3 scripts/train/train_mixed_ferplus_raf.py \
  --fer-data-dir data/ferplus \
  --raf-data-dir data/RAF-DB \
  --arch resnet34 \
  --epochs 25 \
  --output-dir outputs/mixed/ferplus_raf_resnet34_mtcnn
```

### Single Dataset
FER+:
```bash
python3 scripts/train/train_ferplus.py --data-dir data/ferplus --epochs 20
```

RAF-DB:
```bash
python3 scripts/train/train_raf.py --data-dir data/RAF-DB --epochs 25
```

AffectNet:
```bash
python3 scripts/train/train_affectnet.py --data-dir data/AffectNet --epochs 8
```

## Evaluation

FER+:
```bash
python3 scripts/eval/eval_ferplus.py \
  --data-dir data/ferplus \
  --weights /path/to/checkpoint.pth \
  --split test \
  --output-dir outputs/ferplus_eval
```

RAF-DB:
```bash
python3 scripts/eval/eval_raf.py \
  --data-dir data/RAF-DB \
  --weights /path/to/checkpoint.pth \
  --split test \
  --output-dir outputs/raf_eval
```

AffectNet:
```bash
python3 scripts/eval/eval_affectnet.py \
  --data-dir data/AffectNet \
  --weights /path/to/checkpoint.pth \
  --split test \
  --output-dir outputs/affectnet_eval
```

## Inference and Demo

### Folder Prediction
Default run:
```bash
python3 scripts/eval/predict_folder.py
```

Explicit run:
```bash
python3 scripts/eval/predict_folder.py inputdata \
  --weights inference/resnet34_best.pth \
  --output-csv inference/folder_predictions.csv
```

### Video Processing Demo
```bash
python3 scripts/demo/process_video.py
```

### Live Webcam Demo
```bash
python3 scripts/demo/demo_live.py
```
Default checkpoint path for live demo:
`inference/resnet34_best.pth`

## Outputs

- Inference CSV: `inference/folder_predictions.csv`
- Processed video: `inference/processed_<original_filename>`
- Training/evaluation artifacts: under `outputs/` and custom `--output-dir` values.

## Notes

- Inputs are converted to grayscale for cross-dataset consistency.
- MTCNN options are available in training/evaluation/inference scripts (`--use-mtcnn`, `--no-mtcnn`).
- If `python3` cannot import `torch`, activate the correct conda environment before running scripts.
