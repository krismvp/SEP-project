# SEP: Emotion Recognition (FER)

This project implements and compares CNN- and ResNet-based models for facial
emotion recognition on the FER dataset. The focus lies on model comparison,
training pipelines, and explainability using Grad-CAM.

## Environment Setup

We use a shared Conda environment for development.

```bash
conda create -n fer python=3.10 -y
conda activate fer
python -m pip install --upgrade pip
pip install -r requirements.txt  
