# HW1 — Image Classification (100 Classes)

## Introduction

This project implements a multi-class image classification system for 100 categories as part of the NYCU Selected Topics course. The goal is to fine-tune state-of-the-art pre-trained models on the given dataset and maximize validation accuracy under a **100M parameter budget**.

**Key features:**
- Multiple backbone options: ResNet50/101, SEResNextAA101D (32x8d), EVA02-Large, and a custom CNN
- Parameter-efficient fine-tuning via **LoRA (Low-Rank Adaptation)**
- Handles class imbalance through weighted loss and weighted sampling
- Data augmentation: RandAugment, Mixup, color jitter, random erasing
- Distributed training support (multi-GPU via `torchrun`)
- Automatic visualization: training curves, confusion matrix, per-class ROC curves

---

## Environment Setup

**Requirements:** Python 3.8+, CUDA-capable GPU recommended.

### 1. Create and activate a virtual environment

```bash
conda create -n hw1 python=3.10 -y
conda activate hw1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:
- `torch >= 2.0.0`, `torchvision >= 0.15.0`
- `timm >= 0.9.0` — pre-trained model zoo
- `albumentations`, `opencv-python` — image augmentation
- `cleanlab`, `cleanvision` — data quality inspection
- `tensorboard`, `matplotlib`, `seaborn`, `scikit-learn`

### 3. Prepare the dataset

Organize data in the following directory structure:

```
data/
├── train/
│   ├── 0/          # class 0 images
│   ├── 1/
│   └── ... 99/
├── val/
│   ├── 0/
│   └── ... 99/
└── test/
    ├── 00001.jpg
    └── ...
```

---

## Usage

### Training

Select a config from `configs/train/` and run:

```bash
python train.py --config configs/train/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288.yaml --seed 42
```

For multi-GPU distributed training:

```bash
torchrun --nproc_per_node 4 train.py \
    --config configs/train/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288.yaml \
    --seed 42
```

**Available configs:**

| Config file | Backbone | Notes |
|---|---|---|
| `resnet50_clip.openai.yaml` | ResNet50 (CLIP) | Fast baseline |
| `resnet101.yaml` | ResNet101 | Standard fine-tuning |
| `seresnextaa101d_32x8d.sw_in12k_ft_in1k_288.yaml` | SEResNextAA101D | Best single model |
| `seresnextaa101d_32x8d.sw_in12k_ft_in1k_288_lora.yaml` | SEResNextAA101D + LoRA | Parameter-efficient |
| `seresnextaa101d_32x8d.sw_in12k_ft_in1k_288_imbalance.yaml` | SEResNextAA101D | Imbalanced data handling |
| `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.yaml` | EVA02-Large | Highest capacity |

### Inference

```bash
python test.py --config configs/test.yaml --seed 42
```

This generates `submission.csv` with columns `image_name` and `pred_label`.

### Monitoring

```bash
tensorboard --logdir runs/
```

### Outputs

| Path | Description |
|---|---|
| `runs/{model}/{exp_id}/` | TensorBoard logs and epoch CSV |
| `runs/{model}/{exp_id}/performance.csv` | Summary metrics |
| `checkpoints/{model}/{exp_id}/best_model.pth` | Best checkpoint |
| `charts/{model}/{exp_id}/` | Training curves, confusion matrix, ROC curves |
| `submission.csv` | Test set predictions |

---

## Performance Snapshot

Results on the validation set (100 classes):

| Model | Params | Val Acc (Top-1) | Notes |
|---|---|---|---|
| Custom CNN | ~2M | ~30% | Trained from scratch |
| ResNet50 (CLIP pre-trained) | ~25M | ~75% | Full fine-tuning |
| ResNet101 | ~44M | ~78% | Full fine-tuning |
| SEResNextAA101D + LoRA | ~93M (8M trainable) | ~87% | Frozen backbone, LoRA rank=16 |
| SEResNextAA101D (full fine-tune) | ~93M | ~90% | Best single model |
| EVA02-Large (448×448) | ~307M* | ~92% | Exceeds budget — eval only |

> *EVA02-Large exceeds the 100M parameter budget and is used for reference comparison only.

**Training setup (SEResNextAA101D):**
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-2
- Scheduler: Cosine annealing with 5 warmup epochs
- Augmentation: RandAugment (m7), Mixup (α=0.2), random erasing
- Loss: CrossEntropyLoss + label smoothing (0.1)
- Input size: 320×320, effective batch size: 32 (gradient accumulation ×2)
- Epochs: 50
