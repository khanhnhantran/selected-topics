import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader


@dataclass
class LoraConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["Linear"])
    save_lora_only: bool = False


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 100
    drop_rate: float = 0.2
    checkpoint: Optional[str] = None
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class DataConfig:
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    test_dir: str = "data/test"
    image_size: int = 224
    resize_size: int = 334
    crop_size: int = 320
    batch_size: int = 64
    num_workers: int = 4
    use_augmentation: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    mixup_alpha: float = 0.0
    accumulation_steps: int = 1
    use_class_weights: bool = True
    use_weighted_sampler: bool = True


@dataclass
class OutputConfig:
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_top_k: int = 3
    submission_file: str = "submission.csv"


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class TestConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class BatchResult:
    loss: float
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float


@dataclass
class Prediction:
    image_name: str
    label: int


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def start_timer() -> float:
    return time.perf_counter()


def stop_timer(start: float) -> float:
    return time.perf_counter() - start


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


@contextmanager
def timed(label: str = "") -> Generator[None, None, None]:
    t = start_timer()
    yield
    elapsed = stop_timer(t)
    if label:
        print(f"{label}: {format_duration(elapsed)}")


def log_model_size(model: nn.Module, logger: Optional[logging.Logger] = None) -> int:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = f"Model parameters: total={total:,}  trainable={trainable:,}"
    if logger:
        logger.info(msg)
        logger.info("Model architecture:\n%s", str(getattr(model, "model", model)))
    else:
        print(msg)
        print(str(getattr(model, "model", model)))
    if total > 100_000_000:
        raise ValueError(f"Model has {total:,} parameters, exceeding the 100M limit.")
    return total


def log_performance_csv(
    model_name: str,
    model_size: int,
    epochs: int,
    lr: float,
    optimizer: str,
    image_size: int,
    train_acc: float,
    val_acc: float,
    train_loss: float = 0.0,
    val_loss: float = 0.0,
    runs_dir: str = "runs",
) -> None:
    out_dir = Path(runs_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "performance.csv"
    row = pd.DataFrame([{
        "model": model_name,
        "model_size": model_size,
        "epochs": epochs,
        "lr": lr,
        "optimizer": optimizer,
        "image_size": image_size,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
    }])
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), row], ignore_index=True)
    else:
        df = row
    df.to_csv(csv_path, index=False)


def plot_training_curves(csv_path: str, save_dir: Optional[str] = None) -> None:
    df = pd.read_csv(csv_path)
    save_dir = Path(save_dir) if save_dir else Path(csv_path).parent
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["epoch"], df["train_acc"], label="Train")
    axes[1].plot(df["epoch"], df["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df["epoch"], df["lr"])
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].grid(True)

    fig.tight_layout()
    out = save_dir / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved training curves → {out}")


@torch.no_grad()
def plot_confusion_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    normalize: bool = True,
) -> None:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())

    norm = "true" if normalize else None
    cm = confusion_matrix(all_labels, all_preds, normalize=norm)
    num_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, num_classes // 4), max(10, num_classes // 4)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, xticks_rotation="vertical", values_format=".2f" if normalize else "d")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    fig.tight_layout()
    save_dir = Path(save_dir) if save_dir else Path("runs")
    out = save_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {out}")


@torch.no_grad()
def plot_roc_curves(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    plot_individual = num_classes <= 20
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        if plot_individual:
            label = class_names[i] if class_names else f"class {i}"
            ax.plot(fpr, tpr, lw=1, alpha=0.6, label=f"{label} (AUC={roc_auc:.2f})")

    all_fpr = np.unique(np.concatenate([roc_curve(labels_bin[:, i], all_probs[:, i])[0] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= num_classes
    macro_auc = auc(all_fpr, mean_tpr)

    ax.plot(all_fpr, mean_tpr, color="navy", lw=2, linestyle="--", label=f"Macro-avg (AUC={macro_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    if plot_individual:
        ax.legend(loc="lower right", fontsize="x-small", ncol=max(1, num_classes // 10))
    else:
        ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    save_dir = Path(save_dir) if save_dir else Path("runs")
    out = save_dir / "roc_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curves → {out}  (macro AUC={macro_auc:.4f})")
