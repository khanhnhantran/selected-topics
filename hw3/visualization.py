"""Charts for training/validation diagnostics.

All functions are stateless — pass arrays/dicts in, get a saved PNG out.
"""
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "#F9F9F9",
})

NUM_CLASSES = 4
CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4"]
_TAB_COLORS = list(plt.get_cmap("tab10").colors)
MASK_COLORS_RGB = [
    (220, 50, 50), (50, 180, 50), (50, 100, 220), (230, 140, 30),
]
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_fig(fig, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_run_title(cfg: Optional[dict]) -> str:
    if cfg is None:
        return ""
    model = cfg.get("model", {})
    exp = cfg.get("experiment", {})
    name = model.get("name", "model").replace("_", " ").title()
    backbone = model.get("backbone", "").replace("_", "-")
    exp_name = exp.get("name", "")
    parts = [name]
    if backbone:
        parts.append(f"({backbone})")
    if exp_name:
        parts.append(f"Exp {exp_name}")
    return " | ".join(parts)


def denorm_image(img_chw: np.ndarray) -> np.ndarray:
    """(C,H,W) normalized float → (H,W,3) uint8 RGB."""
    img = img_chw.transpose(1, 2, 0) * _STD + _MEAN
    return np.clip(img, 0, 255).astype(np.uint8)


def overlay_masks(image: np.ndarray, masks: List[np.ndarray],
                  labels: List[int], alpha: float = 0.45) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    for mask, label in zip(masks, labels):
        color = MASK_COLORS_RGB[min(label, NUM_CLASSES - 1)]
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask > 0,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c],
            )
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────
#  Plot functions
# ──────────────────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses, save_path,
                     extra_metrics=None, cfg=None) -> None:
    run_tag = make_run_title(cfg)
    n_plots = 1 + (1 if extra_metrics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)
    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2.0,
            color=_TAB_COLORS[0], marker="o", markersize=3)
    if val_losses:
        ax.plot(range(1, len(val_losses) + 1), val_losses,
                label="Val Loss", linewidth=2.0, color=_TAB_COLORS[1],
                linestyle="--", marker="s", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss\n{run_tag}", fontweight="bold")
    ax.legend()

    if extra_metrics and len(axes) > 1:
        ax2 = axes[1]
        for i, (name, values) in enumerate(extra_metrics.items()):
            ax2.plot(range(1, len(values) + 1), values, label=name,
                     linewidth=2.0, color=_TAB_COLORS[i + 2],
                     marker="o", markersize=3)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
        ax2.set_title(f"Validation Metrics\n{run_tag}", fontweight="bold")
        ax2.legend()

    fig.tight_layout()
    save_fig(fig, save_path)


def plot_all_losses(loss_history, val_map_history, save_path, cfg=None) -> None:
    if not loss_history:
        return
    all_keys = sorted({k for ep in loss_history for k in ep.keys() if k != "loss"})
    plot_keys = ["loss"] + all_keys if "loss" in loss_history[0] else all_keys
    n = len(plot_keys) + (1 if val_map_history else 0)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    run_tag = make_run_title(cfg)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    epochs = range(1, len(loss_history) + 1)

    for idx, key in enumerate(plot_keys):
        ax = axes_flat[idx]
        vals = [ep.get(key, float("nan")) for ep in loss_history]
        ax.plot(epochs, vals, linewidth=2.0,
                color=_TAB_COLORS[idx % len(_TAB_COLORS)], marker="o", markersize=3)
        short = key.replace("loss_", "").replace("_", " ").title()
        ax.set_title(f"{short}\n{run_tag}", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Value")

    if val_map_history:
        ax = axes_flat[len(plot_keys)]
        ax.plot(range(1, len(val_map_history) + 1), val_map_history,
                linewidth=2.0,
                color=_TAB_COLORS[len(plot_keys) % len(_TAB_COLORS)],
                marker="s", markersize=4)
        ax.set_title(f"Val Mask AP\n{run_tag}", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Mask AP")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Loss Components - {run_tag}", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_ap_per_class(ap_dict, save_path, cfg=None) -> None:
    run_tag = make_run_title(cfg)
    names = [CLASS_NAMES[i] for i in range(NUM_CLASSES) if i in ap_dict]
    values = [ap_dict[i] for i in range(NUM_CLASSES) if i in ap_dict]
    colors = [_TAB_COLORS[i % len(_TAB_COLORS)] for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    mAP = float(np.mean(values)) if values else 0.0
    ax.axhline(mAP, color="black", linestyle="--", linewidth=1.2, label=f"mAP = {mAP:.3f}")
    ax.set_ylabel("Mask AP (IoU=0.50)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Per-Class Average Precision\n{run_tag}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_pr_curve(precision_per_class, recall_per_class, ap_per_class, save_path, cfg=None) -> None:
    run_tag = make_run_title(cfg)
    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in range(NUM_CLASSES):
        if cls not in precision_per_class:
            continue
        p = np.asarray(precision_per_class[cls])
        r = np.asarray(recall_per_class[cls])
        ap = ap_per_class.get(cls, 0.0)
        ax.plot(r, p, label=f"{CLASS_NAMES[cls]}  AP={ap:.3f}",
                linewidth=2.0, color=_TAB_COLORS[cls % len(_TAB_COLORS)])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Precision-Recall Curve  |  mAP={mAP:.3f}\n{run_tag}", fontweight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_f1_recall_curve(f1_per_class, recall_per_class, ap_per_class, save_path, cfg=None) -> None:
    run_tag = make_run_title(cfg)
    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in range(NUM_CLASSES):
        if cls not in f1_per_class:
            continue
        f1 = np.asarray(f1_per_class[cls])
        r = np.asarray(recall_per_class[cls])
        ap = ap_per_class.get(cls, 0.0)
        ax.plot(r, f1, label=f"{CLASS_NAMES[cls]}  AP={ap:.3f}",
                linewidth=2.0, color=_TAB_COLORS[cls % len(_TAB_COLORS)])
        if len(f1) > 0:
            best = int(np.argmax(f1))
            ax.scatter(r[best], f1[best], s=60,
                       color=_TAB_COLORS[cls % len(_TAB_COLORS)],
                       zorder=5, edgecolors="black", linewidths=0.6)
    ax.set_xlabel("Recall"); ax.set_ylabel("F1 Score")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.05)
    ax.set_title(f"F1-Score vs Recall Curve  |  mAP={mAP:.3f}\n{run_tag}", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_confusion_matrix(confusion, save_path, normalize=True, cfg=None) -> None:
    run_tag = make_run_title(cfg)
    labels = CLASS_NAMES + ["Background"]
    n = len(labels)
    if normalize:
        row_sum = confusion.sum(axis=1, keepdims=True)
        cm_disp = confusion.astype(float) / np.maximum(row_sum, 1)
        fmt = ".2f"
    else:
        cm_disp = confusion.astype(int); fmt = "d"

    cell = max(0.8, 5.0 / n)
    fig, ax = plt.subplots(figsize=(cell * n + 1.5, cell * n + 1.5))
    im = ax.imshow(cm_disp, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=(1 if normalize else None), aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)

    thresh = cm_disp.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_disp[i, j]:{fmt}}", ha="center", va="center",
                    fontsize=max(7, 10 - n // 2),
                    color="white" if cm_disp[i, j] > thresh else "black",
                    fontweight="bold")
    ax.set_ylabel("Ground Truth", fontweight="bold")
    ax.set_xlabel("Predicted", fontweight="bold")
    norm_str = " (Normalized)" if normalize else ""
    ax.set_title(f"Confusion Matrix{norm_str}\n{run_tag}", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, save_path)


def visualize_predictions(images, gt_masks_list, gt_labels_list,
                          pred_masks_list, pred_labels_list, pred_scores_list,
                          save_path, n=6, cfg=None, epoch=None, score_thr=0.05) -> None:
    n = min(n, len(images))
    if n == 0:
        return
    indices = random.sample(range(len(images)), n)
    run_tag = make_run_title(cfg)
    epoch_str = f"  Epoch {epoch}" if epoch is not None else ""

    legend_patches = [
        mpatches.Patch(facecolor=tuple(c / 255 for c in MASK_COLORS_RGB[i]),
                       label=CLASS_NAMES[i], edgecolor="black", linewidth=0.5)
        for i in range(NUM_CLASSES)
    ]

    fig, axes = plt.subplots(n, 2, figsize=(10, 4.5 * n), squeeze=False)
    for row, idx in enumerate(indices):
        img = images[idx]
        keep = [i for i, s in enumerate(pred_scores_list[idx]) if s >= score_thr]
        p_masks = [pred_masks_list[idx][i] for i in keep]
        p_labels = [pred_labels_list[idx][i] for i in keep]

        gt_vis = overlay_masks(img, gt_masks_list[idx], gt_labels_list[idx])
        pred_vis = overlay_masks(img, p_masks, p_labels)
        gt_n = len(gt_masks_list[idx])

        axes[row][0].imshow(gt_vis)
        axes[row][0].set_title(f"GT #{idx}  ({gt_n} instances)", fontsize=10)
        axes[row][0].axis("off")
        axes[row][1].imshow(pred_vis)
        axes[row][1].set_title(f"Pred #{idx}  ({len(p_masks)} dets, thr={score_thr:.2f})", fontsize=10)
        axes[row][1].axis("off")

    fig.legend(handles=legend_patches, loc="lower center", ncol=NUM_CLASSES,
               bbox_to_anchor=(0.5, 0.0), fontsize=9, framealpha=0.9)
    fig.suptitle(f"Predictions vs Ground Truth{epoch_str}\n{run_tag}",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.subplots_adjust(hspace=0.25, top=0.94)
    save_fig(fig, save_path)


def plot_iou_distribution(ious, save_path, cfg=None) -> None:
    run_tag = make_run_title(cfg)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ious, bins=50, range=(0, 1), color=_TAB_COLORS[0],
            edgecolor="black", linewidth=0.4, alpha=0.85)
    med = float(np.median(ious))
    ax.axvline(med, color="red", linestyle="--", linewidth=1.5,
               label=f"Median = {med:.3f}")
    ax.set_xlabel("Mask IoU"); ax.set_ylabel("Count")
    ax.set_title(f"IoU Distribution (Matched Predictions)\n{run_tag}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────
#  Orchestrator: save_all_charts()
# ──────────────────────────────────────────────────────────────────────────

def save_all_charts(chart_dir: str, metrics_data: dict, cfg: Optional[dict] = None) -> None:
    d = Path(chart_dir)
    d.mkdir(parents=True, exist_ok=True)
    epoch = metrics_data.get("epoch")

    if "train_losses" in metrics_data:
        plot_loss_curves(
            metrics_data["train_losses"],
            metrics_data.get("val_losses"),
            str(d / "loss_curves.png"),
            extra_metrics={"Mask AP": metrics_data["val_map_history"]}
            if "val_map_history" in metrics_data else None,
            cfg=cfg,
        )
    if "loss_history" in metrics_data:
        plot_all_losses(
            metrics_data["loss_history"],
            metrics_data.get("val_map_history", []),
            str(d / "loss_components.png"), cfg=cfg,
        )
    if "ap_per_class" in metrics_data:
        plot_ap_per_class(metrics_data["ap_per_class"],
                          str(d / "ap_per_class.png"), cfg=cfg)
    if "pr_data" in metrics_data:
        pr = metrics_data["pr_data"]
        plot_pr_curve(
            {i: pr[i]["precision"] for i in pr},
            {i: pr[i]["recall"] for i in pr},
            {i: pr[i]["ap"] for i in pr},
            str(d / "pr_curve.png"), cfg=cfg,
        )
        plot_f1_recall_curve(
            {i: pr[i]["f1"] for i in pr},
            {i: pr[i]["recall"] for i in pr},
            {i: pr[i]["ap"] for i in pr},
            str(d / "f1_recall_curve.png"), cfg=cfg,
        )
    if "confusion_matrix" in metrics_data:
        plot_confusion_matrix(metrics_data["confusion_matrix"],
                              str(d / "confusion_matrix.png"), cfg=cfg)
    if "vis_data" in metrics_data:
        v = metrics_data["vis_data"]
        visualize_predictions(
            v["images"], v["gt_masks"], v["gt_labels"],
            v["pred_masks"], v["pred_labels"], v["pred_scores"],
            str(d / "predictions_vis.png"), cfg=cfg, epoch=epoch,
        )
