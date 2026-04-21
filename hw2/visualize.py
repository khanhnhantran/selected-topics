from __future__ import annotations

import glob as _glob
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 9,
    }
)

_PALETTE = plt.get_cmap("tab10").colors


def _ema(values: Sequence[float], alpha: float = 0.1) -> np.ndarray:
    vals = np.array(values, dtype=float)
    out = np.empty_like(vals)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
    return out


def _savefig(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")


def _cls_color(idx: int) -> tuple:
    return _PALETTE[idx % len(_PALETTE)]


def plot_training_curves(
    metrics: Dict[str, Dict[str, List[float]]],
    x_label: str = "Epoch",
    ema_alpha: float = 0.1,
    ncols: int = 5,
    figsize_per_cell: Tuple[float, float] = (3.5, 2.8),
    suptitle: str = "",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    if not metrics:
        raise ValueError("metrics dict is empty")

    groups: Dict[str, Dict[str, dict]] = {}
    for key, data in metrics.items():
        parts = key.split("/", 1)
        prefix, suffix = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        groups.setdefault(suffix, {})[prefix] = data

    n_plots = len(groups)
    ncols = min(ncols, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    color_map = {"train": "#3A78D4", "val": "#E05C3A", "": "#3A78D4"}

    for ax_idx, (suffix, series_dict) in enumerate(groups.items()):
        ax = axes_flat[ax_idx]
        for phase, data in series_dict.items():
            xs = np.array(data["x"])
            ys = np.array(data["y"])
            color = color_map.get(phase, "#555555")
            label = phase if phase else suffix
            ax.scatter(xs, ys, s=4, alpha=0.35, color=color, zorder=2)
            if len(ys) > 1:
                ax.plot(
                    xs,
                    _ema(ys, alpha=ema_alpha),
                    color=color,
                    linewidth=1.4,
                    label=label,
                    zorder=3,
                )
            else:
                ax.plot(xs, ys, color=color, linewidth=1.4, label=label, zorder=3)
        ax.set_title(suffix, fontsize=9, pad=4, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=8)
        if len(series_dict) > 1:
            ax.legend(fontsize=7, loc="best", framealpha=0.6)

    for ax_idx in range(n_plots, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.01)

    fig.tight_layout()
    _savefig(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_display = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        cm_display = cm.astype(int)
        fmt = "d"

    n = len(class_names)
    cell = max(0.55, 4.5 / n)
    fig, ax = plt.subplots(figsize=(cell * n + 1.5, cell * n + 1.5))
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)

    thresh = cm_display.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = f"{cm_display[i, j]:{fmt}}"
            ax.text(
                j,
                i,
                val,
                ha="center",
                va="center",
                fontsize=max(6, 9 - n // 4),
                color="white" if cm_display[i, j] > thresh else "black",
            )

    ax.set_ylabel("Ground Truth", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    _savefig(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_pr_curve(
    precisions: Dict[str, np.ndarray],
    recalls: Dict[str, np.ndarray],
    ap_scores: Dict[str, float],
    title: str = "Precision-Recall Curve",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    class_names = list(precisions.keys())
    mean_ap = float(np.mean(list(ap_scores.values())))

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, name in enumerate(class_names):
        ap = ap_scores.get(name, float("nan"))
        ax.plot(
            np.array(recalls[name]),
            np.array(precisions[name]),
            color=_cls_color(idx),
            linewidth=1.8,
            label=f"{name}  AP={ap:.3f}",
        )

    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"{title}  |  mAP={mean_ap:.3f}", fontsize=12, fontweight="bold", pad=10
    )
    ax.legend(
        fontsize=7,
        loc="lower left",
        framealpha=0.7,
        ncol=max(1, len(class_names) // 10),
    )
    fig.tight_layout()
    _savefig(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_pf1_curve(
    precisions: Dict[str, np.ndarray],
    f1_scores: Dict[str, np.ndarray],
    ap_scores: Dict[str, float],
    title: str = "Precision-F1 Curve",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    class_names = list(precisions.keys())
    mean_ap = float(np.mean(list(ap_scores.values())))

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, name in enumerate(class_names):
        ap = ap_scores.get(name, float("nan"))
        ax.plot(
            np.array(precisions[name]),
            np.array(f1_scores[name]),
            color=_cls_color(idx),
            linewidth=1.8,
            label=f"{name}  AP={ap:.3f}",
        )

    ax.set_xlabel("Precision", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"{title}  |  mAP={mean_ap:.3f}", fontsize=12, fontweight="bold", pad=10
    )
    ax.legend(
        fontsize=7,
        loc="lower right",
        framealpha=0.7,
        ncol=max(1, len(class_names) // 10),
    )
    fig.tight_layout()
    _savefig(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_rf1_curve(
    recalls: Dict[str, np.ndarray],
    f1_scores: Dict[str, np.ndarray],
    ap_scores: Dict[str, float],
    title: str = "Recall-F1 Curve",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    class_names = list(recalls.keys())
    mean_ap = float(np.mean(list(ap_scores.values())))

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, name in enumerate(class_names):
        ap = ap_scores.get(name, float("nan"))
        ax.plot(
            np.array(recalls[name]),
            np.array(f1_scores[name]),
            color=_cls_color(idx),
            linewidth=1.8,
            label=f"{name}  AP={ap:.3f}",
        )

    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"{title}  |  mAP={mean_ap:.3f}", fontsize=12, fontweight="bold", pad=10
    )
    ax.legend(
        fontsize=7,
        loc="lower right",
        framealpha=0.7,
        ncol=max(1, len(class_names) // 10),
    )
    fig.tight_layout()
    _savefig(fig, save_path)
    if show:
        plt.show()
    return fig


def visualize_predictions(
    image: Union[np.ndarray, Image.Image, str, Path],
    gt_boxes: Optional[List[List[float]]] = None,
    gt_labels: Optional[List[int]] = None,
    pred_boxes: Optional[List[List[float]]] = None,
    pred_labels: Optional[List[int]] = None,
    pred_scores: Optional[List[float]] = None,
    class_names: Optional[List[str]] = None,
    box_format: str = "xyxy",
    score_threshold: float = 0.3,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        image = np.array(image)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    ax.imshow(image)

    def _to_xyxy(box):
        if box_format == "xywh":
            x, y, w, h = box
            return x, y, x + w, y + h
        return box[0], box[1], box[2], box[3]

    def _label_str(idx, score=None):
        name = class_names[idx] if class_names and idx < len(class_names) else str(idx)
        return f"{name} {score:.2f}" if score is not None else name

    if gt_boxes:
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = _to_xyxy(box)
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1.8,
                    edgecolor="#2ECC71",
                    facecolor="none",
                )
            )
            if gt_labels:
                ax.text(
                    x1,
                    y1 - 3,
                    _label_str(gt_labels[i]),
                    fontsize=7,
                    color="white",
                    bbox=dict(
                        boxstyle="square,pad=1", fc="#2ECC71", ec="none", alpha=0.85
                    ),
                )

    if pred_boxes:
        for i, box in enumerate(pred_boxes):
            score = pred_scores[i] if pred_scores else None
            if score is not None and score < score_threshold:
                continue
            x1, y1, x2, y2 = _to_xyxy(box)
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1.8,
                    edgecolor="#E74C3C",
                    facecolor="none",
                    linestyle="--",
                )
            )
            if pred_labels:
                ax.text(
                    x1,
                    y2 + 3,
                    _label_str(pred_labels[i], score),
                    fontsize=7,
                    color="white",
                    bbox=dict(
                        boxstyle="square,pad=1", fc="#E74C3C", ec="none", alpha=0.85
                    ),
                )

    handles = []
    if gt_boxes:
        handles.append(
            mpatches.Patch(
                edgecolor="#2ECC71",
                facecolor="none",
                linewidth=1.8,
                label="Ground Truth",
            )
        )
    if pred_boxes:
        handles.append(
            mpatches.Patch(
                edgecolor="#E74C3C",
                facecolor="none",
                linewidth=1.8,
                linestyle="--",
                label="Prediction",
            )
        )
    if handles:
        ax.legend(
            handles=handles,
            fontsize=8,
            loc="upper right",
            framealpha=0.7,
            handlelength=1.5,
        )

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")

    if standalone:
        fig.tight_layout()
        _savefig(fig, save_path)
        if show:
            plt.show()
    return fig


def visualize_batch_predictions(
    images: List[Union[np.ndarray, Image.Image, str, Path]],
    gt_boxes_list: Optional[List[Optional[List[List[float]]]]] = None,
    gt_labels_list: Optional[List[Optional[List[int]]]] = None,
    pred_boxes_list: Optional[List[Optional[List[List[float]]]]] = None,
    pred_labels_list: Optional[List[Optional[List[int]]]] = None,
    pred_scores_list: Optional[List[Optional[List[float]]]] = None,
    class_names: Optional[List[str]] = None,
    box_format: str = "xyxy",
    score_threshold: float = 0.3,
    titles: Optional[List[str]] = None,
    ncols: int = 3,
    figsize_per_cell: Tuple[float, float] = (5.0, 4.5),
    suptitle: str = "Validation – Ground Truth (green) vs Predictions (red)",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    max_images: int = 12,
    seed: Optional[int] = None,
) -> plt.Figure:
    n_total = len(images)
    indices = list(range(n_total))
    if n_total > max_images:
        rng = random.Random(seed)
        indices = rng.sample(indices, max_images)

    n = len(indices)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    def _get(lst, i):
        return lst[i] if lst is not None and i < len(lst) else None

    for plot_idx, data_idx in enumerate(indices):
        visualize_predictions(
            image=images[data_idx],
            gt_boxes=_get(gt_boxes_list, data_idx),
            gt_labels=_get(gt_labels_list, data_idx),
            pred_boxes=_get(pred_boxes_list, data_idx),
            pred_labels=_get(pred_labels_list, data_idx),
            pred_scores=_get(pred_scores_list, data_idx),
            class_names=class_names,
            box_format=box_format,
            score_threshold=score_threshold,
            title=titles[data_idx] if titles else f"#{data_idx}",
            ax=axes_flat[plot_idx],
            show=False,
        )

    for ax_idx in range(n, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _savefig(fig, save_path)
    if show:
        plt.show()
    return fig


def metrics_from_csv(
    csv_path: Union[str, Path],
    train_phase: str = "train",
    eval_phase: str = "eval",
    x_col: str = "epoch",
) -> Dict[str, Dict[str, List[float]]]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in {csv_path}")

    skip = {"iter", "epoch", "timestamp", "phase"}
    result: Dict[str, Dict[str, List[float]]] = {}

    for phase_val, prefix in [(train_phase, "train"), (eval_phase, "val")]:
        phase_df = df[df["phase"] == phase_val].dropna(subset=[x_col])
        for col in phase_df.columns:
            if col in skip:
                continue
            series = phase_df[[x_col, col]].dropna()
            if series.empty:
                continue
            result[f"{prefix}/{col}"] = {
                "x": series[x_col].tolist(),
                "y": series[col].tolist(),
            }

    return result


class _HookStore:
    def __init__(self):
        self.activation = None
        self.gradient = None
        self._handles: list = []

    def register(self, module) -> "_HookStore":
        self._handles.append(
            module.register_forward_hook(lambda _m, _i, out: self._fwd(out))
        )
        self._handles.append(
            module.register_full_backward_hook(lambda _m, _gi, go: self._bwd(go))
        )
        return self

    def _fwd(self, out):
        t = out[0] if isinstance(out, (list, tuple)) else out
        self.activation = t.detach()

    def _bwd(self, go):
        g = go[0] if isinstance(go, (list, tuple)) else go
        if g is not None:
            self.gradient = g.detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _find_target_module(m, target_layer: str):
    if hasattr(m, "module"):
        m = m.module

    parts = target_layer.split(".")

    for p in parts:
        if p == "model" and not hasattr(m, "model") and hasattr(m, "timm_model"):
            p = "timm_model"

        if p.isdigit():
            m = m[int(p)]
        else:
            m = getattr(m, p)

    return m


def _resolve_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def gradcam_detection(
    model,
    batch: list,
    target_layer: str = "backbone.model.layer4",
    save_path: Optional[Union[str, Path]] = None,
    val_accuracy: Optional[float] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
    colormap: str = "jet",
    alpha: float = 0.45,
    run_tag: str = "",
    show: bool = False,
    compare_layers: bool = True,
    layer_comparison_save_path: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[plt.Figure], np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    raw_model = _resolve_model(model)

    try:
        target_module = _find_target_module(raw_model, target_layer)
    except AttributeError:
        for candidate in [
            "backbone.model.layer4",
            "backbone.layer4",
            "backbone.model.stages.3",
            "backbone.stages.3",
        ]:
            try:
                target_module = _find_target_module(raw_model, candidate)
                target_layer = candidate
                break
            except AttributeError:
                continue
        else:
            raise ValueError(
                "Could not find a suitable backbone layer. "
                "Pass the correct dot-path via `target_layer`."
            )

    hook = _HookStore().register(target_module)
    was_training = model.training
    model.eval()

    batch = batch[:6]
    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    overlays: List[np.ndarray] = []
    subtitles: List[str] = []

    try:
        for sample in batch:
            import torch

            img_tensor = sample["image"]
            H_orig, W_orig = img_tensor.shape[-2], img_tensor.shape[-1]

            with torch.enable_grad():
                outputs = model([sample])

            instances = outputs[0]["instances"]
            scores = instances.scores if instances.has("scores") else None
            boxes = instances.pred_boxes.tensor if instances.has("pred_boxes") else None
            pred_cls = instances.pred_classes if instances.has("pred_classes") else None

            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (
                img_np.astype(np.uint8)
                if img_np.max() > 1.5
                else (img_np * 255).astype(np.uint8)
            )

            if scores is None or len(scores) == 0:
                overlays.append(img_np)
                subtitles.append(f"{Path(sample.get('file_name','')).name}  [no det]")
                hook.activation = hook.gradient = None
                model.zero_grad()
                continue

            scores[: max(1, min(5, len(scores)))].sum().backward(retain_graph=False)

            if hook.activation is None or hook.gradient is None:
                overlays.append(img_np)
                subtitles.append("")
                model.zero_grad()
                hook.activation = hook.gradient = None
                continue

            import torch.nn.functional as F

            act = hook.activation.squeeze(0)
            grad = hook.gradient.squeeze(0)
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = F.relu((weights * act).sum(dim=0))
            cam = (cam - cam.min()) / (cam.max() + 1e-8)
            cam_np = cam.cpu().numpy()

            if boxes is not None and pred_cls is not None:
                feat_map = act.cpu().float()
                fh, fw = feat_map.shape[1], feat_map.shape[2]
                sx, sy = fw / W_orig, fh / H_orig
                keep_mask = scores >= score_threshold
                for bi in range(len(boxes)):
                    if not keep_mask[bi]:
                        continue
                    x1, y1, x2, y2 = boxes[bi].cpu().tolist()
                    rx1 = max(0, int(x1 * sx))
                    ry1 = max(0, int(y1 * sy))
                    rx2 = min(fw, max(rx1 + 1, int(x2 * sx)))
                    ry2 = min(fh, max(ry1 + 1, int(y2 * sy)))
                    vec = feat_map[:, ry1:ry2, rx1:rx2].mean(dim=(1, 2)).numpy()
                    all_features.append(vec)
                    all_labels.append(int(pred_cls[bi].cpu()))

            cam_resized = (
                np.array(
                    Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
                        (W_orig, H_orig), Image.BILINEAR
                    )
                )
                / 255.0
            )
            cmap_fn = plt.get_cmap(colormap)
            heatmap = (cmap_fn(cam_resized)[:, :, :3] * 255).astype(np.uint8)
            overlay = (
                ((1 - alpha) * img_np.astype(float) + alpha * heatmap.astype(float))
                .clip(0, 255)
                .astype(np.uint8)
            )

            import PIL.ImageDraw as ImageDraw

            ov_pil = Image.fromarray(overlay)
            draw = ImageDraw.Draw(ov_pil)
            if boxes is not None and pred_cls is not None:
                keep_mask = scores >= score_threshold
                for bi in range(len(boxes)):
                    if not keep_mask[bi]:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in boxes[bi].cpu().tolist()]
                    lbl = (
                        class_names[int(pred_cls[bi])]
                        if class_names
                        else str(int(pred_cls[bi]))
                    )
                    draw.rectangle([x1, y1, x2, y2], outline="#FF3B30", width=2)
                    draw.text(
                        (x1 + 2, max(0, y1 - 12)),
                        f"{lbl} {float(scores[bi]):.2f}",
                        fill="#FF3B30",
                    )
            overlay = np.array(ov_pil)

            fname = sample.get("file_name", "")
            n_det = int((scores >= score_threshold).sum()) if boxes is not None else 0
            subtitles.append(f"{Path(fname).name}  [{n_det} det]")
            overlays.append(overlay)
            model.zero_grad()
            hook.activation = hook.gradient = None

    finally:
        hook.remove()
        if was_training:
            model.train()

    empty_feats = np.zeros((0,), dtype=np.float32)
    empty_labels = np.zeros((0,), dtype=np.int32)

    if not overlays:
        return None, empty_feats, empty_labels

    acc_str = f"{val_accuracy:.4f}" if val_accuracy is not None else "n/a"

    ncols_g, nrows_g = 3, 2
    fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(6 * ncols_g, 5 * nrows_g))
    axes_flat = axes.flatten()

    for idx in range(ncols_g * nrows_g):
        ax = axes_flat[idx]
        if idx < len(overlays):
            ax.imshow(overlays[idx])
            ax.set_title(subtitles[idx], fontsize=8, pad=4)
        ax.axis("off")

    tag = f"  |  {run_tag}" if run_tag else ""
    fig.suptitle(
        f"GradCAM  |  {target_layer}{tag}  |  val mAP {acc_str}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    feats = np.stack(all_features, axis=0) if all_features else empty_feats
    labels = np.array(all_labels, dtype=np.int32) if all_labels else empty_labels


    if compare_layers:
        lc_path: Optional[Path] = None
        if layer_comparison_save_path is not None:
            lc_path = Path(layer_comparison_save_path)
        elif save_path is not None:
            p = Path(save_path)
            lc_path = p.with_stem(p.stem + "_layer_comparison")
        try:
            gradcam_layer_comparison(
                model=model,
                batch=batch,
                save_path=lc_path,
                val_accuracy=val_accuracy,
                class_names=class_names,
                score_threshold=score_threshold,
                colormap=colormap,
                alpha=alpha,
                run_tag=run_tag,
                show=show,
            )
        except Exception as _lc_err:
            print(f"[gradcam_detection] layer comparison skipped: {_lc_err}")

    return fig, feats, labels


def _run_gradcam_single(
    model,
    sample: dict,
    target_module,
    score_threshold: float = 0.3,
    colormap: str = "jet",
    alpha: float = 0.45,
    class_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, str]:
    import torch
    import torch.nn.functional as F

    hook = _HookStore().register(target_module)
    try:
        img_tensor = sample["image"]
        H_orig, W_orig = img_tensor.shape[-2], img_tensor.shape[-1]

        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (
            img_np.astype(np.uint8)
            if img_np.max() > 1.5
            else (img_np * 255).astype(np.uint8)
        )

        model.zero_grad()
        with torch.enable_grad():
            outputs = model([sample])

        instances = outputs[0]["instances"]
        scores = instances.scores if instances.has("scores") else None
        boxes = instances.pred_boxes.tensor if instances.has("pred_boxes") else None
        pred_cls = instances.pred_classes if instances.has("pred_classes") else None

        fname = Path(sample.get("file_name", "")).name
        n_det = int((scores >= score_threshold).sum()) if scores is not None else 0
        subtitle = f"{fname}  [{n_det} det]"

        if scores is None or len(scores) == 0 or hook.activation is None:
            hook.remove()
            return img_np, f"{fname}  [no det]"

        scores[: max(1, min(5, len(scores)))].sum().backward(retain_graph=False)

        if hook.activation is None or hook.gradient is None:
            hook.remove()
            return img_np, subtitle

        act = hook.activation.squeeze(0)
        grad = hook.gradient.squeeze(0)
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * act).sum(dim=0))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()

        cam_resized = (
            np.array(
                Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
                    (W_orig, H_orig), Image.BILINEAR
                )
            )
            / 255.0
        )
        cmap_fn = plt.get_cmap(colormap)
        heatmap = (cmap_fn(cam_resized)[:, :, :3] * 255).astype(np.uint8)
        overlay = (
            ((1 - alpha) * img_np.astype(float) + alpha * heatmap.astype(float))
            .clip(0, 255)
            .astype(np.uint8)
        )

        import PIL.ImageDraw as ImageDraw

        ov_pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(ov_pil)
        if boxes is not None and pred_cls is not None:
            keep_mask = scores >= score_threshold
            for bi in range(len(boxes)):
                if not keep_mask[bi]:
                    continue
                x1, y1, x2, y2 = [int(v) for v in boxes[bi].cpu().tolist()]
                lbl = (
                    class_names[int(pred_cls[bi])]
                    if class_names
                    else str(int(pred_cls[bi]))
                )
                draw.rectangle([x1, y1, x2, y2], outline="#FF3B30", width=2)
                draw.text(
                    (x1 + 2, max(0, y1 - 12)),
                    f"{lbl} {float(scores[bi]):.2f}",
                    fill="#FF3B30",
                )
        overlay = np.array(ov_pil)
    finally:
        hook.remove()
        model.zero_grad()

    return overlay, subtitle



_LAYER_CANDIDATES = {
    "layer1": [
        "backbone.model.layer1",
        "backbone.layer1",
        "backbone.model.stages.0",
        "backbone.stages.0",
    ],
    "layer2": [
        "backbone.model.layer2",
        "backbone.layer2",
        "backbone.model.stages.1",
        "backbone.stages.1",
    ],
    "layer3": [
        "backbone.model.layer3",
        "backbone.layer3",
        "backbone.model.stages.2",
        "backbone.stages.2",
    ],
    "layer4": [
        "backbone.model.layer4",
        "backbone.layer4",
        "backbone.model.stages.3",
        "backbone.stages.3",
    ],
}


def gradcam_layer_comparison(
    model,
    batch: list,
    layer_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    val_accuracy: Optional[float] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
    colormap: str = "jet",
    alpha: float = 0.45,
    run_tag: str = "",
    max_samples: int = 4,
    show: bool = False,
) -> Optional[plt.Figure]:
    import torch

    raw_model = _resolve_model(model)


    if layer_names is None:
        layer_names = []
        for key in ("layer1", "layer2", "layer3", "layer4"):
            for candidate in _LAYER_CANDIDATES[key]:
                try:
                    _find_target_module(raw_model, candidate)
                    layer_names.append(candidate)
                    break
                except AttributeError:
                    continue
            else:
                print(f"[gradcam_layer_comparison] WARNING: could not find {key}")

    if not layer_names:
        raise ValueError(
            "No backbone layers found. Pass explicit `layer_names`."
        )

    target_modules = []
    for ln in layer_names:
        target_modules.append(_find_target_module(raw_model, ln))


    short_names = [ln.split(".")[-1] for ln in layer_names]

    samples = batch[: min(max_samples, 6)]
    was_training = model.training
    model.eval()


    n_rows = len(samples)
    n_cols = len(layer_names)

    try:

        all_overlays: List[List[np.ndarray]] = []
        all_subtitles: List[str] = []

        for si, sample in enumerate(samples):
            row_overlays: List[np.ndarray] = []
            row_subtitle = ""
            for li, (ln, tm) in enumerate(zip(layer_names, target_modules)):
                overlay, subtitle = _run_gradcam_single(
                    model,
                    sample,
                    tm,
                    score_threshold=score_threshold,
                    colormap=colormap,
                    alpha=alpha,
                    class_names=class_names,
                )
                row_overlays.append(overlay)
                if li == 0:
                    row_subtitle = subtitle
            all_overlays.append(row_overlays)
            all_subtitles.append(row_subtitle)

    finally:
        if was_training:
            model.train()

    if not all_overlays:
        return None

    cell_w, cell_h = 5, 4
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        squeeze=False,
    )

    for ri in range(n_rows):
        for ci in range(n_cols):
            ax = axes[ri][ci]
            ax.imshow(all_overlays[ri][ci])
            ax.axis("off")

            if ri == 0:
                ax.set_title(short_names[ci], fontsize=11, fontweight="bold", pad=6)

        axes[ri][0].set_ylabel(
            all_subtitles[ri], fontsize=7, labelpad=4, rotation=90, va="center"
        )

    acc_str = f"{val_accuracy:.4f}" if val_accuracy is not None else "n/a"
    tag = f"  |  {run_tag}" if run_tag else ""
    fig.suptitle(
        f"GradCAM Layer Comparison{tag}  |  val mAP {acc_str}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return fig


def _next_index(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    return len(list(directory.glob("*.png")))


def plot_tsne_from_features(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "t-SNE Feature Space",
    perplexity: float = 35.0,
    n_iter: int = 2000,
    pca_components: Optional[int] = 64,
    marker_size: int = 65,
    marker_alpha: float = 0.82,
    kde_alpha: float = 0.18,
    kde_levels: int = 6,
    show_centroids: bool = True,
    centroid_fontsize: int = 14,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 150,
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    from sklearn.manifold import TSNE

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must be a non-empty 2-D array (N, C)")

    n_classes = int(labels.max()) + 1 if len(labels) else 0
    names = (
        class_names[:n_classes] if class_names else [str(i) for i in range(n_classes)]
    )
    palette = list(plt.get_cmap("tab10").colors)
    colors = [palette[i % len(palette)] for i in range(n_classes)]

    if pca_components is not None and features.shape[1] > pca_components:
        from sklearn.decomposition import PCA

        features = PCA(
            n_components=pca_components, random_state=random_state
        ).fit_transform(features)

    perp = min(perplexity, max(5.0, len(features) / 3.0))
    emb = TSNE(
        n_components=2,
        perplexity=perp,
        n_iter=n_iter,
        metric="euclidean",
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features)

    matplotlib.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "font.family": "DejaVu Sans",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#F7F8FA")
    fig.patch.set_facecolor("#FFFFFF")

    try:
        from scipy.stats import gaussian_kde

        for ci in range(n_classes):
            mask = labels == ci
            if mask.sum() < 10:
                continue
            kde = gaussian_kde(emb[mask].T, bw_method=0.25)
            xr = np.linspace(emb[:, 0].min() - 5, emb[:, 0].max() + 5, 200)
            yr = np.linspace(emb[:, 1].min() - 5, emb[:, 1].max() + 5, 200)
            Xi, Yi = np.meshgrid(xr, yr)
            Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            ax.contourf(
                Xi, Yi, Zi, levels=kde_levels, colors=[colors[ci]], alpha=kde_alpha
            )
    except Exception:
        pass

    for ci in range(n_classes):
        mask = labels == ci
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=[colors[ci]],
            s=marker_size,
            alpha=marker_alpha,
            edgecolors="white",
            linewidths=0.5,
            label=names[ci] if ci < len(names) else str(ci),
            zorder=3,
        )

    if show_centroids:
        for ci in range(n_classes):
            mask = labels == ci
            if not mask.any():
                continue
            cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
            name = names[ci] if ci < len(names) else str(ci)
            ax.text(
                cx,
                cy,
                name,
                fontsize=centroid_fontsize,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc=colors[ci], ec="none", alpha=0.9
                ),
                zorder=5,
            )

    legend = ax.legend(
        title="Class",
        fontsize=10,
        title_fontsize=11,
        loc="upper right",
        framealpha=0.92,
        edgecolor="#CCCCCC",
        markerscale=1.4,
        handletextpad=0.5,
        borderpad=0.7,
    )
    legend.get_frame().set_linewidth(0.8)

    ax.set_xlabel("t-SNE dim 1", fontsize=12, fontweight="bold", labelpad=6)
    ax.set_ylabel("t-SNE dim 2", fontsize=12, fontweight="bold", labelpad=6)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#CCCCCC")

    ax.annotate(
        f"n={len(labels):,}  |  {n_classes} classes  |  perplexity={perp:.0f}",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=9,
        color="#555555",
        va="bottom",
    )

    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(
            p.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor()
        )

    if show:
        plt.show()

    return fig


def plot_tsne(
    features_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
    topk: Optional[int] = None,
    perplexity: float = 35.0,
    n_iter: int = 2000,
    pca_components: Optional[int] = 64,
    marker_size: int = 65,
    marker_alpha: float = 0.82,
    kde_alpha: float = 0.18,
    kde_levels: int = 6,
    show_centroids: bool = True,
    centroid_fontsize: int = 14,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 300,
    random_state: int = 42,
    show: bool = False,
) -> List[plt.Figure]:
    features_dir = Path(features_dir)
    if not features_dir.is_dir():
        raise ValueError(f"features_dir does not exist: {features_dir}")

    npz_files = sorted(features_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files found in {features_dir}")

    if topk is not None and topk > 0:
        npz_files = npz_files[-topk:]

    parts = features_dir.resolve().parts
    run = parts[-2] if len(parts) >= 2 else "run-0"
    bbone = parts[-3] if len(parts) >= 3 else "backbone"
    arch = parts[-4] if len(parts) >= 4 else "model"

    tsne_dir = features_dir.parent / "tsne"
    tsne_dir.mkdir(parents=True, exist_ok=True)

    all_figs: List[plt.Figure] = []

    for npz_path in npz_files:
        stem = npz_path.stem
        d = np.load(npz_path)
        feats = d["features"].astype(np.float32)
        labels = d["labels"].astype(np.int32)

        parts_stem = stem.split("_")
        extra = ""
        for part in parts_stem:
            try:
                int(part)
                extra = f"  |  iter {int(part):,}"
                break
            except ValueError:
                continue
        try:
            extra += f"  |  mAP {float(parts_stem[-1]):.4f}"
        except (ValueError, IndexError):
            pass

        title = f"t-SNE Feature Space  |  {arch} / {bbone} / run {run}{extra}  |  n={len(labels):,}"

        fig = plot_tsne_from_features(
            features=feats,
            labels=labels,
            class_names=class_names,
            title=title,
            perplexity=perplexity,
            n_iter=n_iter,
            pca_components=pca_components,
            marker_size=marker_size,
            marker_alpha=marker_alpha,
            kde_alpha=kde_alpha,
            kde_levels=kde_levels,
            show_centroids=show_centroids,
            centroid_fontsize=centroid_fontsize,
            figsize=figsize,
            dpi=dpi,
            random_state=random_state,
            save_path=tsne_dir / f"{stem}.png",
            show=show,
        )
        all_figs.append(fig)
        plt.close(fig)

    return all_figs


def visualize_from_yaml(
    yaml_path: Union[str, Path],
    test_images_dir: Optional[Union[str, Path]] = None,
    n_images: int = 6,
    target_layer: str = "backbone.model.layer4",
    score_threshold: float = 0.3,
    seed: Optional[int] = None,
    show: bool = False,
) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    import random as _random
    import torch
    import yaml as _yaml

    yaml_path = Path(yaml_path)
    cfg_yaml = _yaml.safe_load(yaml_path.open())


    model_config_path = cfg_yaml["model"]["config"]
    checkpoint_path = cfg_yaml["checkpoint"]
    class_names = cfg_yaml["dataset"]["class_names"]
    img_dir = Path(test_images_dir or cfg_yaml["dataset"]["test_images_dir"])



    parts = yaml_path.parts
    try:
        test_idx = list(parts).index("test")
        arch = parts[test_idx + 1]
        backbone = parts[test_idx + 2]
        run = yaml_path.stem
    except (ValueError, IndexError):
        arch, backbone, run = "model", "backbone", "0"

    chart_base = Path("charts") / arch / backbone / run
    gradcam_dir = chart_base / "gradcam"
    tsne_dir = chart_base / "tsne"

    idx = _next_index(gradcam_dir)
    gradcam_save = gradcam_dir / f"{idx}.png"
    tsne_save = tsne_dir / f"{idx}.png"


    from detectron2.config import LazyConfig, instantiate
    from detectron2.checkpoint import DetectionCheckpointer

    cfg = LazyConfig.load(model_config_path)
    model = instantiate(cfg.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    DetectionCheckpointer(model).load(checkpoint_path)


    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    all_imgs = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in exts]
    if not all_imgs:
        raise ValueError(f"No images found in {img_dir}")

    rng = _random.Random(seed)
    sampled = rng.sample(all_imgs, min(n_images, len(all_imgs)))

    batch: list = []
    for i, img_path in enumerate(sampled):
        try:
            import cv2

            img = cv2.imread(str(img_path))
            if img is None:
                raise OSError(f"cv2 could not read {img_path}")
        except ImportError:
            img_pil = Image.open(img_path).convert("RGB")
            img = np.array(img_pil)[:, :, ::-1].copy()

        H, W = img.shape[:2]
        img_tensor = torch.as_tensor(img.transpose(2, 0, 1).astype("float32"))
        batch.append(
            {
                "image": img_tensor,
                "height": H,
                "width": W,
                "file_name": str(img_path),
                "image_id": i,
            }
        )


    run_tag = f"{arch}  /  {backbone}  /  run {run}"
    gradcam_fig, features, labels = gradcam_detection(
        model=model,
        batch=batch,
        target_layer=target_layer,
        save_path=gradcam_save,
        class_names=class_names,
        score_threshold=score_threshold,
        run_tag=run_tag,
        show=show,
    )


    tsne_fig: Optional[plt.Figure] = None
    if features.ndim == 2 and len(features) >= 2:
        title = (
            f"t-SNE Feature Space  |  {run_tag}"
            f"  |  GradCAM run {idx}  |  n={len(labels):,}"
        )
        tsne_fig = plot_tsne_from_features(
            features=features,
            labels=labels,
            class_names=class_names,
            title=title,
            save_path=tsne_save,
            show=show,
        )
    else:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "visualize_from_yaml: not enough detections for t-SNE "
            "(need ≥ 2 detections, got %d).",
            len(features),
        )

    return gradcam_fig, tsne_fig
