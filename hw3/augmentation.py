"""Albumentations pipelines, copy-paste augmentation, oversampling weights.

All functional — no classes. The augmentation "pipeline" is a dict of callables
returned by `build_augmentation_pipeline()` and applied via `apply_augmentation()`.
"""
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# ──────────────────────────────────────────────────────────────────────────
#  Albumentations transforms
# ──────────────────────────────────────────────────────────────────────────

def build_train_transform(cfg: dict):
    assert HAS_ALBUMENTATIONS, "albumentations is required"
    img_size = cfg.get("img_size", 1024)
    min_scale = cfg.get("min_scale", 0.5)
    max_scale = cfg.get("max_scale", 2.0)

    transforms = [
        A.RandomScale(scale_limit=(min_scale - 1.0, max_scale - 1.0), p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.RandomCrop(height=img_size, width=img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
        A.RandomToneCurve(scale=0.1, p=0.2),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        A.ImageCompression(quality_range=(60, 100), p=0.2),
    ]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=0.3),
        additional_targets={"masks": "masks"},
    )


def build_val_transform(cfg: dict):
    assert HAS_ALBUMENTATIONS, "albumentations is required"
    img_size = cfg.get("img_size", 1024)
    return A.Compose(
        [A.LongestMaxSize(max_size=img_size),
         A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
        additional_targets={"masks": "masks"},
    )


# ──────────────────────────────────────────────────────────────────────────
#  Copy-paste augmentation (rare-class boost)
# ──────────────────────────────────────────────────────────────────────────

def copy_paste_augment(
    image: np.ndarray,
    masks: List[np.ndarray],
    category_ids: List[int],
    source_pool: List[Dict],
    paste_prob: float = 0.5,
    max_paste: int = 4,
    rare_classes: Optional[List[int]] = None,
    rare_boost: float = 3.0,
) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    """Paste 1..max_paste rare-class instances from source_pool into image.

    source_pool is a list of dicts with keys: image, masks, category_ids.
    """
    if random.random() > paste_prob or not source_pool:
        return image, masks, category_ids

    rare_set = set(rare_classes or [])
    src = random.choice(source_pool)
    src_image = src["image"]
    src_masks = src["masks"]
    src_cats = src["category_ids"]
    H, W = image.shape[:2]

    # Resize/pad source to match host
    if src_image.shape[:2] != (H, W):
        scale = min(H / src_image.shape[0], W / src_image.shape[1])
        nh, nw = int(src_image.shape[0] * scale), int(src_image.shape[1] * scale)
        src_image = cv2.resize(src_image, (nw, nh))
        src_masks = [cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST) for m in src_masks]
        pad_h, pad_w = H - nh, W - nw
        src_image = cv2.copyMakeBorder(src_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        src_masks = [cv2.copyMakeBorder(m, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0) for m in src_masks]

    weights = [rare_boost if cat in rare_set else 1.0 for cat in src_cats]
    if not weights:
        return image, masks, category_ids

    n_paste = random.randint(1, max_paste)
    total = sum(weights)
    probs = [w / total for w in weights]
    chosen = np.random.choice(len(src_masks), size=min(n_paste, len(src_masks)),
                              replace=False, p=probs)

    image = image.copy()
    new_masks = list(masks)
    new_cats = list(category_ids)

    for idx in chosen:
        paste_mask = src_masks[idx].astype(bool)
        if not paste_mask.any():
            continue
        kernel = np.ones((5, 5), np.uint8)
        border = cv2.dilate(paste_mask.astype(np.uint8), kernel) - paste_mask.astype(np.uint8)
        alpha = paste_mask.astype(np.float32)
        alpha[border.astype(bool)] = 0.7
        for c in range(image.shape[2]):
            image[:, :, c] = (image[:, :, c] * (1 - alpha) + src_image[:, :, c] * alpha).astype(np.uint8)
        new_masks.append(paste_mask.astype(np.uint8))
        new_cats.append(src_cats[idx])

    return image, new_masks, new_cats


# ──────────────────────────────────────────────────────────────────────────
#  Class-balanced oversampling weights
# ──────────────────────────────────────────────────────────────────────────

def compute_oversampling_weights(sample_class_counts: List[List[int]], num_classes: int = 4) -> np.ndarray:
    """Per-sample weight = max(1/N_c) over classes present in the sample."""
    counts = np.array(sample_class_counts)
    total_per_class = np.maximum(counts.sum(axis=0).astype(float), 1)
    class_weights = 1.0 / total_per_class
    class_weights /= class_weights.sum()
    weights = []
    for row in sample_class_counts:
        present = [class_weights[i] for i, c in enumerate(row) if c > 0]
        weights.append(max(present) if present else 0.0)
    return np.array(weights, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────
#  Pipeline build / apply
# ──────────────────────────────────────────────────────────────────────────

def build_augmentation_pipeline(cfg: dict, mode: str = "train") -> dict:
    """Return a dict {transform, copy_paste_kwargs} consumed by apply_augmentation."""
    assert mode in ("train", "val", "test")
    transform = (build_train_transform(cfg) if mode == "train"
                 else build_val_transform(cfg)) if HAS_ALBUMENTATIONS else None

    cp_cfg = cfg.get("copy_paste", {})
    use_cp = mode == "train" and cp_cfg.get("enabled", True)
    cp_kwargs = dict(
        paste_prob=cp_cfg.get("prob", 0.5),
        max_paste=cp_cfg.get("max_paste", 4),
        rare_classes=cp_cfg.get("rare_classes", [4]),
        rare_boost=cp_cfg.get("rare_boost", 3.0),
    ) if use_cp else None

    return {"transform": transform, "copy_paste_kwargs": cp_kwargs, "mode": mode}


def apply_augmentation(
    pipeline: dict,
    image: np.ndarray,
    masks: List[np.ndarray],
    category_ids: List[int],
    bboxes_xyxy: Optional[List[List[float]]] = None,
) -> dict:
    """Apply the Albumentations transform once. Returns dict(image, masks, category_ids, bboxes)."""
    transform = pipeline.get("transform")
    if transform is None:
        return {"image": image, "masks": masks, "category_ids": category_ids, "bboxes": []}

    if bboxes_xyxy is None:
        bboxes_xyxy = []
        for m in masks:
            rows, cols = np.where(m > 0)
            if len(rows) == 0:
                bboxes_xyxy.append([0, 0, 1, 1])
            else:
                bboxes_xyxy.append([float(cols.min()), float(rows.min()),
                                    float(cols.max() + 1), float(rows.max() + 1)])

    H, W = image.shape[:2]
    clipped, valid_idx = [], []
    for i, box in enumerate(bboxes_xyxy):
        x1, y1, x2, y2 = box
        x1, x2 = np.clip([x1, x2], 0, W)
        y1, y2 = np.clip([y1, y2], 0, H)
        if x2 > x1 and y2 > y1:
            clipped.append([x1, y1, x2, y2])
            valid_idx.append(i)
    masks = [masks[i] for i in valid_idx]
    category_ids = [category_ids[i] for i in valid_idx]

    try:
        result = transform(image=image, masks=masks, bboxes=clipped, category_ids=category_ids)
    except Exception:
        result = {"image": image, "masks": masks, "category_ids": category_ids}

    return {
        "image": result["image"],
        "masks": result.get("masks", masks),
        "category_ids": category_ids,
        "bboxes": result.get("bboxes", []),
    }
