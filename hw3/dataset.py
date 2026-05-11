"""Dataset loaders. Two thin torch.utils.data.Dataset wrappers (required by
DataLoader) plus pure data-loading functions that do all the real work.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset

from augmentation import (
    build_augmentation_pipeline,
    apply_augmentation,
    copy_paste_augment,
    compute_oversampling_weights,
)
from io_utils import load_image_rgb

NUM_CLASSES = 4
CLASS_NAMES = ["class1", "class2", "class3", "class4"]

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────
#  Pure functional loaders
# ──────────────────────────────────────────────────────────────────────────

def parse_instance_mask(mask: np.ndarray, cls_id: int) -> List[Dict]:
    """Decode a class-K mask volume (positive int = instance ID) into instances."""
    instances = []
    for iid in np.unique(mask):
        if iid <= 0:
            continue
        binary = (mask == iid).astype(np.uint8)
        rows, cols = np.where(binary)
        if len(rows) == 0:
            continue
        x, y = int(cols.min()), int(rows.min())
        w, h = int(cols.max()) - x + 1, int(rows.max()) - y + 1
        instances.append({
            "category_id": cls_id,
            "instance_id": int(iid),
            "mask": binary,
            "bbox": (float(x), float(y), float(w), float(h)),
            "area": float(binary.sum()),
        })
    return instances


def load_train_sample(sample_dir: str, sample_id: str) -> Dict:
    """Load one training sample directory: image + per-class instance masks."""
    image = load_image_rgb(os.path.join(sample_dir, "image.tif"))
    H, W = image.shape[:2]
    instances: List[Dict] = []
    for cls_id in range(1, NUM_CLASSES + 1):
        mask_path = os.path.join(sample_dir, f"class{cls_id}.tif")
        if os.path.exists(mask_path):
            mask = tifffile.imread(mask_path).astype(np.int32)
            instances.extend(parse_instance_mask(mask, cls_id))
    return {"image_id": sample_id, "image": image, "height": H, "width": W, "instances": instances}


def normalize_image(image_uint8: np.ndarray) -> np.ndarray:
    return (image_uint8.astype(np.float32) - _MEAN) / _STD


def build_source_pool(data_dir: str, sample_ids: List[str], pool_size: int = 8) -> List[Dict]:
    """Pre-load a small cache of training samples for copy-paste."""
    pool = []
    for sid in sample_ids[:pool_size]:
        s = load_train_sample(os.path.join(data_dir, sid), sid)
        pool.append({
            "id": sid,
            "image": s["image"],
            "masks": [inst["mask"] for inst in s["instances"]],
            "category_ids": [inst["category_id"] for inst in s["instances"]],
        })
    return pool


def encode_train_item(sample: Dict, pipeline: dict, source_pool: Optional[List[Dict]] = None) -> Dict:
    """Apply augmentation + format into the dict consumed by mmdet during training."""
    sid = sample["image_id"]
    image = sample["image"]
    masks = [inst["mask"] for inst in sample["instances"]]
    category_ids = [inst["category_id"] for inst in sample["instances"]]

    # Copy-paste (train only)
    cp_kwargs = pipeline.get("copy_paste_kwargs")
    if cp_kwargs is not None and source_pool is not None:
        pool = [p for p in source_pool if p["id"] != sid]
        image, masks, category_ids = copy_paste_augment(image, masks, category_ids, pool, **cp_kwargs)

    if masks:
        result = apply_augmentation(pipeline, image, masks, category_ids)
        image = result["image"]
        masks = result["masks"]
        category_ids = result["category_ids"]

    gt_bboxes, gt_labels, gt_masks_bin = [], [], []
    for m, cat in zip(masks, category_ids):
        m = np.asarray(m, dtype=np.uint8)
        rows, cols = np.where(m > 0)
        if len(rows) == 0:
            continue
        x1, y1 = int(cols.min()), int(rows.min())
        x2, y2 = int(cols.max() + 1), int(rows.max() + 1)
        gt_bboxes.append([x1, y1, x2, y2])
        gt_labels.append(cat - 1)
        gt_masks_bin.append(m)

    H, W = image.shape[:2]
    image_norm = normalize_image(image)

    return {
        "image_id": sid,
        "img": torch.from_numpy(image_norm).permute(2, 0, 1),
        "gt_bboxes": torch.tensor(gt_bboxes, dtype=torch.float32) if gt_bboxes else torch.zeros((0, 4)),
        "gt_labels": torch.tensor(gt_labels, dtype=torch.long) if gt_labels else torch.zeros((0,), dtype=torch.long),
        "gt_masks": gt_masks_bin,
        "img_metas": {
            "img_shape": (H, W, 3),
            "pad_shape": (H, W, 3),
            "ori_shape": (H, W, 3),
            "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "flip": False,
            "filename": sid,
            "batch_input_shape": (H, W),
        },
    }


def encode_test_item(image: np.ndarray, image_id, file_name: str, img_size: int = 1024) -> Dict:
    """Letterbox-resize a test image to img_size x img_size; produce mmdet input."""
    import cv2
    ori_H, ori_W = image.shape[:2]
    scale = img_size / max(ori_H, ori_W)
    new_H, new_W = int(ori_H * scale), int(ori_W * scale)
    image_resized = cv2.resize(image, (new_W, new_H))
    pad_h = img_size - new_H
    pad_w = img_size - new_W
    image_padded = np.pad(image_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    return {
        "image_id": image_id,
        "file_name": file_name,
        "img": torch.from_numpy(image_padded).permute(2, 0, 1).float(),
        "img_metas": {
            "img_shape": (img_size, img_size, 3),
            "ori_shape": (ori_H, ori_W, 3),
            "scale_factor": np.array([scale, scale, scale, scale]),
            "flip": False,
            "filename": file_name,
        },
    }


def get_per_sample_class_counts(data_dir: str, sample_ids: List[str]) -> List[List[int]]:
    """For oversampling: count instances per class for every training sample."""
    counts = []
    for sid in sample_ids:
        s = load_train_sample(os.path.join(data_dir, sid), sid)
        per_class = [0] * NUM_CLASSES
        for inst in s["instances"]:
            per_class[inst["category_id"] - 1] += 1
        counts.append(per_class)
    return counts


# ──────────────────────────────────────────────────────────────────────────
#  Thin Dataset wrappers (required by torch DataLoader)
# ──────────────────────────────────────────────────────────────────────────

class CellInstanceDataset(Dataset):
    """Train/val dataset. Reads raw samples from disk on __getitem__."""

    def __init__(self, data_dir: str, sample_ids: List[str],
                 augmentation_cfg: Optional[dict] = None, mode: str = "train",
                 source_pool_size: int = 8):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.sample_ids = sample_ids or sorted(
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        )
        self.pipeline = build_augmentation_pipeline(augmentation_cfg or {}, mode=mode)
        self._pool: List[Dict] = []
        if mode == "train" and self.pipeline.get("copy_paste_kwargs") is not None:
            self._pool = build_source_pool(str(self.data_dir), self.sample_ids, source_pool_size)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        sid = self.sample_ids[idx]
        sample = load_train_sample(str(self.data_dir / sid), sid)
        return encode_train_item(sample, self.pipeline, self._pool if self._pool else None)

    def get_oversampling_weights(self) -> np.ndarray:
        counts = get_per_sample_class_counts(str(self.data_dir), self.sample_ids)
        return compute_oversampling_weights(counts, NUM_CLASSES)


class CellTestDataset(Dataset):
    """Test dataset: reads test_image_name_to_ids.json and letterbox-resizes images."""

    def __init__(self, data_dir: str, mapping_json: str, img_size: int = 1024):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        with open(mapping_json) as f:
            self.meta = json.load(f)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict:
        m = self.meta[idx]
        img_path = self.data_dir / m["file_name"]
        image = load_image_rgb(str(img_path))
        return encode_test_item(image, m["id"], m["file_name"], img_size=self.img_size)
