"""IO helpers: image loading, JSON, mask RLE, checkpoints, train/val split.

All functions are stateless. Use directly without instantiating any class.
"""
import json
import os
import random
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
import tifffile


def load_tif(path: str) -> np.ndarray:
    return tifffile.imread(path)


def load_image_rgb(path: str) -> np.ndarray:
    """Load image (TIF / PNG / JPG) and return uint8 RGB (H,W,3)."""
    img = tifffile.imread(path) if path.endswith(".tif") else cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img.astype(np.uint8)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def encode_mask_rle(binary_mask: np.ndarray) -> dict:
    from pycocotools import mask as coco_mask
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_mask_rle(rle: dict) -> np.ndarray:
    from pycocotools import mask as coco_mask
    if isinstance(rle["counts"], str):
        rle = dict(rle)
        rle["counts"] = rle["counts"].encode("utf-8")
    return coco_mask.decode(rle).astype(np.uint8)


def save_checkpoint(state: dict, path: str) -> None:
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    import torch
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def scan_dir(data_dir: str) -> List[str]:
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )


def train_val_split(sample_ids: List[str], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = list(sample_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]


def stratified_val_split(
    sample_ids: List[str],
    train_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_classes: int = 4,
) -> Tuple[List[str], List[str]]:
    """Stratified split — guarantees each class has roughly val_ratio in val."""
    rng = random.Random(seed)
    class_to_ids: dict = {c: [] for c in range(1, num_classes + 1)}
    for sid in sample_ids:
        for c in range(1, num_classes + 1):
            if (Path(train_dir) / sid / f"class{c}.tif").exists():
                class_to_ids[c].append(sid)

    val_set: set = set()
    for ids in class_to_ids.values():
        if not ids:
            continue
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n = max(1, round(len(shuffled) * val_ratio))
        val_set.update(shuffled[:n])

    val_ids = [s for s in sample_ids if s in val_set]
    train_ids = [s for s in sample_ids if s not in val_set]
    return train_ids, val_ids


def build_output_paths(cfg: dict) -> dict:
    """Resolve checkpoint_dir / chart_dir relative to base_dir; create them."""
    base = Path(cfg.get("paths", {}).get("base_dir", "."))
    out: dict = {}
    for k in ("checkpoint_dir", "chart_dir", "checkpoint_path"):
        v = cfg.get("paths", {}).get(k)
        if v:
            p = base / v if not os.path.isabs(v) else Path(v)
            out[k] = str(p)
            if k != "checkpoint_path":
                p.mkdir(parents=True, exist_ok=True)
    return out
