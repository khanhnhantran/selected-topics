"""Inference entry point — pure functions, no classes.

Usage:
    python test.py --config TEST_CFG.yaml [--checkpoint CKPT] [--gpu_ids 0]
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CellTestDataset
from io_utils import build_output_paths, load_checkpoint, encode_mask_rle
from models import build_model_cfg, build_mmdet_model

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_COLORS_BGR = [(50, 50, 220), (50, 180, 50), (220, 100, 50), (30, 140, 230)]
_CLASS_NAMES = ["class1", "class2", "class3", "class4"]


# ──────────────────────────────────────────────────────────────────────────
#  CLI / setup
# ──────────────────────────────────────────────────────────────────────────

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Instance Segmentation Inference")
    p.add_argument("--test_folder", default="data/test_release")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--mapping_json", default="data/test_image_name_to_ids.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--score_thr", type=float, default=0.05)
    return p


def load_image_id_mapping(mapping_json: str):
    with open(mapping_json) as f:
        mapping_list = json.load(f)
    name_to_id = {item["file_name"]: item["id"] for item in mapping_list}
    size_dict = {item["file_name"]: (item["height"], item["width"]) for item in mapping_list}
    return name_to_id, size_dict


# ──────────────────────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────────────────────

def build_test_model(cfg: dict, device: str, ckpt_override: str = None) -> torch.nn.Module:
    model_cfg = build_model_cfg(cfg["model"])
    model = build_mmdet_model(model_cfg, device)

    ckpt_path = ckpt_override or cfg.get("paths", {}).get("checkpoint_path", "")
    if not ckpt_path or not os.path.exists(ckpt_path):
        # fall back to checkpoint_dir/best.pth
        ckpt_dir = cfg.get("paths", {}).get("checkpoint_dir", "")
        best = os.path.join(ckpt_dir, "best.pth")
        if os.path.exists(best):
            ckpt_path = best
        else:
            raise FileNotFoundError("Checkpoint not found. Set paths.checkpoint_path.")

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded weights from {ckpt_path}")
    return model


def build_test_loader(cfg: dict, args: argparse.Namespace) -> DataLoader:
    img_size = int(cfg.get("data", {}).get("img_size", 1024))
    num_workers = int(cfg.get("data", {}).get("num_workers", 4))
    ds = CellTestDataset(args.test_folder, args.mapping_json, img_size=img_size)
    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=num_workers, collate_fn=lambda x: x)


# ──────────────────────────────────────────────────────────────────────────
#  Inference per image
# ──────────────────────────────────────────────────────────────────────────

def normalize_test_image(img_uint8: torch.Tensor) -> torch.Tensor:
    img_np = img_uint8.permute(1, 2, 0).numpy().astype(np.float32)
    img_np = (img_np - _MEAN) / _STD
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)


def predict_image(model, item: Dict, device: str, score_thr: float) -> Dict:
    """Run forward pass + filter by score threshold, return numpy results."""
    from mmdet.structures import DetDataSample

    img = normalize_test_image(item["img"]).to(device)
    h, w = img.shape[-2:]

    raw_metas = dict(item["img_metas"])
    raw_metas.setdefault("batch_input_shape", (h, w))
    raw_metas.setdefault("img_shape", (h, w))
    raw_metas.setdefault("ori_shape", (h, w))
    sf = raw_metas.get("scale_factor")
    if sf is not None:
        sf = np.array(sf).reshape(-1)
        raw_metas["scale_factor"] = sf[:2].tolist() if len(sf) >= 2 else float(sf[0])

    ds = DetDataSample()
    ds.set_metainfo(raw_metas)
    with torch.no_grad():
        preds = model(img, [ds], mode="predict")

    pred = preds[0].pred_instances
    scores = pred.scores.cpu().numpy()
    keep = scores >= score_thr

    boxes = pred.bboxes[keep].cpu().numpy() if keep.any() else np.zeros((0, 4))
    scores_k = scores[keep]
    classes = pred.labels[keep].cpu().numpy() if keep.any() else np.zeros((0,), int)

    masks_bin = None
    if hasattr(pred, "masks") and keep.any():
        try:
            raw = pred.masks[keep]
            masks_bin = raw.masks.cpu().numpy() if hasattr(raw, "masks") else raw.cpu().numpy()
        except Exception:
            pass
    return {"bboxes": boxes, "scores": scores_k, "labels": classes, "masks": masks_bin}


# ──────────────────────────────────────────────────────────────────────────
#  Visualisation + COCO formatting
# ──────────────────────────────────────────────────────────────────────────

def read_image_bgr(path: str) -> np.ndarray:
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        img = tifffile.imread(path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 4:
            img = img[:, :, :3]
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.imread(path)


def draw_predictions(image_bgr: np.ndarray, boxes: np.ndarray,
                     masks: List[np.ndarray], classes: np.ndarray,
                     scores: np.ndarray) -> np.ndarray:
    vis = image_bgr.astype(np.float32)
    for mask, cls in zip(masks, classes):
        color = _COLORS_BGR[min(int(cls), 3)]
        for c in range(3):
            vis[:, :, c] = np.where(mask > 0,
                                    vis[:, :, c] * 0.55 + color[c] * 0.45,
                                    vis[:, :, c])
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        color = _COLORS_BGR[min(int(cls), 3)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{_CLASS_NAMES[min(int(cls), 3)]} {score:.2f}",
                    (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis


def to_coco_record(image_id: int, box, score: float, cls: int,
                   mask: np.ndarray = None) -> Dict:
    bbox = [float(box[0]), float(box[1]),
            float(box[2] - box[0]), float(box[3] - box[1])]
    seg = None
    if mask is not None:
        try:
            rle = encode_mask_rle(mask.astype(np.uint8))
            seg = {"size": list(rle["size"]), "counts": rle["counts"]}
        except Exception:
            pass
    return {
        "image_id": int(image_id),
        "bbox": bbox,
        "score": float(score),
        "category_id": int(cls) + 1,
        "segmentation": seg,
    }


# ──────────────────────────────────────────────────────────────────────────
#  main()
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # build paths
    cfg["paths"] = {**cfg.get("paths", {}), **build_output_paths(cfg)}
    if args.checkpoint:
        cfg.setdefault("paths", {})["checkpoint_path"] = args.checkpoint
    cfg.setdefault("data", {}).update({
        "test_dir": args.test_folder,
        "mapping_json": args.mapping_json,
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_test_model(cfg, device, ckpt_override=args.checkpoint)
    loader = build_test_loader(cfg, args)
    img_size = int(cfg.get("data", {}).get("img_size", 1024))

    # output dirs
    model_name = cfg["model"]["name"]
    backbone = cfg["model"]["backbone"]
    exp_name = cfg.get("experiment", {}).get("name", "v1")
    output_dir = args.output_dir or os.path.join("submissions", model_name, backbone, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    visualize_dir = os.path.join(output_dir, "visualize")
    os.makedirs(visualize_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "test-results.json")
    zip_path = os.path.join(output_dir, f"{os.path.basename(output_dir)}.zip")

    name_to_id, size_dict = load_image_id_mapping(args.mapping_json)

    results = []
    for batch in tqdm(loader, desc="Instance prediction"):
        item = batch[0]
        file_name = item["file_name"]
        if file_name not in name_to_id:
            print(f"Skipping unknown file: {file_name}")
            continue
        H, W = size_dict[file_name]
        image_id = item["image_id"]

        pred = predict_image(model, item, device, args.score_thr)
        boxes, scores, classes, masks_bin = (pred["bboxes"], pred["scores"],
                                             pred["labels"], pred["masks"])

        # ── visualize on original image ──
        scale = img_size / max(H, W)
        vis_boxes = boxes / scale if len(boxes) > 0 else boxes
        vis_masks: List[np.ndarray] = []
        if masks_bin is not None:
            for m in masks_bin:
                vis_masks.append(cv2.resize(m.astype(np.uint8), (W, H),
                                            interpolation=cv2.INTER_NEAREST))
        image_bgr = read_image_bgr(os.path.join(args.test_folder, file_name))
        vis = draw_predictions(image_bgr, vis_boxes, vis_masks, classes, scores)
        cv2.imwrite(os.path.join(visualize_dir, file_name.replace(".tif", ".png")), vis)

        # ── COCO records (masks at model letterbox resolution) ──
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            mask = masks_bin[i] if (masks_bin is not None and i < len(masks_bin)) else None
            results.append(to_coco_record(image_id, box, score, cls, mask))

    results = sorted(results, key=lambda x: x["image_id"])
    with open(json_path, "w") as f:
        json.dump(results, f)
    print(f"\n✅ JSON saved to: {json_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname=os.path.basename(json_path))
    print(f"✅ ZIP saved to: {zip_path}")


if __name__ == "__main__":
    main()
