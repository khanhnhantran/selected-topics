"""Training entry point — pure functions, no classes.

Usage:
    python train.py --config config.yaml [--resume CKPT] [--gpu-ids 0]

Each function below has a single responsibility; main() composes them.
"""
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import CellInstanceDataset
from io_utils import (
    build_output_paths, scan_dir, stratified_val_split,
    save_checkpoint as ckpt_save, load_checkpoint as ckpt_load,
)
from models import build_model_cfg, build_mmdet_model
from metrics import (
    compute_mask_ap_per_class, compute_ap_per_class, compute_map,
    compute_pr_per_class, compute_confusion_matrix,
)
from visualization import save_all_charts, denorm_image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger("train")


# ──────────────────────────────────────────────────────────────────────────
#  Setup helpers
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instance Segmentation Training")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--gpu-ids", default=None,
                   help="GPU IDs override, e.g. '3' or '3,6,7'.")
    return p.parse_args()


def setup_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.gpu_ids:
        cfg.setdefault("training", {})["gpu_ids"] = args.gpu_ids
    if args.device:
        cfg.setdefault("training", {})["device"] = args.device
    if args.epochs:
        cfg.setdefault("training", {})["max_epochs"] = args.epochs
    if args.resume:
        cfg.setdefault("training", {})["resume_from"] = args.resume
    return cfg


def select_device(cfg: dict) -> str:
    gpu_ids = str(cfg.get("training", {}).get("gpu_ids", "")).strip()
    if torch.cuda.is_available() and gpu_ids:
        # CUDA_VISIBLE_DEVICES has been set by the launcher, so cuda:0 always works
        return "cuda:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────
#  Optimiser / scheduler
# ──────────────────────────────────────────────────────────────────────────

def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg.get("training", {})
    lr = float(train_cfg.get("lr", 0.02))
    wd = float(train_cfg.get("weight_decay", 0.0001))
    backbone = cfg.get("model", {}).get("backbone", "")
    auto_adamw = backbone in ("swin_b", "swin_l", "convnext_l")
    opt_type = train_cfg.get("optimizer", "auto").lower()
    if opt_type == "auto":
        opt_type = "adamw" if auto_adamw else "sgd"
    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)


def build_scheduler(optimizer: torch.optim.Optimizer, n_iters_per_epoch: int,
                    cfg: dict) -> torch.optim.lr_scheduler._LRScheduler:
    train_cfg = cfg.get("training", {})
    warmup_epochs = int(train_cfg.get("warmup_epochs", 1))
    milestones = train_cfg.get("lr_milestones", [8, 11])
    iter_milestones = [m * max(n_iters_per_epoch, 1) for m in milestones]
    total_warmup = warmup_epochs * max(n_iters_per_epoch, 1)

    def lr_lambda(step):
        if step < total_warmup:
            return step / max(total_warmup, 1)
        factor = 1.0
        for m in iter_milestones:
            if step >= m:
                factor *= 0.1
        return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────
#  Data loaders
# ──────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, List[str]]:
    data_cfg = cfg.get("data", {})
    train_dir = data_cfg.get("train_dir", "data/train")
    aug_cfg = data_cfg.get("augmentation", {})
    batch_size = int(cfg.get("training", {}).get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 4))
    use_oversampling = data_cfg.get("use_oversampling", True)
    val_ratio = float(data_cfg.get("val_ratio", 0.2))

    all_ids = scan_dir(train_dir)
    train_ids, val_ids = stratified_val_split(all_ids, train_dir, val_ratio=val_ratio)
    logger.info(f"Train: {len(train_ids)} | Val: {len(val_ids)} (stratified {val_ratio:.0%})")

    train_ds = CellInstanceDataset(train_dir, train_ids, augmentation_cfg=aug_cfg, mode="train")
    val_ds = CellInstanceDataset(train_dir, val_ids, augmentation_cfg=aug_cfg, mode="val")

    sampler = None
    if use_oversampling:
        weights = train_ds.get_oversampling_weights()
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(train_ds),
            replacement=True,
        )

    pin = torch.cuda.is_available()
    collate = lambda b: b
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, collate_fn=collate, pin_memory=pin,
        persistent_workers=(num_workers > 0), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=collate, pin_memory=pin, persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, val_ids


# ──────────────────────────────────────────────────────────────────────────
#  Forward helpers (train + validate)
# ──────────────────────────────────────────────────────────────────────────

def fix_img_metas(img_metas: dict, h: int, w: int) -> dict:
    out = dict(img_metas)
    if "scale_factor" in out:
        sf = out["scale_factor"]
        if isinstance(sf, torch.Tensor):
            sf = sf.cpu().numpy()
        sf = np.array(sf).reshape(-1)
        if len(sf) == 4:
            out["scale_factor"] = sf[:2].tolist()
        elif len(sf) == 2:
            out["scale_factor"] = sf.tolist()
    out.setdefault("batch_input_shape", (h, w))
    out.setdefault("img_shape", (h, w))
    out.setdefault("ori_shape", (h, w))
    return out


def forward_train_batch(model, batch: List[Dict], device: str, model_name: str,
                        use_fp16: bool) -> Optional[Dict[str, torch.Tensor]]:
    if not batch:
        return None
    try:
        from mmdet.structures import DetDataSample
        from mmdet.structures.mask import BitmapMasks
        from mmengine.structures import InstanceData, PixelData

        inputs = []
        data_samples = []
        for item in batch:
            img = item["img"].to(device)
            h, w = img.shape[-2:]
            inputs.append(img)

            gt_inst = InstanceData()
            gt_inst.bboxes = item["gt_bboxes"].to(device)
            gt_inst.labels = item["gt_labels"].to(device)

            raw_masks = item.get("gt_masks", [])
            if raw_masks:
                arr = np.stack([np.asarray(m, dtype=np.uint8) for m in raw_masks], axis=0)
                gt_inst.masks = BitmapMasks(arr, height=h, width=w)
            else:
                gt_inst.masks = BitmapMasks(np.zeros((0, h, w), dtype=np.uint8), height=h, width=w)

            ds = DetDataSample()
            ds.gt_instances = gt_inst
            ds.set_metainfo(fix_img_metas(item["img_metas"], h, w))

            if model_name == "htc":
                # HTC FusedSemanticHead requires a class-id semantic map (0=bg)
                sem = np.zeros((h, w), dtype=np.int64)
                labels_np = item["gt_labels"].numpy()
                for m, l in zip(raw_masks, labels_np):
                    sem[np.asarray(m, dtype=bool)] = int(l) + 1
                gt_sem = PixelData()
                gt_sem.sem_seg = torch.from_numpy(sem).unsqueeze(0).to(device)
                ds.gt_sem_seg = gt_sem
            data_samples.append(ds)

        img = torch.stack(inputs, dim=0)
        with torch.cuda.amp.autocast(enabled=use_fp16):
            losses = model(img, data_samples, mode="loss")

        result: Dict[str, torch.Tensor] = {}
        for k, v in losses.items():
            if "loss" not in k:
                continue
            if isinstance(v, torch.Tensor):
                result[k] = v
            elif isinstance(v, (list, tuple)):
                valid = [t for t in v if isinstance(t, torch.Tensor)]
                if valid:
                    result[k] = sum(valid)
        return result if result else None
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning("Train batch OOM — skipping")
        return None
    except Exception as e:
        logger.warning(f"Train forward error ({type(e).__name__}): {e}")
        return None


def forward_validate_item(model, item: Dict, device: str, score_thr: float) -> Dict:
    try:
        from mmdet.structures import DetDataSample
        img = item["img"].unsqueeze(0).to(device)
        h, w = img.shape[-2:]
        ds = DetDataSample()
        ds.set_metainfo(fix_img_metas(item["img_metas"], h, w))

        results = model(img, [ds], mode="predict")
        pred = results[0].pred_instances
        keep = (pred.scores >= score_thr).nonzero(as_tuple=False).squeeze(1).cpu().numpy()
        bboxes = pred.bboxes[keep].cpu().tolist()
        scores = pred.scores[keep].cpu().numpy()
        labels = pred.labels[keep].cpu().numpy()

        masks: List[np.ndarray] = []
        if hasattr(pred, "masks") and len(keep) > 0:
            raw = pred.masks[keep]
            mask_arr = raw.masks if hasattr(raw, "masks") else np.asarray(raw.cpu())
            masks = [mask_arr[i].astype(np.uint8) for i in range(len(mask_arr))]

        return {"bboxes": bboxes, "labels": labels, "scores": scores, "masks": masks}
    except Exception as e:
        logger.exception(f"Val forward error: {e}")
        return {"bboxes": [], "labels": np.zeros(0, int), "scores": np.zeros(0), "masks": []}


# ──────────────────────────────────────────────────────────────────────────
#  Training / validation loops
# ──────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    device: str, model_name: str, grad_clip: float,
                    use_fp16: bool, epoch: int, max_epochs: int) -> Dict[str, float]:
    model.train()
    total_loss, n_batches = 0.0, 0
    running: Dict[str, float] = {}

    pbar = tqdm(loader, desc=f"Train E{epoch + 1}/{max_epochs}", unit="batch",
                dynamic_ncols=True, leave=False, colour="green")
    for batch in pbar:
        losses = forward_train_batch(model, batch, device, model_name, use_fp16)
        if losses is None:
            optimizer.zero_grad()
            continue

        loss = sum(losses.values())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()

        pbar.set_postfix(total=f"{loss.item():.3f}", avg=f"{total_loss/n_batches:.3f}")

    avg = {k: v / max(n_batches, 1) for k, v in running.items()}
    avg["loss"] = total_loss / max(n_batches, 1)
    return avg


def validate_one_epoch(model, loader, device: str, num_classes: int,
                       score_thr: float, epoch: int) -> Dict[str, Any]:
    model.eval()
    all_preds, all_gt, vis_raw = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Val E{epoch + 1}", unit="batch",
                          dynamic_ncols=True, leave=False):
            for item in batch:
                preds = forward_validate_item(model, item, device, score_thr)
                all_preds.append(preds)
                all_gt.append({
                    "gt_bboxes": item["gt_bboxes"].numpy(),
                    "gt_labels": item["gt_labels"].numpy(),
                    "gt_masks": item.get("gt_masks", []),
                })
                if len(vis_raw) < 6:
                    vis_raw.append({
                        "img_chw": item["img"].numpy(),
                        "gt_masks": item.get("gt_masks", []),
                        "gt_labels": item["gt_labels"].numpy().tolist(),
                        "pred_masks": preds["masks"],
                        "pred_labels": list(np.asarray(preds["labels"], int)),
                        "pred_scores": list(np.asarray(preds["scores"], float)),
                    })

    metrics = compute_val_metrics(all_preds, all_gt, num_classes)
    metrics["vis_raw"] = vis_raw
    return metrics


def compute_val_metrics(all_preds: List[Dict], all_gt: List[Dict],
                        num_classes: int, iou_threshold: float = 0.5) -> Dict[str, Any]:
    pred_scores = [np.asarray(p["scores"]) for p in all_preds]
    pred_labels = [np.asarray(p["labels"], dtype=int) for p in all_preds]
    gt_labels = [np.asarray(g["gt_labels"], dtype=int) for g in all_gt]

    has_masks = all(len(p["masks"]) == len(p["bboxes"]) for p in all_preds)
    if has_masks:
        pred_masks = [p["masks"] for p in all_preds]
        gt_masks = [g["gt_masks"] for g in all_gt]
        ap_dict = compute_mask_ap_per_class(
            pred_masks, pred_scores, pred_labels, gt_masks, gt_labels,
            num_classes=num_classes, iou_threshold=iou_threshold)
        pr_data = compute_pr_per_class(
            pred_masks, pred_scores, pred_labels, gt_masks, gt_labels,
            num_classes=num_classes, iou_threshold=iou_threshold)
        cm = compute_confusion_matrix(
            pred_masks, pred_scores, pred_labels, gt_masks, gt_labels,
            num_classes=num_classes, iou_threshold=iou_threshold)
    else:
        logger.warning("Masks missing in some predictions; falling back to box AP")
        pred_bb = [np.array(p["bboxes"]).reshape(-1, 4) if p["bboxes"] else np.zeros((0, 4))
                   for p in all_preds]
        gt_bb = [g["gt_bboxes"] for g in all_gt]
        ap_dict = compute_ap_per_class(pred_bb, pred_scores, pred_labels,
                                       gt_bb, gt_labels, num_classes=num_classes,
                                       iou_threshold=iou_threshold)
        pr_data = {}
        cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    return {
        "mask_ap": compute_map(ap_dict),
        "ap_per_class": ap_dict,
        "pr_data": pr_data,
        "confusion_matrix": cm,
    }


# ──────────────────────────────────────────────────────────────────────────
#  Checkpoint / charts
# ──────────────────────────────────────────────────────────────────────────

def save_state(path: str, model, optimizer, epoch: int, best_metric: float,
               is_best: bool, cfg: dict) -> None:
    ckpt_save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric,
        "is_best": is_best,
        "cfg": cfg,
    }, path)


def maybe_resume(model, optimizer, cfg: dict, device: str) -> Tuple[int, float]:
    """Returns (start_epoch, best_metric)."""
    resume_path = cfg.get("training", {}).get("resume_from")
    pretrain = cfg.get("model", {}).get("pretrain_weights")
    if isinstance(pretrain, dict):
        pretrain = pretrain.get("path")

    if pretrain and os.path.exists(pretrain):
        ck = ckpt_load(pretrain, map_location=device)
        state = ck.get("state_dict", ck.get("model", ck))
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained weights from {pretrain}")

    if resume_path and os.path.exists(resume_path):
        ck = ckpt_load(resume_path, map_location=device)
        model.load_state_dict(ck.get("state_dict", ck))
        if "optimizer" in ck and optimizer:
            optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0) + 1
        best_metric = ck.get("best_metric", 0.0)
        logger.info(f"Resumed from {resume_path} (next epoch={start_epoch}, best={best_metric:.4f})")
        return start_epoch, best_metric
    return 0, 0.0


def save_charts(chart_dir: str, train_losses, val_map_history, loss_history,
                val_metrics: Dict[str, Any], cfg: dict, epoch: int) -> None:
    metrics_data: dict = {
        "epoch": epoch + 1,
        "train_losses": train_losses,
        "val_map_history": val_map_history,
        "loss_history": loss_history,
        "ap_per_class": val_metrics.get("ap_per_class", {}),
    }
    if "pr_data" in val_metrics and val_metrics["pr_data"]:
        metrics_data["pr_data"] = val_metrics["pr_data"]
    if "confusion_matrix" in val_metrics:
        metrics_data["confusion_matrix"] = val_metrics["confusion_matrix"]
    if "vis_raw" in val_metrics and val_metrics["vis_raw"]:
        items = val_metrics["vis_raw"]
        metrics_data["vis_data"] = {
            "images": [denorm_image(v["img_chw"]) for v in items],
            "gt_masks": [v["gt_masks"] for v in items],
            "gt_labels": [v["gt_labels"] for v in items],
            "pred_masks": [v["pred_masks"] for v in items],
            "pred_labels": [v["pred_labels"] for v in items],
            "pred_scores": [v["pred_scores"] for v in items],
        }
    save_all_charts(chart_dir, metrics_data, cfg=cfg)


# ──────────────────────────────────────────────────────────────────────────
#  main()
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_cli_overrides(cfg, args)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cfg.get("training", {}).get("gpu_ids"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["training"]["gpu_ids"])

    cfg["paths"] = {**cfg.get("paths", {}), **build_output_paths(cfg)}

    model_name = cfg["model"]["name"].lower()
    backbone = cfg["model"]["backbone"]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg["paths"].get("base_dir", "."),
                            "logs", model_name, backbone, f"{run_id}.log")
    setup_logging(log_file)
    logger.info(f"Config: {args.config}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Ckpt dir: {cfg['paths']['checkpoint_dir']}")
    logger.info(f"Chart dir: {cfg['paths']['chart_dir']}")

    device = select_device(cfg)
    logger.info(f"Using device: {device}")

    # ---- model + data ----
    model_cfg = build_model_cfg(cfg["model"])
    model = build_mmdet_model(model_cfg, device)
    train_loader, val_loader, _val_ids = build_dataloaders(cfg)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, len(train_loader), cfg)

    use_fp16 = bool(cfg.get("training", {}).get("fp16", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    grad_clip = float(cfg.get("training", {}).get("grad_clip", 35.0))
    score_thr = float(cfg.get("inference", {}).get("score_thr", 0.05))
    num_classes = int(cfg["model"].get("num_classes", 4))
    max_epochs = int(cfg["training"]["max_epochs"])

    # ---- resume / pretrain ----
    start_epoch, best_metric = maybe_resume(model, optimizer, cfg, device)

    # ---- main loop ----
    train_loss_history: List[float] = []
    val_map_history: List[float] = []
    loss_history: List[Dict[str, float]] = []
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])

    for epoch in range(start_epoch, max_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, model_name, grad_clip, use_fp16, epoch, max_epochs,
        )
        train_loss_history.append(train_metrics.get("loss", 0.0))
        loss_history.append(train_metrics)

        val_metrics = validate_one_epoch(
            model, val_loader, device, num_classes, score_thr, epoch)
        mask_ap = val_metrics.get("mask_ap", 0.0)
        val_map_history.append(mask_ap)
        logger.info(f"Epoch [{epoch + 1}/{max_epochs}] "
                    f"loss={train_metrics.get('loss', 0):.4f} mAP={mask_ap:.4f}")

        save_state(str(ckpt_dir / "last.pth"), model, optimizer, epoch,
                   best_metric, False, cfg)
        if mask_ap > best_metric:
            best_metric = mask_ap
            save_state(str(ckpt_dir / "best.pth"), model, optimizer, epoch,
                       best_metric, True, cfg)
            logger.info(f"  New best mAP={mask_ap:.4f} -> best.pth")

        save_charts(cfg["paths"]["chart_dir"], train_loss_history,
                    val_map_history, loss_history, val_metrics, cfg, epoch)

    logger.info(f"Training complete. Best val mAP = {best_metric:.4f}")


if __name__ == "__main__":
    main()
