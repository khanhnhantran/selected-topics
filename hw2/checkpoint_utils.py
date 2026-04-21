import glob
import logging
import os
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir):
    final = os.path.join(output_dir, "model_final.pth")
    if os.path.isfile(final):
        return final

    checkpoints = glob.glob(os.path.join(output_dir, "model_*.pth"))
    if not checkpoints:
        return None

    def _iter_from_name(path):
        match = re.search(r"model_(\d+)\.pth$", path)
        return int(match.group(1)) if match else -1

    latest = max(checkpoints, key=_iter_from_name)
    logger.info("Latest checkpoint found: %s", latest)
    return latest


def load_model_weights(model, checkpoint_path, strict=False):
    from detectron2.checkpoint import DetectionCheckpointer

    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    if strict:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("model", state)
        model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded checkpoint (strict) from '%s'.", checkpoint_path)
    else:
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(checkpoint_path)
        logger.info("Loaded checkpoint (non-strict) from '%s'.", checkpoint_path)

    return model


def save_checkpoint(model, path, iteration, optimizer=None, metric=None, best_metric=None, grad_scaler=None):
    unwrapped = model.module if hasattr(model, "module") else model
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": unwrapped.state_dict(),
        "iteration": iteration,
        "optimizer_type": type(optimizer).__name__ if optimizer is not None else None,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if grad_scaler is not None:
        ckpt["grad_scaler"] = grad_scaler.state_dict()
    if metric is not None:
        ckpt["metric"] = metric
    if best_metric is not None:
        ckpt["best_metric"] = best_metric
    torch.save(ckpt, path)
    logger.info(
        "Saved checkpoint → %s  (iter %d, metric=%s)",
        path,
        iteration,
        f"{metric:.4f}" if metric is not None else "N/A",
    )


def load_custom_checkpoint(trainer_obj, path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = trainer_obj.model
        unwrapped = model.module if hasattr(model, "module") else model

        if not (isinstance(ckpt, dict) and "model" in ckpt):
            unwrapped.load_state_dict(ckpt, strict=False)
            logger.info("Loaded weights from %s (legacy format, iter 0)", path)
            return 0, -1.0

        unwrapped.load_state_dict(ckpt["model"], strict=False)

        if "optimizer" in ckpt and trainer_obj.optimizer is not None:
            trainer_obj.optimizer.load_state_dict(ckpt["optimizer"])

        if "grad_scaler" in ckpt:
            gs = getattr(trainer_obj, "grad_scaler", None)
            if gs is not None:
                gs.load_state_dict(ckpt["grad_scaler"])

        best_metric = float(ckpt.get("best_metric", -1.0) or -1.0)
        saved_iter = int(ckpt.get("iteration", 0))
        trainer_obj.iter = saved_iter

        logger.info(
            "Resumed from %s  →  iter %d, best_metric=%s",
            path,
            saved_iter,
            f"{best_metric:.4f}" if best_metric > 0 else "N/A",
        )
        return saved_iter + 1, best_metric

    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s", path, e)
        return 0, -1.0
