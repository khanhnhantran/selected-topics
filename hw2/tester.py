import logging

import torch
from tqdm import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.layers.nms import batched_nms

logger = logging.getLogger(__name__)


def setup_model(cfg):
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)

    ema_kwargs = {}
    try:
        from detrex.modeling import ema

        ema.may_build_model_ema(cfg, model)
        ema_kwargs = ema.may_get_ema_checkpointer(cfg, model)
    except Exception:
        pass

    checkpointer = DetectionCheckpointer(model, **ema_kwargs)
    checkpointer.load(cfg.train.init_checkpoint)
    logger.info("Loaded checkpoint: %s", cfg.train.init_checkpoint)

    try:
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            from detrex.modeling import ema as _ema

            _ema.apply_model_ema(model)
    except Exception:
        pass

    model.eval()
    return model


def soft_nms(boxes, scores, labels, sigma=0.5, min_score=1e-4):
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    max_coord = boxes.max() + 1.0
    offsets = labels.float() * (max_coord + 1.0)
    shifted = boxes + offsets.unsqueeze(1)

    scores = scores.clone().float()
    x1, y1, x2, y2 = shifted.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    active = torch.ones(len(scores), dtype=torch.bool, device=boxes.device)
    keep = []

    for _ in range(len(scores)):
        tmp = scores.clone()
        tmp[~active] = -1.0
        i = int(tmp.argmax())
        if scores[i] < min_score:
            break
        keep.append(i)
        active[i] = False

        rest = active.nonzero(as_tuple=False).squeeze(1)
        if rest.numel() == 0:
            break

        ix1 = torch.max(shifted[i, 0], shifted[rest, 0])
        iy1 = torch.max(shifted[i, 1], shifted[rest, 1])
        ix2 = torch.min(shifted[i, 2], shifted[rest, 2])
        iy2 = torch.min(shifted[i, 3], shifted[rest, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)

        scores[rest] *= torch.exp(-(iou**2) / sigma)
        active[rest] &= scores[rest] > min_score

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def post_process_detections(boxes, scores, classes, pp):
    nms_type = pp.get("nms_type", "none")
    pp_iou = float(pp.get("nms_iou_threshold", 0.3))
    sigma = float(pp.get("soft_sigma", 0.5))
    min_size = float(pp.get("min_box_size", 0))
    max_det = int(pp.get("max_det", 300))

    if boxes.numel() == 0:
        return boxes, scores, classes

    if min_size > 0:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        mask = torch.max(w, h) >= min_size
        boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

    if boxes.numel() == 0:
        return boxes, scores, classes

    if nms_type == "soft":
        keep = soft_nms(boxes, scores, classes, sigma=sigma)
    elif nms_type == "hard":
        keep = batched_nms(boxes, scores, classes, pp_iou)
    else:
        keep = scores.argsort(descending=True)

    keep = keep[:max_det]
    return boxes[keep], scores[keep], classes[keep]


def run_inference(cfg, score_threshold=0.05, post_process=None):
    model = setup_model(cfg)
    test_loader = instantiate(cfg.dataloader.test)
    model.eval()
    predictions = []
    pp = post_process or {}

    if pp:
        logger.info(
            "Post-processing: nms_type=%s  iou=%.2f  sigma=%.2f  min_box_size=%s  max_det=%s",
            pp.get("nms_type", "none"),
            float(pp.get("nms_iou_threshold", 0.3)),
            float(pp.get("soft_sigma", 0.5)),
            pp.get("min_box_size", 0),
            pp.get("max_det", 300),
        )

    raw_total = 0
    kept_total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference", unit="batch", colour="green"):
            outputs = model(batch)
            for inp, out in zip(batch, outputs):
                instances = out.get("instances")
                if instances is None:
                    continue

                image_id = inp.get("image_id", inp.get("file_name", "unknown"))
                boxes = instances.pred_boxes.tensor.cpu()
                scores = instances.scores.cpu()
                classes = instances.pred_classes.cpu()
                raw_total += len(scores)

                boxes, scores, classes = post_process_detections(boxes, scores, classes, pp)
                kept_total += len(scores)

                for box, score, cls in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
                    if score < score_threshold:
                        continue
                    x1, y1, x2, y2 = box
                    predictions.append(
                        {
                            "image_id": image_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                            "category_id": cls + 1,
                        }
                    )

    logger.info(
        "Post-processing: %d raw → %d kept → %d above score_threshold=%.3f",
        raw_total,
        kept_total,
        len(predictions),
        score_threshold,
    )
    return predictions
