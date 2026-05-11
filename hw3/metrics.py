"""COCO-style mask AP, PR curves, confusion matrix — pure functions."""
from typing import Dict, List

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
#  IoU primitives
# ──────────────────────────────────────────────────────────────────────────

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / max(float(union), 1e-6)


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def mask_iou_matrix(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> np.ndarray:
    N, M = len(pred_masks), len(gt_masks)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    # Sub-sample large masks to keep matmul tractable
    stride = 4 if pred_masks[0].size > 256 * 256 else 1
    pred_flat = np.stack([m[::stride, ::stride].ravel().astype(np.float32) for m in pred_masks])
    gt_flat = np.stack([m[::stride, ::stride].ravel().astype(np.float32) for m in gt_masks])
    inter = pred_flat @ gt_flat.T
    pred_area = pred_flat.sum(axis=1, keepdims=True)
    gt_area = gt_flat.sum(axis=1, keepdims=True).T
    union = pred_area + gt_area - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
#  AP computation
# ──────────────────────────────────────────────────────────────────────────

def compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    rec_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for thr in rec_thresholds:
        p = precision[recall >= thr]
        ap += p.max() if p.size > 0 else 0.0
    return ap / 101


def _greedy_match_per_class(
    iou_mat: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> tuple:
    """Greedy descending-score matching. Returns (tp_list, fp_list, scores_list)."""
    tp, fp = [], []
    matched_gt = set()
    order = np.argsort(-scores)
    for i in order:
        best_j = int(iou_mat[i].argmax())
        if iou_mat[i, best_j] >= iou_threshold and best_j not in matched_gt:
            tp.append(1); fp.append(0); matched_gt.add(best_j)
        else:
            tp.append(0); fp.append(1)
    return tp, fp, scores[order].tolist()


def _ap_per_class_generic(
    iou_fn, pred_items, pred_scores, pred_labels, gt_items, gt_labels,
    num_classes: int, iou_threshold: float = 0.5,
) -> Dict[int, float]:
    """Shared AP loop. iou_fn(pred_subset, gt_subset) → (n_pred,n_gt) matrix."""
    ap_dict: Dict[int, float] = {}
    for cls in range(num_classes):
        all_scores, all_tp, all_fp = [], [], []
        n_gt = 0
        for p_items, p_sc, p_lb, g_items, g_lb in zip(
            pred_items, pred_scores, pred_labels, gt_items, gt_labels
        ):
            pi = np.where(p_lb == cls)[0]
            gi = np.where(g_lb == cls)[0]
            cp = [p_items[i] for i in pi] if isinstance(p_items, list) else p_items[pi]
            cg = [g_items[i] for i in gi] if isinstance(g_items, list) else g_items[gi]
            cs = p_sc[pi]
            n_gt += len(cg)

            if len(cp) == 0:
                continue
            if len(cg) == 0:
                all_scores.extend(cs.tolist())
                all_tp.extend([0] * len(cs))
                all_fp.extend([1] * len(cs))
                continue

            iou_mat = iou_fn(cp, cg)
            tp, fp, sc = _greedy_match_per_class(iou_mat, cs, iou_threshold)
            all_tp.extend(tp); all_fp.extend(fp); all_scores.extend(sc)

        if n_gt == 0 or not all_scores:
            ap_dict[cls] = 0.0
            continue

        order = np.argsort(-np.array(all_scores))
        tp_cum = np.cumsum(np.array(all_tp)[order])
        fp_cum = np.cumsum(np.array(all_fp)[order])
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        ap_dict[cls] = compute_ap_from_pr(precision, recall)
    return ap_dict


def compute_mask_ap_per_class(
    pred_masks, pred_scores, pred_labels, gt_masks, gt_labels,
    num_classes: int, iou_threshold: float = 0.5,
) -> Dict[int, float]:
    return _ap_per_class_generic(
        mask_iou_matrix, pred_masks, pred_scores, pred_labels, gt_masks, gt_labels,
        num_classes, iou_threshold,
    )


def compute_ap_per_class(
    pred_bboxes, pred_scores, pred_labels, gt_bboxes, gt_labels,
    num_classes: int, iou_threshold: float = 0.5,
) -> Dict[int, float]:
    return _ap_per_class_generic(
        lambda a, b: bbox_iou(np.asarray(a), np.asarray(b)),
        pred_bboxes, pred_scores, pred_labels, gt_bboxes, gt_labels,
        num_classes, iou_threshold,
    )


def compute_map(ap_dict: Dict[int, float]) -> float:
    vals = list(ap_dict.values())
    return float(np.mean(vals)) if vals else 0.0


# ──────────────────────────────────────────────────────────────────────────
#  Per-class PR + F1 curves (used for diagnostic plots)
# ──────────────────────────────────────────────────────────────────────────

def compute_pr_per_class(
    pred_masks, pred_scores, pred_labels, gt_masks, gt_labels,
    num_classes: int, iou_threshold: float = 0.5,
) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for cls in range(num_classes):
        all_scores, all_tp, all_fp = [], [], []
        n_gt = 0
        for p_masks, p_sc, p_lb, g_masks, g_lb in zip(
            pred_masks, pred_scores, pred_labels, gt_masks, gt_labels
        ):
            pi = np.where(p_lb == cls)[0]
            gi = np.where(g_lb == cls)[0]
            cp = [p_masks[i] for i in pi]
            cg = [g_masks[i] for i in gi]
            cs = p_sc[pi]
            n_gt += len(cg)
            if len(cp) == 0:
                continue
            if len(cg) == 0:
                all_scores.extend(cs.tolist())
                all_tp.extend([0] * len(cs))
                all_fp.extend([1] * len(cs))
                continue
            iou_mat = mask_iou_matrix(cp, cg)
            tp, fp, sc = _greedy_match_per_class(iou_mat, cs, iou_threshold)
            all_tp.extend(tp); all_fp.extend(fp); all_scores.extend(sc)

        if n_gt == 0 or not all_scores:
            out[cls] = {"precision": np.zeros(1), "recall": np.zeros(1),
                        "f1": np.zeros(1), "ap": 0.0, "scores": np.zeros(1)}
            continue
        order = np.argsort(-np.array(all_scores))
        tp_cum = np.cumsum(np.array(all_tp)[order])
        fp_cum = np.cumsum(np.array(all_fp)[order])
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        with np.errstate(invalid="ignore"):
            f1 = np.where((precision + recall) > 0,
                          2 * precision * recall / (precision + recall), 0.0)
        out[cls] = {
            "precision": precision, "recall": recall, "f1": f1,
            "ap": compute_ap_from_pr(precision, recall),
            "scores": np.array(all_scores)[order],
        }
    return out


def compute_confusion_matrix(
    pred_masks, pred_scores, pred_labels, gt_masks, gt_labels,
    num_classes: int, iou_threshold: float = 0.5, score_threshold: float = 0.05,
) -> np.ndarray:
    """(num_classes+1) x (num_classes+1). Last row/col = background (FP/FN)."""
    n = num_classes + 1
    cm = np.zeros((n, n), dtype=np.int64)
    bg = num_classes

    for p_masks, p_sc, p_lb, g_masks, g_lb in zip(
        pred_masks, pred_scores, pred_labels, gt_masks, gt_labels
    ):
        keep = np.where(p_sc >= score_threshold)[0]
        cp_masks = [p_masks[i] for i in keep]
        cp_lb = p_lb[keep]
        matched_pred, matched_gt = set(), set()

        if len(cp_masks) > 0 and len(g_masks) > 0:
            iou_mat = mask_iou_matrix(cp_masks, list(g_masks))
            rows, cols = np.where(iou_mat >= iou_threshold)
            taken_r, taken_c = set(), set()
            pairs = sorted(zip(iou_mat[rows, cols], rows, cols), reverse=True)
            for _, r, c in pairs:
                if r not in taken_r and c not in taken_c:
                    cm[int(g_lb[c]), int(cp_lb[r])] += 1
                    taken_r.add(r); taken_c.add(c)
                    matched_pred.add(r); matched_gt.add(c)

        for i in range(len(cp_masks)):
            if i not in matched_pred:
                cm[bg, int(cp_lb[i])] += 1
        for j in range(len(g_masks)):
            if j not in matched_gt:
                cm[int(g_lb[j]), bg] += 1
    return cm


# ──────────────────────────────────────────────────────────────────────────
#  COCO-style evaluation (using pycocotools)
# ──────────────────────────────────────────────────────────────────────────

def coco_evaluate(coco_gt, coco_results: List[dict], iou_type: str = "segm") -> dict:
    from pycocotools.cocoeval import COCOeval
    if not coco_results:
        return {}
    coco_dt = coco_gt.loadRes(coco_results)
    e = COCOeval(coco_gt, coco_dt, iou_type)
    e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats
    return {"mAP": float(s[0]), "mAP_50": float(s[1]), "mAP_75": float(s[2]),
            "mAP_s": float(s[3]), "mAP_m": float(s[4]), "mAP_l": float(s[5]),
            "mask_ap": float(s[0])}
