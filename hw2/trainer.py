import logging
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
    category=UserWarning,
)
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.engine import SimpleTrainer, hooks
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventWriter,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.file_io import PathManager

from checkpoint_utils import load_custom_checkpoint, save_checkpoint
from logger_utils import (
    log_config,
    log_eval_metrics,
    log_model,
    log_train_metrics,
    resolve_event_csv_path,
)
from visualize import (
    metrics_from_csv,
    plot_confusion_matrix,
    plot_pf1_curve,
    plot_pr_curve,
    plot_rf1_curve,
    plot_training_curves,
    visualize_batch_predictions,
)

logger = logging.getLogger(__name__)


def derive_ckpt_dir(output_dir):
    parts = os.path.normpath(output_dir).split(os.sep)
    return os.path.join("checkpoints", parts[-3], parts[-2], parts[-1])


def derive_chart_dir(output_dir):
    parts = os.path.normpath(output_dir).split(os.sep)
    arch = parts[-3] if len(parts) >= 3 else "model"
    backbone = parts[-2] if len(parts) >= 2 else "backbone"
    run = parts[-1] if len(parts) >= 1 else "0"
    return os.path.join("charts", arch, backbone, run)


def _clip_grads(params, clip_grad_params):
    params = [p for p in params if p.requires_grad and p.grad is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, **clip_grad_params)


def get_class_names_from_cfg(cfg):
    try:
        from detectron2.data import MetadataCatalog

        meta = MetadataCatalog.get(cfg.dataloader.train.dataset.names)
        return list(meta.thing_classes)
    except Exception:
        return None


def save_training_curves(csv_path, chart_dir, run_tag, iters_per_epoch, trainer_iter):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = metrics_from_csv(csv_path, x_col="epoch" if iters_per_epoch else "iter")
    if not metrics:
        return
    title = f"Training Curves  |  {run_tag}  |  iter {trainer_iter:,}"
    out = os.path.join(chart_dir, "training_curves.png")
    fig = plot_training_curves(metrics, suptitle=title, show=False, save_path=out)
    plt.close(fig)
    logger.info("Saved training curves → %s", out)


def save_pr_curves(coco_eval_obj, chart_dir, run_tag, class_names, trainer_iter):
    import matplotlib.pyplot as plt

    prec_arr = coco_eval_obj.eval["precision"]
    n_classes = prec_arr.shape[2]
    recall_pts = np.linspace(0.0, 1.0, prec_arr.shape[1])
    names = class_names[:n_classes] if class_names else [str(i) for i in range(n_classes)]

    precisions, recalls, f1_scores, ap_scores = {}, {}, {}, {}
    for k, name in enumerate(names):
        p = prec_arr[0, :, k, 0, 2]
        valid = p >= 0
        pv, rv = p[valid], recall_pts[valid]
        ap = float(pv.mean()) if pv.size else 0.0
        f1_curve = 2 * pv * rv / (pv + rv + 1e-8)
        precisions[name] = pv
        recalls[name] = rv
        f1_scores[name] = f1_curve
        ap_scores[name] = ap

    tag = f"{run_tag}  |  iter {trainer_iter:,}"
    for fn, x_data, y_data, base_title, fname in [
        (plot_pr_curve, recalls, precisions, "Precision-Recall Curve", "pr_curve.png"),
        (plot_pf1_curve, precisions, f1_scores, "Precision-F1 Curve", "pf1_curve.png"),
        (plot_rf1_curve, recalls, f1_scores, "Recall-F1 Curve", "rf1_curve.png"),
    ]:
        out = os.path.join(chart_dir, fname)
        fig = fn(x_data, y_data, ap_scores, title=f"{base_title}  |  {tag}", show=False, save_path=out)
        plt.close(fig)
    logger.info("Saved PR/F1 curves → %s", chart_dir)


def _iou_xyxy(b1, b2):
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def save_confusion_matrix_chart(evaluator, chart_dir, run_tag, class_names, trainer_iter):
    import matplotlib.pyplot as plt

    try:
        coco_api = evaluator._coco_api
        predictions = evaluator._predictions
    except AttributeError:
        return

    n_cls = len(class_names) if class_names else len(coco_api.cats)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    pred_by_img = {}
    for entry in predictions:
        img_id = entry["image_id"]
        pred_by_img.setdefault(img_id, []).extend(entry.get("instances", []))

    for img_id, gt_anns in coco_api.imgToAnns.items():
        preds = [p for p in pred_by_img.get(img_id, []) if p.get("score", 1.0) >= 0.3]
        for gt in gt_anns:
            gx, gy, gw, gh = gt["bbox"]
            gt_box = [gx, gy, gx + gw, gy + gh]
            gt_cls = gt["category_id"] - 1
            best_iou, best_cls = 0.0, gt_cls
            for p in preds:
                px, py, pw, ph = p["bbox"]
                iou = _iou_xyxy(gt_box, [px, py, px + pw, py + ph])
                if iou > best_iou:
                    best_iou, best_cls = iou, p["category_id"] - 1
            if best_iou >= 0.5 and 0 <= gt_cls < n_cls and 0 <= best_cls < n_cls:
                cm[gt_cls, best_cls] += 1

    names = class_names[:n_cls] if class_names else [str(i) for i in range(n_cls)]
    cm_title = f"Confusion Matrix  |  {run_tag}  |  iter {trainer_iter:,}"
    out = os.path.join(chart_dir, "confusion_matrix.png")
    fig = plot_confusion_matrix(cm, names, normalize=True, title=cm_title, show=False, save_path=out)
    plt.close(fig)
    logger.info("Saved confusion_matrix.png → %s", out)


def sample_val_batch(val_loader_cfg, n=6):
    if val_loader_cfg is None:
        return []
    try:
        from omegaconf import OmegaConf

        loader_cfg = OmegaConf.merge(val_loader_cfg, {"num_workers": 0})
        val_loader = instantiate(loader_cfg)
        pool = []
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                pool.extend(batch)
            else:
                pool.append(batch)
            if len(pool) >= n * 4:
                break
        if not pool:
            return []
        import random

        return random.sample(pool, min(n, len(pool)))
    except Exception as e:
        logger.warning("Could not sample val batch: %s", e)
        return []


def save_val_predictions(
    model, val_loader_cfg, chart_dir, run_tag, class_names, trainer_iter,
    val_accuracy=None, score_threshold=0.4,
):
    if val_loader_cfg is None:
        return
    try:
        import glob as _glob
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        batch = sample_val_batch(val_loader_cfg, n=6)
        if not batch:
            return

        raw_by_id = {}
        try:
            from detectron2.data import DatasetCatalog

            for d in DatasetCatalog.get(val_loader_cfg.dataset.names):
                raw_by_id[d["image_id"]] = d
        except Exception:
            pass

        was_training = model.training
        model.eval()
        images, gt_boxes_l, gt_labels_l = [], [], []
        pred_boxes_l, pred_labels_l, pred_scores_l, titles = [], [], [], []

        with torch.no_grad():
            for sample in batch:
                img_t = sample["image"]
                img_h, img_w = img_t.shape[-2], img_t.shape[-1]
                img_np = img_t.permute(1, 2, 0).cpu().numpy()
                img_np = (
                    img_np.astype("uint8") if img_np.max() > 1.5 else (img_np * 255).astype("uint8")
                )

                orig_h = sample.get("height", img_h)
                orig_w = sample.get("width", img_w)
                scale_h, scale_w = img_h / orig_h, img_w / orig_w
                image_id = sample.get("image_id")
                raw_annots = raw_by_id.get(image_id, {}).get("annotations", []) if image_id else []

                if raw_annots:
                    gt_boxes_l.append(
                        [
                            [
                                a["bbox"][0] * scale_w,
                                a["bbox"][1] * scale_h,
                                (a["bbox"][0] + a["bbox"][2]) * scale_w,
                                (a["bbox"][1] + a["bbox"][3]) * scale_h,
                            ]
                            for a in raw_annots
                        ]
                    )
                    gt_labels_l.append([a["category_id"] for a in raw_annots])
                else:
                    gt_boxes_l.append(None)
                    gt_labels_l.append(None)

                infer_sample = {**sample, "height": img_h, "width": img_w}
                out_inst = model([infer_sample])[0]["instances"]
                pred_boxes_l.append(
                    out_inst.pred_boxes.tensor.cpu().tolist() if out_inst.has("pred_boxes") else None
                )
                pred_labels_l.append(
                    out_inst.pred_classes.cpu().tolist() if out_inst.has("pred_classes") else None
                )
                pred_scores_l.append(
                    out_inst.scores.cpu().tolist() if out_inst.has("scores") else None
                )
                images.append(img_np)
                titles.append(os.path.basename(sample.get("file_name", "")))

        if was_training:
            model.train()

        for old in _glob.glob(os.path.join(chart_dir, "val_predictions_*.png")):
            try:
                os.remove(old)
            except OSError:
                pass

        acc_tag = f"_{val_accuracy:.4f}" if val_accuracy is not None else ""
        out_path = os.path.join(chart_dir, f"val_predictions{acc_tag}.png")
        acc_s = f"  |  mAP {val_accuracy:.4f}" if val_accuracy is not None else ""
        sup = (
            f"Validation – GT (green) vs Predictions (red)  |  "
            f"{run_tag}  |  iter {trainer_iter:,}{acc_s}"
        )

        fig = visualize_batch_predictions(
            images=images,
            gt_boxes_list=gt_boxes_l,
            gt_labels_list=gt_labels_l,
            pred_boxes_list=pred_boxes_l,
            pred_labels_list=pred_labels_l,
            pred_scores_list=pred_scores_l,
            class_names=class_names,
            score_threshold=score_threshold,
            titles=titles,
            ncols=3,
            figsize_per_cell=(5.5, 4.5),
            suptitle=sup,
            show=False,
            save_path=out_path,
        )
        plt.close(fig)
        logger.info("Saved val predictions → %s", out_path)
    except Exception as e:
        logger.warning("save_val_predictions failed: %s", e)


def unwrap_coco_evaluator(evaluator):
    try:
        from detectron2.evaluation import DatasetEvaluators

        if isinstance(evaluator, DatasetEvaluators):
            for ev in evaluator._evaluators:
                if hasattr(ev, "_coco_eval"):
                    return ev
    except Exception:
        pass
    return evaluator


def patch_coco_evaluator(evaluator):
    try:
        from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

        inner = evaluator
        if isinstance(evaluator, DatasetEvaluators):
            for ev in evaluator._evaluators:
                if isinstance(ev, COCOEvaluator):
                    inner = ev
                    break

        if not isinstance(inner, COCOEvaluator):
            return

        orig_derive = inner._derive_coco_results.__func__

        def _patched(self_ev, coco_eval, iou_type, class_names=None):
            if coco_eval is not None:
                if not hasattr(self_ev, "_coco_eval"):
                    self_ev._coco_eval = {}
                self_ev._coco_eval[iou_type] = coco_eval
            return orig_derive(self_ev, coco_eval, iou_type, class_names=class_names)

        import types

        inner._derive_coco_results = types.MethodType(_patched, inner)
    except Exception as e:
        logger.warning("patch_coco_evaluator failed: %s", e)


def run_eval(cfg, trainer_obj, csv_path, ckpt_dir, chart_dir, class_names, iters_per_epoch, vis_state):
    model = trainer_obj.model
    logger.info("Running evaluation …")
    evaluator = instantiate(cfg.dataloader.evaluator)
    patch_coco_evaluator(evaluator)
    ret = inference_on_dataset(model, instantiate(cfg.dataloader.test), evaluator)
    print_csv_format(ret)

    if comm.is_main_process():
        if csv_path:
            try:
                log_eval_metrics(csv_path, trainer_obj.iter, ret, iters_per_epoch)
            except Exception as e:
                logger.warning("log_eval_metrics failed: %s", e)

        metric = None
        for key in ("bbox/AP", "segm/AP"):
            try:
                metric = float(ret["bbox"]["AP"] if key == "bbox/AP" else ret["segm"]["AP"])
                break
            except (KeyError, TypeError):
                pass
        if metric is None:
            try:
                task = next(iter(ret))
                metric = float(next(iter(ret[task].values())))
            except Exception:
                pass

        run_tag = vis_state.get("run_tag", "")

        ev = unwrap_coco_evaluator(evaluator)
        coco_eval_obj = None
        if hasattr(ev, "_coco_eval"):
            coco_eval_obj = ev._coco_eval.get("bbox", None)

        if coco_eval_obj and hasattr(coco_eval_obj, "eval") and coco_eval_obj.eval:
            try:
                save_pr_curves(coco_eval_obj, chart_dir, run_tag, class_names, trainer_obj.iter)
            except Exception as e:
                logger.warning("PR curves failed: %s", e)
            try:
                save_confusion_matrix_chart(ev, chart_dir, run_tag, class_names, trainer_obj.iter)
            except Exception as e:
                logger.warning("Confusion matrix failed: %s", e)

        save_val_predictions(
            model=model,
            val_loader_cfg=cfg.dataloader.test,
            chart_dir=chart_dir,
            run_tag=run_tag,
            class_names=class_names,
            trainer_iter=trainer_obj.iter,
            val_accuracy=metric,
        )

        if csv_path and os.path.exists(csv_path):
            try:
                save_training_curves(csv_path, chart_dir, run_tag, iters_per_epoch, trainer_obj.iter)
            except Exception as e:
                logger.warning("Training curves failed: %s", e)

        if ckpt_dir and metric is not None:
            best_metric = vis_state.get("best_metric", -1.0)
            optimizer = trainer_obj.optimizer
            grad_scaler = getattr(trainer_obj, "grad_scaler", None)
            save_checkpoint(
                model,
                os.path.join(ckpt_dir, "last_model.pth"),
                iteration=trainer_obj.iter,
                optimizer=optimizer,
                metric=metric,
                best_metric=best_metric if best_metric > 0 else None,
                grad_scaler=grad_scaler,
            )
            if metric > best_metric:
                vis_state["best_metric"] = metric
                save_checkpoint(
                    model,
                    os.path.join(ckpt_dir, "best_model.pth"),
                    iteration=trainer_obj.iter,
                    optimizer=optimizer,
                    metric=metric,
                    best_metric=metric,
                    grad_scaler=grad_scaler,
                )
                logger.info("New best metric %.4f → saved best_model.pth", metric)

        vis_state["last_metric"] = metric

    return ret


# Thin wrapper required by Detectron2's SimpleTrainer interface
class _AMPTrainer(SimpleTrainer):
    def __init__(self, model, dataloader, optimizer, amp=False, clip_grad_params=None):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)
        assert not isinstance(model, DataParallel)
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1)
        self.amp = amp
        self.clip_grad_params = clip_grad_params
        self.grad_scaler = torch.amp.GradScaler("cuda") if amp else None

    def run_step(self):
        assert self.model.training
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with torch.amp.autocast("cuda", enabled=self.amp):
            loss_dict = self.model(data)
            losses = loss_dict if isinstance(loss_dict, torch.Tensor) else sum(loss_dict.values())
            if isinstance(loss_dict, torch.Tensor):
                loss_dict = {"total_loss": loss_dict}

        self.optimizer.zero_grad()
        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params:
                self.grad_scaler.unscale_(self.optimizer)
                _clip_grads(self.model.parameters(), self.clip_grad_params)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params:
                _clip_grads(self.model.parameters(), self.clip_grad_params)
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


class _CSVWriter(EventWriter):
    def __init__(self, csv_path, iters_per_epoch):
        self.csv_path = csv_path
        self.iters_per_epoch = iters_per_epoch

    def write(self):
        log_train_metrics(self.csv_path, self.iters_per_epoch)

    def close(self):
        pass


class _VizHook(hooks.HookBase):
    """Periodic training-curve saves; all logic delegated to module functions."""

    def __init__(self, csv_path, chart_dir, run_tag, vis_period, iters_per_epoch):
        self._csv_path = csv_path
        self._chart_dir = chart_dir
        self._run_tag = run_tag
        self._vis_period = vis_period
        self._iters_per_epoch = iters_per_epoch

    def after_step(self):
        cur = self.trainer.iter + 1
        if cur % self._vis_period == 0 and os.path.exists(self._csv_path):
            try:
                save_training_curves(
                    self._csv_path, self._chart_dir, self._run_tag,
                    self._iters_per_epoch, cur,
                )
            except Exception as e:
                logger.warning("VizHook curves failed: %s", e)

    def after_train(self):
        if os.path.exists(self._csv_path):
            try:
                save_training_curves(
                    self._csv_path, self._chart_dir, self._run_tag,
                    self._iters_per_epoch, self.trainer.iter + 1,
                )
            except Exception as e:
                logger.warning("VizHook final curves failed: %s", e)


def setup_trainer(cfg):
    if comm.is_main_process():
        log_config(cfg, logger)

    model = instantiate(cfg.model)
    if comm.is_main_process():
        log_model(model, logger)
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    train_loader = instantiate(cfg.dataloader.train)
    model = create_ddp_model(model, **cfg.train.ddp)

    ema_checkpointer_kwargs = {}
    try:
        from detrex.modeling import ema

        ema.may_build_model_ema(cfg, model)
        ema_checkpointer_kwargs = ema.may_get_ema_checkpointer(cfg, model)
    except Exception:
        pass

    clip_grad = cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None
    trainer_obj = _AMPTrainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=clip_grad,
    )

    output_dir = cfg.train.output_dir
    PathManager.mkdirs(output_dir)

    checkpointer = DetectionCheckpointer(
        model, output_dir, trainer=trainer_obj, **ema_checkpointer_kwargs
    )

    iters_per_epoch = None
    try:
        from detectron2.data import DatasetCatalog

        total_images = len(DatasetCatalog.get(cfg.dataloader.train.dataset.names))
        bs = cfg.dataloader.train.total_batch_size
        iters_per_epoch = max(1, total_images // bs)
    except Exception:
        pass

    csv_path = None
    ckpt_dir = None
    chart_dir = None
    run_tag = ""

    if comm.is_main_process():
        csv_path = resolve_event_csv_path(output_dir)
        ckpt_dir = derive_ckpt_dir(output_dir)
        chart_dir = derive_chart_dir(output_dir)
        PathManager.mkdirs(ckpt_dir)
        PathManager.mkdirs(chart_dir)

        parts = os.path.normpath(chart_dir).split(os.sep)
        run = parts[-1] if len(parts) >= 1 else "0"
        bbone = parts[-2] if len(parts) >= 2 else "backbone"
        arch = parts[-3] if len(parts) >= 3 else "model"
        run_tag = f"{arch}  /  {bbone}  /  run {run}"

        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
            _CSVWriter(csv_path, iters_per_epoch),
        ]
    else:
        writers = []

    ema_hook = None
    try:
        from detrex.modeling import ema

        ema_hook = ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None
    except Exception:
        pass

    vis_hook = None
    if comm.is_main_process() and chart_dir is not None:
        vis_period = getattr(cfg.train, "vis_period", cfg.train.eval_period)
        vis_hook = _VizHook(csv_path, chart_dir, run_tag, vis_period, iters_per_epoch)

    vis_state = {"run_tag": run_tag, "best_metric": -1.0, "last_metric": None}
    class_names = get_class_names_from_cfg(cfg)

    trainer_obj.register_hooks(
        [
            hooks.IterationTimer(),
            ema_hook,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.EvalHook(
                cfg.train.eval_period,
                lambda: run_eval(
                    cfg, trainer_obj, csv_path, ckpt_dir, chart_dir,
                    class_names, iters_per_epoch, vis_state,
                ),
            ),
            vis_hook,
            (
                hooks.PeriodicWriter(writers, period=cfg.train.log_period)
                if comm.is_main_process()
                else None
            ),
        ]
    )

    logger.info("Trainer setup complete. Output dir: %s", output_dir)
    return trainer_obj, checkpointer, {
        "csv_path": csv_path,
        "ckpt_dir": ckpt_dir,
        "chart_dir": chart_dir,
        "iters_per_epoch": iters_per_epoch,
        "vis_state": vis_state,
    }


def train_model(cfg, resume=False):
    trainer_obj, checkpointer, state = setup_trainer(cfg)

    start_iter = 0
    ckpt_dir = derive_ckpt_dir(cfg.train.output_dir)
    last_pth = os.path.join(ckpt_dir, "last_model.pth")

    if os.path.isfile(last_pth):
        logger.info("Found last_model.pth → loading from %s", last_pth)
        start_iter, best = load_custom_checkpoint(trainer_obj, last_pth)
        state["vis_state"]["best_metric"] = best
    elif cfg.train.init_checkpoint:
        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=False)
        logger.info("Loaded init_checkpoint as weights only. Starting from iter 0.")

    logger.info("Starting training from iter %d to %d.", start_iter, cfg.train.max_iter)
    trainer_obj.train(start_iter, cfg.train.max_iter)
