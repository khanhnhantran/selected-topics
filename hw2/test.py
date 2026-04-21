import argparse
import json
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config_utils import setup_sys_path

setup_sys_path(_PROJECT_ROOT)

from detectron2.config import LazyConfig
from detectron2.engine import default_setup

from config_utils import apply_yaml_overrides, load_yaml_config
from dataset import HW2_CLASS_NAMES, register_test_split
from logger_utils import setup_logger
from tester import run_inference

logger = logging.getLogger("hw2.test")


def build_parser():
    parser = argparse.ArgumentParser(description="HW2 inference script")
    parser.add_argument("--config", required=True, metavar="YAML")
    parser.add_argument("--gpu-ids", default=None, metavar="IDS")
    return parser


def main():
    args = build_parser().parse_args()
    yaml_cfg = load_yaml_config(args.config)

    gpu_ids = args.gpu_ids or str(yaml_cfg.get("testing", {}).get("gpu_ids", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    output_file = str(yaml_cfg.get("output_file"))
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)

    setup_logger()
    logger.info("Config      : %s", args.config)
    logger.info("GPU IDs     : %s", gpu_ids)
    logger.info("Output file : %s", output_file)

    ds = yaml_cfg.dataset
    test_images_dir = str(ds.get("test_images_dir", "data/test"))
    test_dataset_name = register_test_split(test_images_dir)

    model_config_path = os.path.join(_PROJECT_ROOT, str(yaml_cfg.model.config))
    lazy_cfg = LazyConfig.load(model_config_path)
    lazy_cfg = apply_yaml_overrides(lazy_cfg, yaml_cfg)

    lazy_cfg.dataloader.test.dataset.names = test_dataset_name
    lazy_cfg.dataloader.test.num_workers = 0

    _model_yaml = yaml_cfg.get("model", {})
    for _attr in ("nms_iou_threshold", "nms_max_per_image", "nms_pre_topk"):
        if _attr in _model_yaml:
            try:
                setattr(
                    lazy_cfg.model,
                    _attr,
                    float(_model_yaml[_attr]) if "threshold" in _attr else int(_model_yaml[_attr]),
                )
            except Exception:
                pass

    checkpoint = str(yaml_cfg.get("checkpoint", "")).strip()
    score_threshold = float(yaml_cfg.model.get("score_threshold", 0.05))
    if not checkpoint or not os.path.isfile(checkpoint):
        logger.error("Checkpoint not found at '%s'.", checkpoint)
    logger.info("Checkpoint  : %s", checkpoint)

    lazy_cfg.train.init_checkpoint = checkpoint
    lazy_cfg.train.output_dir = output_dir

    class _FakeArgs:
        config_file = args.config
        dist_url = "auto"
        eval_only = True
        machine_rank = 0
        num_gpus = 1
        num_machines = 1
        opts = []
        resume = False

    default_setup(lazy_cfg, _FakeArgs())

    post_process = dict(yaml_cfg.get("post_process", {}))
    if post_process:
        logger.info("Post-process config: %s", post_process)

    logger.info("Running inference with score_threshold=%.2f …", score_threshold)
    predictions = run_inference(lazy_cfg, score_threshold=score_threshold, post_process=post_process)

    with open(output_file, "w") as f:
        json.dump(predictions, f)
    logger.info("Saved %d predictions → %s", len(predictions), output_file)


if __name__ == "__main__":
    main()
