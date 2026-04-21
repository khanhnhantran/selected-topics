import argparse
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config_utils import setup_sys_path

setup_sys_path(_PROJECT_ROOT)

from detectron2.config import LazyConfig
from detectron2.engine import default_setup, launch

from config_utils import apply_cli_opts, apply_yaml_overrides, load_yaml_config
from dataset import register_dataset
from logger_utils import setup_logger
from trainer import train_model

logger = logging.getLogger("hw2.train")


def build_parser():
    parser = argparse.ArgumentParser(description="HW2 training script")
    parser.add_argument("--config", required=True, metavar="YAML")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


def derive_output_dir(config_path):
    parts = os.path.normpath(config_path).split(os.sep)
    try:
        idx = parts.index("train")
        arch = parts[idx + 1]
        backbone = parts[idx + 2]
        run = os.path.splitext(parts[-1])[0]
    except (ValueError, IndexError):
        arch, backbone, run = "model", "backbone", "0"
    return os.path.join("logs", arch, backbone, run)


def update_lr_schedule(lazy_cfg):
    max_iter = lazy_cfg.train.max_iter
    try:
        lazy_cfg.lr_multiplier.scheduler.milestones = [
            int(max_iter * 0.75),
            int(max_iter * 0.90),
        ]
        lazy_cfg.lr_multiplier.scheduler.num_updates = max_iter
        warmup_iters = max(100, int(max_iter * 0.01))
        lazy_cfg.lr_multiplier.warmup_length = warmup_iters / max_iter
    except Exception:
        pass


def main(args):
    from detectron2.utils import comm

    yaml_cfg = load_yaml_config(args.config)
    if args.opts:
        yaml_cfg = apply_cli_opts(yaml_cfg, args.opts)

    resume = args.resume or bool(yaml_cfg.get("training", {}).get("resume", False))
    output_dir = derive_output_dir(args.config)
    os.makedirs(output_dir, exist_ok=True)

    setup_logger(output_dir=output_dir, distributed_rank=comm.get_rank())
    logger.info("Config  : %s", args.config)
    logger.info("Output  : %s", output_dir)

    ds = yaml_cfg.dataset
    register_dataset(
        train_json=ds.train_json,
        train_images_dir=ds.train_images_dir,
        valid_json=ds.get("valid_json"),
        valid_images_dir=ds.get("valid_images_dir"),
        dataset_name_train=ds.get("register_train", "train"),
        dataset_name_valid=ds.get("register_valid", "valid"),
    )

    model_config_path = os.path.join(_PROJECT_ROOT, str(yaml_cfg.model.config))
    lazy_cfg = LazyConfig.load(model_config_path)
    lazy_cfg = apply_yaml_overrides(lazy_cfg, yaml_cfg)
    lazy_cfg.train.output_dir = output_dir
    update_lr_schedule(lazy_cfg)

    num_gpus = args.num_gpus or 1
    try:
        per_gpu_bs = int(yaml_cfg.get("training", {}).get("batch_size", 2))
        lazy_cfg.dataloader.train.total_batch_size = per_gpu_bs * num_gpus * args.num_machines
        logger.info(
            "total_batch_size = %d  (%d per-GPU × %d GPUs × %d machines)",
            lazy_cfg.dataloader.train.total_batch_size,
            per_gpu_bs,
            num_gpus,
            args.num_machines,
        )
    except Exception:
        pass

    class _FakeArgs:
        config_file = args.config
        dist_url = args.dist_url
        eval_only = False
        machine_rank = args.machine_rank
        num_gpus = num_gpus
        num_machines = args.num_machines
        opts = []
        resume = resume

    default_setup(lazy_cfg, _FakeArgs())
    train_model(lazy_cfg, resume=resume)


if __name__ == "__main__":
    args = build_parser().parse_args()
    yaml_cfg = load_yaml_config(args.config)
    t_cfg = yaml_cfg.get("training", {})

    gpu_ids_raw = args.gpu_ids or t_cfg.get("gpu_ids", None)
    if gpu_ids_raw is not None:
        if isinstance(gpu_ids_raw, (list, tuple)):
            gpu_ids_str = ",".join(str(g) for g in gpu_ids_raw)
        else:
            gpu_ids_str = str(gpu_ids_raw).strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        num_gpus = len(gpu_ids_str.split(","))
        print(f"[train.py] CUDA_VISIBLE_DEVICES={gpu_ids_str}  →  {num_gpus} GPU(s)")
    else:
        num_gpus = int(args.num_gpus or t_cfg.get("num_gpus", 1))

    args.num_gpus = num_gpus

    launch(
        main,
        num_gpus_per_machine=num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
