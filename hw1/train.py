import argparse
import dataclasses
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import ClassificationDataset, get_train_transforms, get_val_transforms
from model import create_model, get_lora_state_dict, count_lora_parameters
from utils import (
    BatchResult,
    DataConfig,
    EpochResult,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainingConfig,
    format_duration,
    get_logger,
    log_model_size,
    log_performance_csv,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_curves,
    set_seed,
    start_timer,
    stop_timer,
)


def load_config(config_path: str) -> TrainConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    model_raw = dict(raw.get("model", {}))
    lora_raw = model_raw.pop("lora", {})
    model_cfg = ModelConfig(**model_raw, lora=LoraConfig(**lora_raw))
    data_raw = raw.get("data", {})
    data_cfg = DataConfig(
        train_dir=data_raw.get("train_dir", "data/train"),
        val_dir=data_raw.get("val_dir", "data/val"),
        test_dir=data_raw.get("test_dir", "data/test"),
        image_size=data_raw.get("image_size", 224),
        crop_size=data_raw.get("crop_size", 320),
        resize_size=data_raw.get("resize_size", 334),
        batch_size=data_raw.get("batch_size", 64),
        num_workers=data_raw.get("num_workers", 4),
        use_augmentation=data_raw.get("use_augmentation", True),
    )
    return TrainConfig(
        model=model_cfg,
        data=data_cfg,
        training=TrainingConfig(**raw.get("training", {})),
        output=OutputConfig(**raw.get("output", {})),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1, 0, False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank, True


def setup_logging(log_dir: str, rank: int) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(Path(log_dir) / f"train_{timestamp}.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)


def log_config(config: TrainConfig, args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"  config file : {args.config}")
    logger.info(f"  seed        : {args.seed}")
    for section_name, section in dataclasses.asdict(config).items():
        logger.info(f"  [{section_name}]")
        for k, v in section.items():
            logger.info(f"    {k}: {v}")
    logger.info("=" * 60)


def get_device(is_distributed: bool, local_rank: int) -> torch.device:
    if is_distributed:
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config: TrainConfig, device: torch.device, is_distributed: bool, local_rank: int) -> nn.Module:
    base = create_model(config.model).to(device)
    if is_distributed:
        return DDP(base, device_ids=[local_rank], output_device=local_rank)
    return base


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def build_dataloaders(config: TrainConfig, model: nn.Module, rank: int, world_size: int, is_distributed: bool):
    train_tf = get_train_transforms(model=unwrap_model(model))
    val_tf = get_val_transforms(resize_size=config.data.resize_size, crop_size=config.data.crop_size)

    train_ds = ClassificationDataset(config.data.train_dir, transform=train_tf)
    val_ds = ClassificationDataset(config.data.val_dir, transform=val_tf)

    if is_distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        train_shuffle = False
    elif config.training.use_weighted_sampler:
        train_sampler = WeightedRandomSampler(
            weights=train_ds.get_sample_weights(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


def build_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad] if config.model.lora.enabled else list(model.parameters())
    cfg = config.training
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(config: TrainConfig, optimizer: torch.optim.Optimizer, train_loader: DataLoader):
    cfg = config.training
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)

    if cfg.scheduler == "cosine":
        min_lr_ratio = cfg.min_lr / cfg.lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return 1e-4 + (1.0 - 1e-4) * step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif cfg.scheduler == "step":
        if warmup_steps > 0:
            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return 1e-4 + (1.0 - 1e-4) * step / max(1, warmup_steps)
                return 0.1 ** ((step - warmup_steps) // 10)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
    rank: int,
    world_size: int,
    is_distributed: bool,
    train_sampler,
) -> BatchResult:
    model.train()
    if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(epoch)

    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()
    is_main = rank == 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False, disable=not is_main, colour="green")
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images, labels_a, labels_b, lam = mixup_data(images, labels, config.training.mixup_alpha)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam) / config.training.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.training.accumulation_steps == 0:
            if config.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and scaler.get_scale() == scale_before:
                scheduler.step()

        total_loss += loss.item() * config.training.accumulation_steps
        preds = outputs.argmax(dim=1)
        correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
        total += labels.size(0)
        if is_main:
            pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

    if is_distributed:
        metrics = torch.tensor([total_loss, correct, total], dtype=torch.float64, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss = metrics[0].item() / world_size
        correct = int(metrics[1].item())
        total = int(metrics[2].item())

    return BatchResult(loss=total_loss / len(train_loader), correct=correct, total=total)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    is_distributed: bool,
) -> BatchResult:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    if is_distributed:
        val_sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, sampler=val_sampler, num_workers=val_loader.num_workers, pin_memory=True)
    else:
        loader = val_loader

    for images, labels in tqdm(loader, desc="Validation", leave=False, disable=rank != 0):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    if is_distributed:
        metrics = torch.tensor([total_loss, correct, total], dtype=torch.float64, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss = metrics[0].item() / world_size
        correct = int(metrics[1].item())
        total = int(metrics[2].item())

    return BatchResult(loss=total_loss / len(loader), correct=correct, total=total)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    val_acc: float,
    is_best: bool,
    checkpoints: List[str],
    logger,
) -> None:
    ckpt_dir = Path(config.output.checkpoint_dir)
    lora_cfg = config.model.lora
    model_state = get_lora_state_dict(unwrap_model(model)) if lora_cfg.enabled and lora_cfg.save_lora_only else unwrap_model(model).state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "backbone": config.model.backbone,
        "num_classes": config.model.num_classes,
        "lora": lora_cfg.enabled,
    }
    ckpt_path = str(ckpt_dir / f"epoch_{epoch:03d}_acc{val_acc:.4f}.pth")
    torch.save(checkpoint, ckpt_path)
    checkpoints.append(ckpt_path)

    if is_best:
        torch.save(checkpoint, str(ckpt_dir / "best_model.pth"))
        if logger:
            logger.info(f"New best model saved: val_acc={val_acc:.4f}")

    if len(checkpoints) > config.output.save_top_k:
        old = checkpoints.pop(0)
        if os.path.exists(old):
            os.remove(old)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str, device: torch.device, logger) -> float:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_val_acc = checkpoint.get("val_acc", 0.0)
    if logger:
        logger.info(f"Resumed from {path} (epoch {checkpoint.get('epoch', 0)}, val_acc={best_val_acc:.4f})")
    return best_val_acc


def run_training(
    config: TrainConfig,
    rank: int,
    world_size: int,
    local_rank: int,
    is_distributed: bool,
    model_name: str,
) -> None:
    is_main = rank == 0
    device = get_device(is_distributed, local_rank)
    logger = get_logger("trainer", log_dir=config.output.log_dir) if is_main else None

    model = build_model(config, device, is_distributed, local_rank)
    train_loader, val_loader, train_sampler = build_dataloaders(config, model, rank, world_size, is_distributed)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, train_loader)

    class_weights = train_loader.dataset.get_class_weights().to(device) if config.training.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.training.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    csv_path = None
    charts_dir = None
    writer = None

    if is_main:
        Path(config.output.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=config.output.log_dir)
        charts_base = Path("charts") / model_name
        charts_base.mkdir(parents=True, exist_ok=True)
        exp_id = sum(1 for p in charts_base.iterdir() if p.is_dir())
        charts_dir = charts_base / str(exp_id)
        charts_dir.mkdir(parents=True, exist_ok=True)
        csv_path = charts_dir / "training.csv"
        pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]).to_csv(csv_path, index=False)

        if logger:
            logger.info(f"Device: {device}  |  World size: {world_size}")
            logger.info(f"Backbone: {config.model.backbone}")
            if config.model.lora.enabled:
                trainable, total = count_lora_parameters(unwrap_model(model))
                logger.info(f"LoRA rank={config.model.lora.rank} alpha={config.model.lora.alpha} | trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
            logger.info(f"Train samples: {len(train_loader.dataset)}")
            logger.info(f"Val samples:   {len(val_loader.dataset)}")
            log_model_size(unwrap_model(model), logger)

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    checkpoints: List[str] = []

    if config.model.checkpoint:
        best_val_acc = load_checkpoint(model, optimizer, config.model.checkpoint, device, logger)

    if is_main and logger:
        logger.info(f"Starting training for {config.training.epochs} epochs")

    for epoch in range(1, config.training.epochs + 1):
        t_start = start_timer()
        train_result = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, config, device, epoch, rank, world_size, is_distributed, train_sampler)
        val_result = validate(model, val_loader, criterion, device, rank, world_size, is_distributed)
        elapsed = stop_timer(t_start)

        current_lr = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_result.accuracy)

        result = EpochResult(epoch=epoch, train_loss=train_result.loss, train_acc=train_result.accuracy, val_loss=val_result.loss, val_acc=val_result.accuracy, lr=current_lr)

        if is_main and logger:
            logger.info(f"Epoch {epoch:03d} | train_loss={result.train_loss:.4f} train_acc={result.train_acc:.4f} | val_loss={result.val_loss:.4f} val_acc={result.val_acc:.4f} | lr={result.lr:.2e} | time={format_duration(elapsed)}")

        if is_main and writer:
            writer.add_scalars("loss", {"train": result.train_loss, "val": result.val_loss}, epoch)
            writer.add_scalars("acc", {"train": result.train_acc, "val": result.val_acc}, epoch)
            writer.add_scalar("lr", result.lr, epoch)

        if is_main and csv_path:
            pd.DataFrame([{"epoch": epoch, "train_loss": round(result.train_loss, 6), "train_acc": round(result.train_acc, 6), "val_loss": round(result.val_loss, 6), "val_acc": round(result.val_acc, 6), "lr": round(result.lr, 8)}]).to_csv(csv_path, mode="a", index=False, header=False)

        if is_main and charts_dir:
            plot_training_curves(str(csv_path), save_dir=str(charts_dir))
            class_names = val_loader.dataset.classes if hasattr(val_loader.dataset, "classes") else None
            plot_confusion_matrix(unwrap_model(model), val_loader, device, class_names=class_names, save_dir=str(charts_dir))
            plot_roc_curves(unwrap_model(model), val_loader, device, num_classes=config.model.num_classes, class_names=class_names, save_dir=str(charts_dir))

        is_best = result.val_acc > best_val_acc
        if is_best:
            best_val_acc = result.val_acc
            best_train_acc = result.train_acc
            best_val_loss = result.val_loss
            best_train_loss = result.train_loss

        if is_main:
            save_checkpoint(model, optimizer, config, epoch, result.val_acc, is_best, checkpoints, logger)

        if is_distributed:
            dist.barrier()

    if is_main:
        if logger:
            logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
        if writer:
            writer.close()
        model_size = sum(p.numel() for p in unwrap_model(model).parameters())
        log_performance_csv(
            model_name=model_name,
            model_size=model_size,
            epochs=config.training.epochs,
            lr=config.training.lr,
            optimizer=config.training.optimizer,
            image_size=config.data.crop_size,
            train_acc=best_train_acc,
            val_acc=best_val_acc,
            train_loss=best_train_loss,
            val_loss=best_val_loss,
        )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, is_distributed = init_distributed()
    set_seed(args.seed)
    config = load_config(args.config)
    setup_logging(config.output.log_dir, rank)
    if rank == 0:
        log_config(config, args)
    run_training(config, rank, world_size, local_rank, is_distributed, model_name=Path(args.config).stem)
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
