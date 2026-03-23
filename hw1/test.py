import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TestDataset, get_val_transforms
from model import load_model_from_checkpoint
from utils import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    Prediction,
    TestConfig,
    get_logger,
    set_seed,
)


def load_config(config_path: str) -> TestConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    model_cfg = ModelConfig(
        backbone=raw["model"].get("backbone", "resnet50"),
        pretrained=False,
        num_classes=raw["model"].get("num_classes", 100),
        checkpoint=raw["model"].get("checkpoint", None),
    )
    data_raw = raw.get("data", {})
    data_cfg = DataConfig(
        test_dir=data_raw.get("test_dir", "data/test"),
        image_size=data_raw.get("image_size", 224),
        crop_size=data_raw.get("crop_size", 320),
        resize_size=data_raw.get("resize_size", 334),
        batch_size=data_raw.get("batch_size", 64),
        num_workers=data_raw.get("num_workers", 4),
    )
    output_cfg = OutputConfig(submission_file=raw.get("output", {}).get("submission_file", "submission.csv"))
    return TestConfig(model=model_cfg, data=data_cfg, output=output_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and generate submission CSV")
    parser.add_argument("--config", type=str, default="configs/test.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_test_dataloader(config: TestConfig) -> DataLoader:
    transform = get_val_transforms(resize_size=config.data.resize_size, crop_size=config.data.crop_size)
    test_ds = TestDataset(config.data.test_dir, transform=transform)
    return DataLoader(
        test_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )


def get_inference_device() -> torch.device:
    local_rank = os.environ.get("LOCAL_RANK", 0)
    return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> List[Prediction]:
    model.eval()
    predictions: List[Prediction] = []
    for images, image_names in tqdm(test_loader, desc="Inference"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
        for name, label in zip(image_names, outputs.argmax(dim=1).cpu().tolist()):
            predictions.append(Prediction(image_name=name, label=label))
    return predictions


def save_submission(predictions: List[Prediction], output_path: str) -> None:
    df = pd.DataFrame([{"image_name": p.image_name, "pred_label": p.label} for p in predictions])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger("tester")

    config = load_config(args.config)
    device = get_inference_device()

    test_loader = build_test_dataloader(config)
    logger.info(f"Device: {device}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    model = load_model_from_checkpoint(
        checkpoint_path=config.model.checkpoint,
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
    ).to(device)
    logger.info(f"Loaded checkpoint: {config.model.checkpoint}")

    predictions = predict(model, test_loader, device)
    logger.info(f"Predicted {len(predictions)} images")

    save_submission(predictions, config.output.submission_file)
    logger.info(f"Submission saved to {config.output.submission_file} ({len(predictions)} rows)")


if __name__ == "__main__":
    main()
