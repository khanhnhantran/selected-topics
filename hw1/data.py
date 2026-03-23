from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFile
from timm.data.config import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_train_transforms(model=None, **kwargs: Dict[str, Any]) -> transforms.Compose:
    data_config = resolve_model_data_config(model=model)
    return create_transform(
        input_size=(3, 320, 320),
        is_training=True,
        use_prefetcher=False,
        no_aug=False,
        scale=(0.08, 1.0),
        ratio=(3. / 4., 4. / 3.),
        hflip=0.5,
        vflip=0.0,
        color_jitter=0.4,
        auto_augment='rand-m7-mstd0.5-inc1',
        interpolation=data_config.get("interpolation", "bicubic"),
        mean=data_config["mean"],
        std=data_config["std"],
        re_prob=0.25,
        re_mode='pixel',
        re_count=3,
    )


def get_val_transforms(resize_size: int = 256, crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ClassificationDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        class_dirs = sorted(self.root.iterdir(), key=lambda p: int(p.name))
        self.classes = [d.name for d in class_dirs if d.is_dir()]
        self.class_to_idx = {cls: int(cls) for cls in self.classes}

        for class_dir in class_dirs:
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> torch.Tensor:
        counts = torch.zeros(len(self.classes), dtype=torch.long)
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.mean()

    def get_sample_weights(self) -> List[float]:
        counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        class_w = 1.0 / counts.clamp(min=1)
        return [class_w[label].item() for _, label in self.samples]


class TestDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.image_paths: List[Path] = sorted(
            p for p in self.root.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path.stem
