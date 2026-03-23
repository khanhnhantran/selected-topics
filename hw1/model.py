import math
from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ModelConfig


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


def make_layer(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes: int = 100, drop_rate: float = 0.0) -> None:
        super().__init__()
        self._num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(3, 44, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(44),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = make_layer(44, 44, num_blocks=2, stride=1)
        self.layer2 = make_layer(44, 88, num_blocks=2, stride=2)
        self.layer3 = make_layer(88, 176, num_blocks=2, stride=2)
        self.layer4 = make_layer(176, 352, num_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(352, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.pool(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))

    @property
    def num_classes(self) -> int:
        return self._num_classes


class ResNetClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet50", num_classes: int = 100, pretrained: bool = True, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)

    @property
    def num_classes(self) -> int:
        return self.model.num_classes

    @property
    def pretrained_cfg(self):
        return self.model.pretrained_cfg


class ResNetNNClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet50", num_classes: int = 100, pretrained: bool = True, drop_rate: float = 0.0) -> None:
        super().__init__()
        self._num_classes = num_classes
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        self.backbone.requires_grad_(False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.backbone.forward_features(x)), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def pretrained_cfg(self):
        return self.backbone.pretrained_cfg


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base + lora * self.scaling

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, scaling={self.scaling:.3f}"


def lora_from_linear(linear: nn.Linear, rank: int, alpha: float, dropout: float) -> LoRALinear:
    lora = LoRALinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=linear.bias is not None,
    )
    lora.weight.data.copy_(linear.weight.data)
    if linear.bias is not None:
        lora.bias.data.copy_(linear.bias.data)
    return lora


def apply_lora(model: nn.Module, rank: int, alpha: float, dropout: float, target_modules: List[str]) -> nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and type(child).__name__ in target_modules:
            setattr(model, name, lora_from_linear(child, rank, alpha, dropout))
        else:
            apply_lora(child, rank, alpha, dropout, target_modules)
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def count_lora_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def create_model(cfg: ModelConfig) -> nn.Module:
    if cfg.backbone == "neural_network":
        model = NeuralNetwork(num_classes=cfg.num_classes, drop_rate=cfg.drop_rate)
    elif cfg.backbone.startswith("resnet_nn:"):
        model = ResNetNNClassifier(
            backbone=cfg.backbone[len("resnet_nn:"):],
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
            drop_rate=cfg.drop_rate,
        )
    else:
        model = ResNetClassifier(
            backbone=cfg.backbone,
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
            drop_rate=cfg.drop_rate,
        )

    if cfg.lora.enabled:
        for param in model.parameters():
            param.requires_grad_(False)
        apply_lora(model, rank=cfg.lora.rank, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, target_modules=cfg.lora.target_modules)

    return model


def load_model_from_checkpoint(checkpoint_path: str, backbone: str, num_classes: int, drop_rate: float = 0.0) -> nn.Module:
    if backbone.startswith("resnet_nn:"):
        model = ResNetNNClassifier(backbone=backbone[len("resnet_nn:"):], num_classes=num_classes, pretrained=False, drop_rate=drop_rate)
    else:
        model = ResNetClassifier(backbone=backbone, num_classes=num_classes, pretrained=False, drop_rate=drop_rate)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return model
