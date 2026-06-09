from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .config import CHECKPOINT_PATH, NUM_CLASSES


def build_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(model, checkpoint_path=CHECKPOINT_PATH, map_location="cpu"):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint}. Train the model first with `python -m src.train` "
            "or place a compatible checkpoint in the models/ directory."
        )

    state_dict = torch.load(str(checkpoint), map_location=map_location)
    model.load_state_dict(state_dict)
    return model


def build_inference_model(checkpoint_path=CHECKPOINT_PATH, device="cpu"):
    model = build_model(pretrained=False, freeze_backbone=False)
    model = load_checkpoint(model, checkpoint_path=checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model
