from __future__ import annotations

from pathlib import Path

import torch

from src.models.sign_classifier import TrafficSignClassifierLitModule


def load_classifier(
    checkpoint_path: str | Path,
    sign_classes: list[str],
    device: str,
) -> TrafficSignClassifierLitModule:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hyperparameters = state.get("hyper_parameters", {})
    if hyperparameters.get("sign_classes", sign_classes) != sign_classes:
        raise ValueError("Classifier checkpoint categories do not match the requested categories.")
    model = TrafficSignClassifierLitModule(
        sign_classes=sign_classes,
        pretrained=False,
        architecture=hyperparameters.get("architecture", "resnet18"),
    )
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model
