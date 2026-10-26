from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.utils.video_pipeline import load_classifier


@hydra.main(version_base="1.3", config_path="../configs", config_name="export")
def main(cfg: DictConfig) -> None:
    model = load_classifier(cfg.ckpt_path, list(cfg.data.sign_classes), "cpu")

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["sign_crop"],
        output_names=["sign_logits"],
        dynamic_axes={
            "sign_crop": {0: "batch"},
            "sign_logits": {0: "batch"},
        },
        opset_version=cfg.opset_version,
        dynamo=False,
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
