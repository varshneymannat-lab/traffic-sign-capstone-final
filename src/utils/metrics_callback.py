from __future__ import annotations

import json
from pathlib import Path

import torch
from lightning import Callback, Trainer
from lightning.pytorch import LightningModule


class JsonMetricsWriter(Callback):
    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = float(value.detach().cpu())
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
        self.output_path.write_text(json.dumps(metrics, indent=2))
