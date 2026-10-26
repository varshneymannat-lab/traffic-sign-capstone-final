from __future__ import annotations

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchvision.models import get_model, get_model_weights


class TrafficSignClassifierLitModule(LightningModule):
    def __init__(
        self,
        sign_classes: list[str],
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        architecture: str = "resnet18",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        supported = {"resnet18", "mobilenet_v3_small", "efficientnet_b0"}
        if architecture not in supported:
            raise ValueError(f"Unsupported architecture: {architecture}")
        weights = get_model_weights(architecture).DEFAULT if pretrained else None
        self.backbone = get_model(architecture, weights=weights)
        if architecture == "resnet18":
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            feature_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Identity()
        self.sign_head = nn.Linear(feature_dim, len(sign_classes))
        self.loss_fn = nn.CrossEntropyLoss()

        self.stage_metrics = nn.ModuleDict({
            f"{stage}_metrics": MetricCollection({
                "acc": MulticlassAccuracy(num_classes=len(sign_classes), average="micro"),
                "f1": MulticlassF1Score(num_classes=len(sign_classes), average="macro"),
                "precision": MulticlassPrecision(num_classes=len(sign_classes), average="macro"),
                "recall": MulticlassRecall(num_classes=len(sign_classes), average="macro"),
            }) for stage in ("train", "val", "test")
        })

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.sign_head(features)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        sign_logits = self(batch["image"])
        loss = self.loss_fn(sign_logits, batch["sign_label"])

        sign_preds = sign_logits.argmax(dim=1)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        metrics = self.stage_metrics[f"{stage}_metrics"]
        metrics.update(sign_preds, batch["sign_label"])
        for name, metric in metrics.items():
            self.log(f"{stage}/{name}", metric, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
