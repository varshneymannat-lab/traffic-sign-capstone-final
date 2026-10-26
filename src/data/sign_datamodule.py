from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TrafficSignCropDataset(Dataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        data_dir: str | Path,
        sign_classes: list[str],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.annotations = annotations.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.sign_to_idx = {label: idx for idx, label in enumerate(sign_classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.annotations.iloc[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.data_dir / "raw" / image_path

        image = Image.open(image_path).convert("RGB")
        crop = image.crop((row["xmin"], row["ymin"], row["xmax"], row["ymax"]))
        if self.transform:
            crop = self.transform(crop)

        return {
            "image": crop,
            "sign_label": torch.tensor(self.sign_to_idx[row["sign_label"]], dtype=torch.long),
        }


class TrafficSignDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        annotations_file: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        sign_classes: list[str],
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.annotations_file = Path(annotations_file)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sign_classes = sign_classes

    def setup(self, stage: str | None = None) -> None:
        annotations = pd.read_csv(self.annotations_file)
        self.train_df = annotations[annotations["split"] == "train"]
        self.val_df = annotations[annotations["split"] == "val"]
        self.test_df = annotations[annotations["split"] == "test"]

        train_tfms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
                transforms.RandomRotation(8),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        eval_tfms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.train_dataset = TrafficSignCropDataset(
            self.train_df, self.data_dir, self.sign_classes, train_tfms
        )
        self.val_dataset = TrafficSignCropDataset(
            self.val_df, self.data_dir, self.sign_classes, eval_tfms
        )
        self.test_dataset = TrafficSignCropDataset(
            self.test_df, self.data_dir, self.sign_classes, eval_tfms
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
