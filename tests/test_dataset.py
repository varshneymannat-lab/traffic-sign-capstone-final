from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.prepare_dataset import _stratify_or_none
from src.data.sign_datamodule import TrafficSignCropDataset


def test_traffic_sign_crop_dataset_loads_sample(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "images"
    raw_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(raw_dir / "sign.jpg")
    df = pd.DataFrame(
        [
            {
                "image_path": "images/sign.jpg",
                "xmin": 4,
                "ymin": 4,
                "xmax": 40,
                "ymax": 40,
                "sign_label": "stop",
            }
        ]
    )
    dataset = TrafficSignCropDataset(
        df,
        tmp_path,
        sign_classes=["stop", "speed_limit"],
        transform=None,
    )
    sample = dataset[0]
    assert sample["sign_label"].item() == 0


def test_small_dataset_does_not_force_stratification():
    labels = pd.Series(["stop", "speed_limit", "traffic_light"])

    assert _stratify_or_none(labels, test_size=0.30) is None
