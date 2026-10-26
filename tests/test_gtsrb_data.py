import pandas as pd
from PIL import Image
from torchvision import transforms

from src.data.sign_datamodule import TrafficSignDataModule


def test_real_sign_preprocessing_does_not_flip_digits(tmp_path):
    (tmp_path / "raw").mkdir()
    Image.new("RGB", (40, 30), "red").save(tmp_path / "raw/sign.png")
    rows = [{"image_path": "sign.png", "xmin": 2, "ymin": 2, "xmax": 38, "ymax": 28,
             "sign_label": "stop", "split": split} for split in ["train", "val", "test"]]
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame(rows).to_csv(annotations, index=False)
    data = TrafficSignDataModule(str(tmp_path), str(annotations), 128, 1, 0, ["stop", "no_entry"])
    data.setup()
    assert not any(isinstance(t, transforms.RandomHorizontalFlip) for t in data.train_dataset.transform.transforms)
    sample = data.test_dataset[0]
    assert sample["image"].shape == (3, 128, 128)
    assert sample["sign_label"].item() == 0
