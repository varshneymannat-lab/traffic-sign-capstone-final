from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from scripts.prepare_mtsd_crops import crop_bounds, prepare


def test_crop_bounds_round_outwards_and_clamp():
    row = SimpleNamespace(x_center=0.05, y_center=0.5, box_width=0.2, box_height=0.25)
    assert crop_bounds(row, 100, 100) == (0, 37, 16, 63)


def test_crop_bounds_reject_empty():
    row = SimpleNamespace(x_center=2, y_center=2, box_width=0.1, box_height=0.1)
    with pytest.raises(ValueError, match="Empty"):
        crop_bounds(row, 100, 100)


def test_prepare_inherits_source_scene_split(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (100, 80), (30, 70, 40)).save(source / "scene.jpg")
    rows = [{"image_path": "scene.jpg", "image_key": "scene", "split": "val",
             "official_split": "train", "group_id": 5, "sign_label": "stop",
             "x_center": 0.5, "y_center": 0.5, "box_width": 0.2, "box_height": 0.5}]
    pd.DataFrame(rows).to_csv(source / "annotations_split.csv", index=False)
    destination = tmp_path / "crops"
    prepare(source, destination)
    frame = pd.read_csv(destination / "processed/annotations_split.csv")
    assert frame.iloc[0].split == "val"
    assert frame.iloc[0].group_id == 5
    assert frame.iloc[0].source_image_key == "scene"
    with Image.open(destination / "raw" / frame.iloc[0].image_path) as image:
        assert image.size == (20, 40)
    with pytest.raises(ValueError, match="empty destination"):
        prepare(source, destination)


def test_prepare_rejects_group_leakage(tmp_path):
    rows = [{"image_key": "a", "group_id": 1, "split": "train"},
            {"image_key": "b", "group_id": 1, "split": "test"}]
    pd.DataFrame(rows).to_csv(tmp_path / "annotations_split.csv", index=False)
    with pytest.raises(ValueError, match="group crosses"):
        prepare(tmp_path, tmp_path / "crops")
