from unittest.mock import patch

import pytest
import torch

from src.utils.video_pipeline import load_classifier


def test_missing_classifier_checkpoint_fails_before_model_initialization(tmp_path):
    with patch("src.utils.video_pipeline.TrafficSignClassifierLitModule") as model:
        with pytest.raises(FileNotFoundError, match="Classifier checkpoint not found"):
            load_classifier(tmp_path / "missing.ckpt", ["stop"], "cpu")
        model.assert_not_called()


def test_checkpoint_loading_does_not_download_pretrained_weights(tmp_path):
    checkpoint = tmp_path / "classifier.ckpt"
    torch.save({"state_dict": {}}, checkpoint)
    with patch("src.utils.video_pipeline.TrafficSignClassifierLitModule") as model:
        loaded = load_classifier(checkpoint, ["stop", "traffic_light"], "cpu")
        model.assert_called_once_with(
            sign_classes=["stop", "traffic_light"], pretrained=False, architecture="resnet18"
        )
        loaded.load_state_dict.assert_called_once_with({})
        loaded.eval.assert_called_once()


def test_checkpoint_category_order_must_match(tmp_path):
    checkpoint = tmp_path / "classifier.ckpt"
    torch.save({"state_dict": {}, "hyper_parameters": {"sign_classes": ["traffic_light", "stop"]}}, checkpoint)
    with pytest.raises(ValueError, match="categories do not match"):
        load_classifier(checkpoint, ["stop", "traffic_light"], "cpu")
