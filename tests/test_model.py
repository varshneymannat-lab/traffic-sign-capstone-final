import torch
import pytest

from src.models.sign_classifier import TrafficSignClassifierLitModule


@pytest.mark.parametrize("architecture", ["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
def test_traffic_sign_classifier_forward_shapes(architecture):
    model = TrafficSignClassifierLitModule(
        sign_classes=["stop", "speed_limit", "traffic_light"],
        pretrained=False,
        architecture=architecture,
    )
    sign_logits = model(torch.randn(2, 3, 224, 224))
    assert sign_logits.shape == (2, 3)


def test_metrics_accumulate_samples_and_keep_stages_separate():
    model = TrafficSignClassifierLitModule(sign_classes=["stop", "no_entry"], pretrained=False)
    validation = model.stage_metrics["val_metrics"]
    validation.update(torch.tensor([0]), torch.tensor([0]))
    validation.update(torch.tensor([0, 0, 0]), torch.tensor([1, 1, 1]))
    assert validation.compute()["acc"].item() == pytest.approx(0.25)
    model.stage_metrics["train_metrics"].update(torch.tensor([1]), torch.tensor([1]))
    assert validation.compute()["acc"].item() == pytest.approx(0.25)


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        TrafficSignClassifierLitModule(sign_classes=["stop", "no_entry"], pretrained=False, architecture="unknown")
