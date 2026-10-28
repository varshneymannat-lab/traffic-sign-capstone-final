from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from src.infer_video import inference_arguments, run_video


def config(**overrides):
    values = {"input_video": "input.mp4", "output_video": "outputs/prediction.mp4",
              "yolo_model": "models/mtsd/detector.onnx", "device": "cpu", "refine": False,
              "classifier_dir": "models/mtsd_classifier", "max_frames": None,
              "conf_threshold": None, "image_size": None, "inference_mode": None}
    return OmegaConf.create({**values, **overrides})


def test_original_config_name_inherits_mapillary_categories():
    directory = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=directory, version_base="1.3"):
        cfg = compose(config_name="infer")
    assert cfg.data.sign_classes == ["stop", "no_entry", "yield", "speed_limit_30", "speed_limit_50"]
    assert cfg.yolo_model == "models/mtsd/detector.onnx"


def test_argument_adapter_preserves_metadata_defaults():
    assert inference_arguments(config()) == ["input.mp4", "--model", "models/mtsd/detector.onnx",
                                            "--output-dir", "outputs/prediction"]


def test_optional_crop_stage_and_frame_limit_are_explicit():
    args = inference_arguments(config(refine=True, max_frames=2))
    assert args[-3:] == ["--refine", "--classifier-dir", "models/mtsd_classifier"]
    assert args[args.index("--max-frames") + 1] == "2"


def test_existing_video_is_not_overwritten(tmp_path):
    video = tmp_path / "previous.mp4"
    video.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        run_video(config(output_video=str(video)))
    assert video.read_bytes() == b"existing"


def test_configured_video_is_saved_with_csv_beside_it(tmp_path, monkeypatch):
    import src.infer_video as module

    def fake_detector(arguments):
        folder = Path(arguments[arguments.index("--output-dir") + 1])
        folder.mkdir()
        (folder / "annotated.mp4").write_bytes(b"test-output")
        (folder / "detections.csv").write_text("frame,class\n")

    monkeypatch.setattr(module, "detect_signs", fake_detector)
    output = tmp_path / "prediction.mp4"
    run_video(config(output_video=str(output)))
    assert output.read_bytes() == b"test-output"
    assert (tmp_path / "prediction/detections.csv").is_file()


def test_unsupported_output_extension_is_rejected():
    with pytest.raises(ValueError, match=".mp4"):
        inference_arguments(config(output_video="prediction.avi"))
