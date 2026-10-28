from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.detect_signs import main as detect_signs


def inference_arguments(cfg: DictConfig):
    output = Path(cfg.output_video)
    if output.suffix.lower() != ".mp4":
        raise ValueError("output_video must use the .mp4 extension")
    if cfg.device != "cpu":
        raise ValueError("The recorded Mapillary video pipeline uses CPU inference")
    arguments = [str(cfg.input_video), "--model", str(cfg.yolo_model),
                 "--output-dir", str(output.with_suffix(""))]
    for key, option in [("conf_threshold", "--confidence"), ("image_size", "--image-size"),
                        ("inference_mode", "--inference-mode"), ("max_frames", "--max-frames")]:
        if cfg.get(key) is not None:
            arguments.extend([option, str(cfg[key])])
    if cfg.refine:
        arguments.extend(["--refine", "--classifier-dir", str(cfg.classifier_dir)])
    return arguments


def run_video(cfg: DictConfig):
    arguments = inference_arguments(cfg)
    output = Path(cfg.output_video)
    if output.exists():
        raise FileExistsError("Use a new output_video name to preserve previous predictions")
    detect_signs(arguments)
    produced = output.with_suffix("") / "annotated.mp4"
    if not produced.is_file():
        raise ValueError("The video pipeline did not produce an annotated MP4")
    produced.replace(output)
    print(f"Annotated video: {output}; CSV and timing: {output.with_suffix('')}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    run_video(cfg)


if __name__ == "__main__":
    main()
