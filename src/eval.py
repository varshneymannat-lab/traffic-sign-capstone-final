from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.video_pipeline import load_classifier


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.data)
    model = load_classifier(cfg.ckpt_path, list(cfg.data.sign_classes), "cpu")
    trainer = instantiate(cfg.trainer, logger=False)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
