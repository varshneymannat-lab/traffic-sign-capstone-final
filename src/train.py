from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    Path(cfg.paths.model_dir, "checkpoints").mkdir(parents=True, exist_ok=True)

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)
    callbacks = [instantiate(callback_cfg) for callback_cfg in cfg.callbacks.values()]
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    if cfg.train:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if cfg.test_after_training:
        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path="best" if cfg.train else cfg.ckpt_path,
        )


if __name__ == "__main__":
    main()
