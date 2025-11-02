from typing import Any, Dict, List, Tuple

import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)


class Task:
    def __init__(self, cfg: DictConfig) -> None:
        self.task_cfg = cfg
        self.name = self.task_cfg.get("task_name", "Task")

    def run(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        log.info(f"Instantiating datamodule <{self.task_cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(self.task_cfg.data)

        log.info(f"Instantiating model <{self.task_cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(self.task_cfg.model)

        log.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(self.task_cfg.get("logger"))

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(
            self.task_cfg.get("callbacks")
        )

        log.info(f"Instantiating trainer <{self.task_cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            self.task_cfg.trainer, callbacks=callbacks, logger=logger
        )

        object_dict = {
            "cfg": self.task_cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }
        # TODO(shikhar): For api based models, do we still need to create logger inside trainer?
        if logger:
            log.info("Logging hyperparameters!")
            log_hyperparameters(object_dict)

        if self.task_cfg.get("train"):
            log.info("Starting training!")
            trainer.fit(
                model=model,
                datamodule=datamodule,
                ckpt_path=self.task_cfg.get("ckpt_path"),
            )

        train_metrics = trainer.callback_metrics

        if self.task_cfg.get("test"):
            log.info("Starting testing!")
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            log.info(f"Best ckpt path: {ckpt_path}")

        test_metrics = trainer.callback_metrics

        # merge train and test metrics
        metric_dict = {**train_metrics, **test_metrics}

        return metric_dict, object_dict
