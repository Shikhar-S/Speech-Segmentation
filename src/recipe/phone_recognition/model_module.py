"""Module for phone recognition.

Usage:
    python -m src.recipe.phone_recognition.model_module
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from lightning.pytorch.utilities import grad_norm


class PhoneRecognitionModel(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        inference: Optional[Any] = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "inference"])
        self.freeze_encoder = freeze_encoder
        self.net = net
        self.inference = inference
        self.blank_id: Optional[int] = getattr(self.net, "blank_id", None)
        # Loss tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # TODO(shikhar): fix typo throughtout length --> lengths
        speech = batch["speech"]
        speech_length = batch["speech_length"]
        text = batch["text"]
        text_length = batch["text_length"]
        return self.net(
            speech=speech,
            speech_lengths=speech_length,
            text=text,
            text_lengths=text_length,
        )

    def on_before_optimizer_step(self, optimizer) -> None:
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_loss_best.reset()

    def _run_stage(
        self,
        split: str,
        batch: Dict[str, torch.Tensor],
        *,
        log_on_step: bool,
    ) -> Dict[str, torch.Tensor]:
        out = self(batch)
        loss_metric = getattr(self, f"{split}_loss")
        loss_metric(out["loss"].detach())
        self.log(
            f"{split}/loss",
            loss_metric,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
        )
        # log all stats
        for k, v in out["stats"].items():
            self.log(
                f"{split}/{k}",
                v,
                on_step=log_on_step,
                on_epoch=True,
                prog_bar=False,
            )
        return out

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._run_stage("train", batch, log_on_step=True)["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._run_stage("val", batch, log_on_step=False)

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._run_stage("test", batch, log_on_step=False)

    def _pr_inference(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ) -> Any:
        speech = batch["speech"]
        speech_length = batch["speech_length"]
        segment_ids = batch.get("segment_id")

        predictions: List[Dict[str, Any]] = []
        for idx in range(speech.size(0)):
            waveform = speech[idx, : speech_length[idx].item()]
            inference_result = self.inference(
                speech=waveform,
                speech_length=speech_length[idx],
            )
            entry: Dict[str, Any] = {
                "segment_id": (
                    segment_ids[idx]
                    if segment_ids is not None
                    else f"d{dataloader_idx}_b{batch_idx}_utt{idx}"
                ),
                "predicted_transcript": inference_result[0].get("predicted_transcript"),
            }
            predictions.append(entry)
        return predictions

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Any:
        if self.inference is None:
            raise RuntimeError("Inference object is required for predict_step.")
        return self._pr_inference(batch, batch_idx, dataloader_idx)

    def configure_optimizers(self) -> Dict[str, Any]:
        # skip params for inference obj
        trainable_params = self.net.get_trainable_parameters(self.freeze_encoder)
        optimizer = self.hparams.optimizer(params=trainable_params)
        scheduler_cls = self.hparams.scheduler

        if scheduler_cls is not None:
            scheduler = scheduler_cls(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
