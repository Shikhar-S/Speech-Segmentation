"""Lightning style Classification Model.

Run main:
    python -m src.recipe.common.classification_model_module
"""

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)
from lightning.pytorch.utilities import grad_norm
from src.utils import RankedLogger
from src.model.heads.base_head import BaseHead, InputType, TaskType

log = RankedLogger(__name__, rank_zero_only=True)


class ClassificationModel(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        head: BaseHead,
        num_classes: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        freeze_encoder: bool = True,
        input_type: InputType = "audio",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["head", "net"])
        self.net = net
        self.encoder_dim = self.net.encoder_output_size()
        self.num_classes = num_classes
        self.classification_head = head(input_dim=self.encoder_dim)
        if (
            getattr(self.classification_head, "task_type", TaskType.CLASSIFICATION)
            != TaskType.CLASSIFICATION
        ):
            raise ValueError(
                f"ClassificationModel requires a classification head, "
                f"got {getattr(self.classification_head, 'task_type', None)}"
            )

        self.criterion = nn.CrossEntropyLoss()
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.net.eval()
            self.net.requires_grad_(False)
        else:
            self.net.train()
            self.net.requires_grad_(True)

        # Input mode: "audio" or "ipa"
        self.input_type: InputType = input_type
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> torch.Tensor:
        h, h_len = self.net.encode(x, x_lengths)  # (B, T, D), (B,)
        logits = self.classification_head(h, h_len)  # (B, num_classes)
        return logits

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["speech"] if self.input_type == "audio" else batch["text"]
        x_lengths = (
            batch["speech_length"] if self.input_type == "audio" else batch["lengths"]
        )
        y_target = batch["target"]
        logits = self(x, x_lengths)
        loss = self.criterion(logits, y_target)
        preds = logits.argmax(dim=-1)
        return {
            "loss": loss,
            "logits": logits.detach(),
            "targets": y_target,
            "preds": preds.detach(),
        }

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_loss_best.reset()

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        out = self.model_step(batch)
        self.train_loss(out["loss"])
        self.train_acc(out["preds"], out["targets"])
        self.log(
            "train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return out["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model_step(batch)
        self.val_loss(out["loss"])
        self.val_acc(out["preds"], out["targets"])
        self.val_f1(out["preds"], out["targets"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model_step(batch)

        self.test_loss(out["loss"])
        self.test_acc(out["preds"], out["targets"])
        self.test_f1(out["preds"], out["targets"])

        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.freeze_encoder:
            optimizable_params = [
                p for n, p in self.named_parameters() if not n.startswith("net.")
            ]
        else:
            optimizable_params = list(self.parameters())
        optimizer = self.hparams.optimizer(params=optimizable_params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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
