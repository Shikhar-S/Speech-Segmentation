"""Classification / Ordinal classification Model.

Uses implementation ideas from https://arxiv.org/pdf/1901.07884
without constrained posteriors.

Run main:
    python -m src.recipe.common.classification_model_module
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.classification import (
    CohenKappa,
    MulticlassAccuracy,
    MulticlassF1Score,
)
from torchmetrics.regression import PearsonCorrCoef, MeanAbsoluteError

from src.model.heads.base_head import BaseHead, InputType, TaskType
from src.utils import RankedLogger
from src.recipe.common.geolocation_loss import GeolocationAngularLoss
from src.metrics.kendalltau import KendallTau
from src.metrics.geolocation import GeolocationMissRate, GeolocationDistanceError

log = RankedLogger(__name__, rank_zero_only=True)


# TODO(shikhar): Rename to probing module, here and configs
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
        ordinal_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["head", "net"])
        self.net = net
        self.encoder_dim = self.net.encoder_output_size()
        self.num_classes = num_classes
        self.classification_head = head(input_dim=self.encoder_dim)
        if self.classification_head.task_type == TaskType.CLASSIFICATION:
            self.criterion = nn.CrossEntropyLoss()
        elif self.classification_head.task_type == TaskType.ORDINAL_REGRESSION:
            assert num_classes >= 2, "Ordinal regression requires at least 2 classes"
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.classification_head.task_type == TaskType.REGRESSION:
            self.criterion = nn.MSELoss()
        elif self.classification_head.task_type == TaskType.GEOLOCATION:
            self.criterion = GeolocationAngularLoss()
        else:
            raise ValueError(
                f"Unsupported task type: {self.classification_head.task_type}"
            )
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.net.eval()
            self.net.requires_grad_(False)
        else:
            self.net.train()
            self.net.requires_grad_(True)
        self.input_type: InputType = input_type
        self.metrics: Dict[str, Dict[str, nn.Module]] = {
            "train": {},
            "val": {},
            "test": {},
        }
        self._register_task_metrics()
        self.passthrough_keys = [
            "split",
            "utt_id",
            "metadata_idx",
            "lang_sym",
            "audio_path",
        ]

    def _register_task_metrics(self):
        # Always track losses
        self.metrics["train"]["loss"] = MeanMetric()
        self.metrics["val"]["loss"] = MeanMetric()
        self.metrics["test"]["loss"] = MeanMetric()
        self.val_loss_best = MinMetric()

        if self.classification_head.task_type in (
            TaskType.CLASSIFICATION,
            TaskType.ORDINAL_REGRESSION,
            TaskType.REGRESSION,
        ):
            self.metrics["train"]["acc"] = MulticlassAccuracy(
                num_classes=self.num_classes
            )
            self.metrics["val"]["acc"] = MulticlassAccuracy(
                num_classes=self.num_classes
            )
            self.metrics["test"]["acc"] = MulticlassAccuracy(
                num_classes=self.num_classes
            )
            self.metrics["val"]["f1"] = MulticlassF1Score(
                num_classes=self.num_classes, average="macro"
            )
            self.metrics["test"]["f1"] = MulticlassF1Score(
                num_classes=self.num_classes, average="macro"
            )
        if self.classification_head.task_type in (
            TaskType.ORDINAL_REGRESSION,
            TaskType.REGRESSION,
        ):
            # Ordinal-focused metrics
            self.metrics["train"]["mae"] = MeanAbsoluteError()
            self.metrics["val"]["mae"] = MeanAbsoluteError()
            self.metrics["test"]["mae"] = MeanAbsoluteError()

            self.metrics["val"]["cohenkappa"] = CohenKappa(
                task="multiclass", num_classes=self.num_classes, weights="quadratic"
            )
            self.metrics["test"]["cohenkappa"] = CohenKappa(
                task="multiclass", num_classes=self.num_classes, weights="quadratic"
            )

            self.metrics["val"]["pcc"] = PearsonCorrCoef()
            self.metrics["test"]["pcc"] = PearsonCorrCoef()

            self.metrics["val"]["kendalltau"] = KendallTau()
            self.metrics["test"]["kendalltau"] = KendallTau()

        if self.classification_head.task_type == TaskType.GEOLOCATION:
            for stage in ["val", "test"]:
                self.metrics[stage]["err_km"] = GeolocationDistanceError()
                self.metrics[stage]["miss_rate_top1"] = GeolocationMissRate(k=1)
                self.metrics[stage]["miss_rate_top5"] = GeolocationMissRate(k=5)
                self.metrics[stage]["miss_rate_top10"] = GeolocationMissRate(k=10)

        for stage, stage_metrics in self.metrics.items():
            for name, metric in stage_metrics.items():
                setattr(self, f"{stage}_{name}", metric)

    def on_fit_start(self) -> None:
        for stage_metrics in self.metrics.values():
            for m in stage_metrics.values():
                m.to(self.device)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> torch.Tensor:
        # print(x_lengths)
        h, h_len = self.net.encode(x, x_lengths)  # (B, T, D), (B,)
        logits = self.classification_head(
            h, h_len
        )  # (B, K) for multiclass, (B, K-1) for ordinal
        return logits

    def _ordinal_targets(self, y: torch.Tensor) -> torch.Tensor:
        """
        y in {0..K-1}
        returns (B, K-1): t[b,k] = 1 if y[b] > k else 0
        eg:     K=4
        y:      [2]
        target: [[1,1,0]]
        """
        thresh = torch.arange(self.num_classes - 1, device=y.device)
        return (y.unsqueeze(1) > thresh.unsqueeze(0)).float()

    def _ordinal_predict(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, K-1) -> predicted class in {0..K-1}
        """
        p = torch.sigmoid(logits)
        return (p > self.hparams.ordinal_threshold).sum(dim=1).long()

    def _update_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        # alwyas update loss
        self.metrics[stage]["loss"](loss)

        if "acc" in self.metrics[stage]:
            self.metrics[stage]["acc"](preds, targets)

        if stage in {"val", "test"} and "f1" in self.metrics[stage]:
            self.metrics[stage]["f1"](preds, targets)

        if "mae" in self.metrics[stage]:
            self.metrics[stage]["mae"](preds.float(), targets.float())

        if stage in {"val", "test"}:
            if "cohenkappa" in self.metrics[stage]:
                self.metrics[stage]["cohenkappa"](preds, targets)
            if "pcc" in self.metrics[stage]:
                # Use the mean of the K-1 logits as the single continuous score for ranking correlation
                # in regression this is just the output value
                continuous_score = logits.mean(dim=-1)
                self.metrics[stage]["pcc"](
                    continuous_score.flatten(), targets.flatten().float()
                )  # pcc on continuous values
            if "kendalltau" in self.metrics[stage]:
                # kendalltau on predicted class labels
                self.metrics[stage]["kendalltau"](preds, targets)

        if self.classification_head.task_type == TaskType.GEOLOCATION:
            if stage in {"val", "test"}:
                self.metrics[stage]["err_km"](logits, targets)
                for key, metric in self.metrics[stage].items():
                    if key.startswith("miss_rate"):
                        metric(logits, targets)

    def _log_stage_metrics(
        self,
        stage: str,
        *,
        on_step: bool,
        on_epoch: bool,
        prog_bar_keys: Tuple[str, ...],
    ) -> None:
        for name, metric in self.metrics[stage].items():
            key = f"{stage}/{name}"
            self.log(
                key,
                metric,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=(name in prog_bar_keys),
                sync_dist=True if stage in {"val", "test"} else False,
                metric_attribute=f"{stage}_{name}",
            )

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["speech"] if self.input_type == "audio" else batch["text"]
        x_lengths = (
            batch["speech_length"] if self.input_type == "audio" else batch["lengths"]
        )
        y = batch["target"]  # (B,) int in [0..K-1] OR (B,2) float for geolocation
        logits = self(x, x_lengths)

        if self.classification_head.task_type == TaskType.CLASSIFICATION:
            y = y.long()
            loss = self.criterion(logits, y)
            preds = logits.argmax(dim=-1)
        elif self.classification_head.task_type == TaskType.ORDINAL_REGRESSION:
            y = y.long()
            # (B, K-1) TODO(shikhar): refactor out in the loss, like geolocation loss
            y_ord = self._ordinal_targets(y)
            loss = self.criterion(logits, y_ord)
            preds = self._ordinal_predict(logits)
        elif self.classification_head.task_type == TaskType.REGRESSION:
            y = y.float()
            loss = self.criterion(logits.squeeze(-1), y)
            preds = logits.squeeze(-1).round().clamp(0, self.num_classes - 1).long()
        elif self.classification_head.task_type == TaskType.GEOLOCATION:
            y = y.float()
            loss = self.criterion(logits, y)
            preds = logits
        return {
            "loss": loss,
            "preds": preds.detach(),
            "targets": y,
            "logits": logits.detach(),
        }

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self) -> None:
        for stage in ("train", "val", "test"):
            for m in self.metrics[stage].values():
                m.reset()
        self.val_loss_best.reset()

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        out = self.model_step(batch)
        self._update_metrics(
            "train", out["loss"], out["preds"], out["targets"], out["logits"]
        )
        if self.classification_head.task_type == TaskType.CLASSIFICATION:
            self._log_stage_metrics(
                "train", on_step=True, on_epoch=True, prog_bar_keys=("loss", "acc")
            )
        elif self.classification_head.task_type in (
            TaskType.ORDINAL_REGRESSION,
            TaskType.REGRESSION,
        ):
            self._log_stage_metrics(
                "train", on_step=True, on_epoch=True, prog_bar_keys=("loss", "mae")
            )
        elif self.classification_head.task_type == TaskType.GEOLOCATION:
            self._log_stage_metrics(
                "train", on_step=True, on_epoch=True, prog_bar_keys=("loss")
            )
        else:
            raise ValueError(
                f"Unsupported task type: {self.classification_head.task_type}"
            )
        return out["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model_step(batch)
        self._update_metrics(
            "val", out["loss"], out["preds"], out["targets"], out["logits"]
        )
        if self.classification_head.task_type == TaskType.CLASSIFICATION:
            self._log_stage_metrics(
                "val", on_step=False, on_epoch=True, prog_bar_keys=("loss", "acc")
            )
        elif self.classification_head.task_type in (
            TaskType.ORDINAL_REGRESSION,
            TaskType.REGRESSION,
        ):
            self._log_stage_metrics(
                "val",
                on_step=False,
                on_epoch=True,
                prog_bar_keys=("loss", "mae", "cohenkappa", "pcc"),
            )
        elif self.classification_head.task_type == TaskType.GEOLOCATION:
            self._log_stage_metrics(
                "val",
                on_step=False,
                on_epoch=True,
                prog_bar_keys=("loss", "err_km", "miss_rate_top1"),
            )
        else:
            raise ValueError(
                f"Unsupported task type: {self.classification_head.task_type}"
            )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model_step(batch)
        self._update_metrics(
            "test", out["loss"], out["preds"], out["targets"], out["logits"]
        )
        if self.classification_head.task_type == TaskType.CLASSIFICATION:
            self._log_stage_metrics(
                "test", on_step=False, on_epoch=True, prog_bar_keys=("loss", "acc")
            )
        elif self.classification_head.task_type in (
            TaskType.ORDINAL_REGRESSION,
            TaskType.REGRESSION,
        ):
            self._log_stage_metrics(
                "test",
                on_step=False,
                on_epoch=True,
                prog_bar_keys=("loss", "mae", "cohenkappa", "pcc"),
            )
        elif self.classification_head.task_type == TaskType.GEOLOCATION:
            self._log_stage_metrics(
                "test", on_step=False, on_epoch=True, prog_bar_keys=("loss")
            )
        else:
            raise ValueError(
                f"Unsupported task type: {self.classification_head.task_type}"
            )

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        out = self.model_step(batch)
        logits, targets, preds = out["logits"], out["targets"], out["preds"]

        # Calculate per-example loss
        if self.classification_head.task_type == TaskType.CLASSIFICATION:
            loss_vec = torch.nn.functional.cross_entropy(
                logits, targets.long(), reduction="none"
            )
        elif self.classification_head.task_type == TaskType.ORDINAL_REGRESSION:
            y_ord = self._ordinal_targets(targets.long())
            loss_vec = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_ord, reduction="none"
            ).mean(dim=-1)
        elif self.classification_head.task_type == TaskType.REGRESSION:
            loss_vec = torch.nn.functional.mse_loss(
                logits.squeeze(-1), targets.float(), reduction="none"
            )
        elif self.classification_head.task_type == TaskType.GEOLOCATION:
            raise NotImplementedError(
                "Per-example loss not implemented for Geolocation. Add a reduce param"
            )
        else:
            raise NotImplementedError(
                "Per-example metrics not implemented for Geolocation task."
            )

        # Prepare passthrough data (pre-indexing to avoid redundant lookups in loop)
        passthrough = {k: batch[k] for k in self.passthrough_keys if k in batch}

        return [
            {
                "target": targets[i].cpu().tolist(),
                "prediction": preds[i].cpu().tolist(),
                "logit": logits[i].cpu().tolist(),
                "loss": loss_vec[i].cpu().tolist(),
                **{
                    k: v[i] if isinstance(v, (list, torch.Tensor)) else v
                    for k, v in passthrough.items()
                },
            }
            for i in range(targets.size(0))
        ]

    def on_validation_epoch_end(self) -> None:
        loss = self.metrics["val"]["loss"].compute()
        self.val_loss_best(loss)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizable_params = []
        for n, p in self.named_parameters():
            if n.startswith("net") and self.freeze_encoder:
                p.requires_grad = False
                continue
            optimizable_params.append(p)
        optimizer = self.hparams.optimizer(params=optimizable_params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
