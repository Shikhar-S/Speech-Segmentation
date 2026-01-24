"""Module for phone recognition.

Usage:
    python -m src.recipe.phone_recognition.model_module
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from lightning.pytorch.utilities import grad_norm


def get_w2v2ph_schedule(
    optimizer, num_training_steps: int, encoder_unfreeze_step: int = 0
):
    """
    Implements the schedule:
    1. Warmup (0-10%): Linear increase from 0 to 1.
    2. Constant (10-50%): Constant at 1.
    3. Decay (50-100%): Linear decay from 1 to 0.

    Additionally, if encoder_unfreeze_step > 0, the encoder parameters are
    frozen until that step is reached.
    """

    def three_piece_factor(current_step: int):
        warmup_steps = int(0.1 * num_training_steps)
        # The constant phase lasts for 40% of updates, so it ends at 10% + 40% = 50%
        constant_end_step = int(0.5 * num_training_steps)

        if current_step < warmup_steps:
            # Phase 1: Linear Warmup
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < constant_end_step:
            # Phase 2: Constant
            return 1.0
        else:
            # Phase 3: Linear Decay
            decay_steps = num_training_steps - constant_end_step
            progress = current_step - constant_end_step
            return max(0.0, 1.0 - (progress / float(max(1, decay_steps))))

    def encoder_lambda(current_step: int):
        # encoder layers
        if current_step < encoder_unfreeze_step:
            return 0.0
        else:
            return three_piece_factor(current_step)

    def head_lambda(current_step: int):
        # ctc head
        return three_piece_factor(current_step)

    return LambdaLR(optimizer, lr_lambda=[head_lambda, encoder_lambda])


class PhoneRecognitionModel(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        inference_strategy: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "inference_strategy"])
        self.net = net
        self.inference_strategy = inference_strategy
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
        if hasattr(self.net, "frontend") and self.net.frontend is not None:
            self.net.frontend.eval()
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_loss_best.reset()  # clears on !resume!

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

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if self.inference_strategy is None:
            raise RuntimeError("Inference engine not provided.")
        speech = batch["speech"]
        speech_lengths = batch["speech_length"]
        results = self.inference_strategy(
            model=self.net, speech=speech, speech_lengths=speech_lengths
        )
        return results

    def set_inference_strategy(self, inference_strategy_cls: Any) -> None:
        self.inference_strategy = inference_strategy_cls(
            self.net.token_list, self.blank_id
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_params = self.net.get_trainable_parameters()
        # must have two keys - head and encoder
        # for ctc and encoder respectively
        optimizer = self.hparams.optimizer(
            params=[
                {"params": trainable_params["head"], "name": "head"},
                {"params": trainable_params["encoder"], "name": "encoder"},
            ]
        )
        scheduler_cls = self.hparams.scheduler

        if scheduler_cls is not None:
            scheduler = scheduler_cls(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # only for ReduceLROnPlateau
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
