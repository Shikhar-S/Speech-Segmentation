"""Multitasking Segmentation + Recognition model."""

from typing import Any, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MeanMetric, MinMetric

class SegmentRecognizeModel(LightningModule):
    """Multitasking Segmentation + Recognition model.

    Args:
        net: Shared encoder module.
        optimizer: Partial optimizer constructor.
        scheduler: Optional LR scheduler constructor.
        seg_losses: Hydra config dict for segmentation losses.
        pr_losses: Hydra config dict for recognition losses.
        resolution: Temporal upsample factor (output-side).
        audio_sr: Audio sample rate, for timestamp conversions.
        tolerance_ms: Unused. Boundary tolerance is fixed at the
            phone_metrics default (20ms) via
            ``src.metrics.evaluate_boundaries``.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        seg_losses: dict[str, Any] | None = None,
        pr_losses: dict[str, Any] | None = None,
        resolution: int = 1,
        audio_sr: int = 16000,
        tolerance_ms: int = 20,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.audio_sr = audio_sr
        self.resolution = resolution

        dim = net.encoder_output_size()
        self.upsample = (
            nn.Linear(dim, dim * resolution)
            if resolution > 1
            else nn.Identity()
        )

        inject = dict(
            encoder_dim=dim,
            effective_pbf=net.points_by_frames() / resolution,
            audio_sr=audio_sr,
        )
        self.seg_losses = nn.ModuleDict(
            {n: c(**inject) for n, c in (seg_losses or {}).items()}
        )
        self.pr_losses = nn.ModuleDict(
            {n: c(**inject) for n, c in (pr_losses or {}).items()}
        )
        dups = set(self.seg_losses) & set(self.pr_losses)
        if dups:
            raise ValueError(
                f"Head name collision between seg_losses and "
                f"pr_losses: {sorted(dups)}"
            )

        self.val_seg_loss = MeanMetric()
        self.val_seg_loss_best = MinMetric()

    @property
    def effective_pbf(self) -> float:
        """Audio samples per encoder frame after upsampling."""
        return self.net.points_by_frames() / self.resolution

    def _encode(
        self,
        speech: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode speech and apply learned temporal upsampling."""
        features, out_lens = self.net.encode(speech, lengths)
        if isinstance(features, tuple):
            features = features[0]
        B, T, D = features.shape
        R = self.resolution
        features = self.upsample(features).view(B, T, R, D).reshape(B, T * R, D)
        return features, out_lens * R

    def _prepare_seg_targets(self, batch: dict[str, Any]) -> None:
        """Write frame-space indices alongside sample-space boundaries."""
        # Resolution reduction can produce overlapping indices; the
        # seg loss shares target weight across the overlap.
        pbf = self.effective_pbf
        batch["target_start_idx"] = torch.floor(
            batch["target_start"] / pbf
        ).long()
        batch["target_end_idx"] = torch.floor(batch["target_end"] / pbf).long()

    def _apply_losses(
        self,
        stage: str,
        losses: nn.ModuleDict,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: dict[str, Any],
        prefix: str,
        on_step: bool,
    ) -> torch.Tensor:
        """Run every head in ``losses`` and return their weighted sum."""
        total = torch.zeros((), device=features.device)
        for head in losses.values():
            out = head(features, feature_lens, batch, net=self.net)
            total = total + head.weight * out["loss"]
            metrics = head.eval_metrics(out, feature_lens, batch)
            head.log_output(
                stage=stage,
                pl_module=self,
                prefix=prefix,
                output=out,
                eval_metrics=metrics,
                on_step=on_step,
            )
        return total

    def _step(
        self,
        stepname: str,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        on_step = stepname == "train"
        losses = {}
        pr_batch = batch.get("recognition")
        if pr_batch is not None:
            feat, lens = self._encode(
                pr_batch["speech"],
                pr_batch["speech_length"],
            )
            rec_loss = self._apply_losses(
                stepname,
                self.pr_losses,
                feat,
                lens,
                pr_batch,
                f"{stepname}_rec",
                on_step=on_step,
            )
            losses["rec_loss"] = rec_loss
        seg_batch = batch.get("segmentation")
        if seg_batch is not None:
            feat, lens = self._encode(
                seg_batch["speech"],
                seg_batch["speech_length"],
            )
            self._prepare_seg_targets(seg_batch)
            seg_loss = self._apply_losses(
                stepname,
                self.seg_losses,
                feat,
                lens,
                seg_batch,
                f"{stepname}_seg",
                on_step=on_step,
            )
            losses["seg_loss"] = seg_loss

        for name, loss in losses.items():
            self.log(
                f"{stepname}/{name}",
                loss.detach(),
                on_step=on_step,
                on_epoch=not on_step,
            )
        return losses

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        losses = self._step("train", batch, batch_idx)
        total = sum(losses.values())
        self.log(f"train/loss", total.detach(), on_step=True, on_epoch=True)
        return total

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        losses = self._step("val", batch, batch_idx)
        assert (
            "seg_loss" in losses
        ), "Validation step requires segmentation loss, the data must contain 'segmentation' key!"
        self.val_seg_loss(losses["seg_loss"].detach())

    def on_validation_epoch_end(self) -> None:
        """Track best validation segmentation loss."""
        if self.val_seg_loss.weight.item() == 0:
            return
        self.val_seg_loss_best(self.val_seg_loss.compute())
        self.log(
            "val/seg_loss_best",
            self.val_seg_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._step("test", batch, batch_idx)

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        """Decode every registered head on shared encoder features.
        Return schema:
            {'pred': { head_name: head-specific output dict, ... } }
            head-specific output dict is usually {utt_id: List[SegmentationUnit], ...}
        Distributed inference harness will pick up the pred dict and
            store it in jsonl
        """
        features, lens = self._encode(batch["speech"], batch["speech_length"])
        results: dict[str, Any] = {}

        for loss_heads in (self.seg_losses, self.pr_losses):
            for name, head in loss_heads.items():
                out = head.decode(features, lens, batch, net=self.net)
                if out is not None:
                    results[name] = out

        return results

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Log gradient norms."""
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self) -> dict[str, Any]:
        """Build optimizer and optional scheduler."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_seg/rval",
                "interval": "step",
                "frequency": 1,
            },
        }
