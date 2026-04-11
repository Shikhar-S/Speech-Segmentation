"""Multitasking Segmentation + Recognition model.

Usage:
    python -m src.recipe.segment_recognize.model_module
"""

from typing import Any, Tuple

import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MeanMetric, MinMetric

def convert_pointstamps_to_frame_indices(
    start: torch.Tensor,
    end: torch.Tensor,
    points_by_frames: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert target start and end to indices based on net's resolution."""
    # At 40ms shift, 25 logmel features per second (16000 points),
    # 640 points per frames
    # P / (points/frames) = F
    start_idx = torch.floor(start / points_by_frames).long()
    end_idx = torch.floor(end / points_by_frames).long()
    # NOTE(shikhar): Since we reduce the resolution at this step,
    # there may be overlaps. In the loss, target weight is shared.
    return start_idx, end_idx


class SegmentRecognizeModel(LightningModule):
    """Multitasking Segmentation + Recognition model.

    Losses are provided as Hydra config dicts and instantiated at
    construction time.  Each ``LossModule`` owns its parameters,
    criterion, and metric trackers.

    Args:
        net: Shared encoder module.
        optimizer: Partial optimizer constructor.
        scheduler: Optional LR scheduler constructor.
        seg_losses: Hydra config dict for segmentation losses.
        pr_losses: Hydra config dict for recognition losses.
        resolution: Temporal upsample factor for segmentation (upsample at output by this factor).
        audio_sr: Audio sample rate for timestamp conversions (upsample at input to this rate).
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.audio_sr = audio_sr
        self.resolution = resolution

        dim = net.encoder_output_size()
        self.upsample = nn.Linear(dim, dim * resolution) if resolution > 1 else nn.Identity()

        # Runtime-derived values injected into each loss.
        pbf = net.points_by_frames() / resolution
        injection_args = dict(
            encoder_dim=dim,
            effective_pbf=pbf,
            audio_sr=audio_sr,
        )
        self.seg_losses = nn.ModuleDict(
            _instantiate_losses(seg_losses, injection_args),
        )
        self.pr_losses = nn.ModuleDict(
            _instantiate_losses(pr_losses, injection_args),
        )

        # Group-level tracking for checkpoint monitoring.
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
        """Encode speech and upsample for segmentation."""
        features, out_lens = self.net.encode(speech, lengths)
        if isinstance(features, tuple):
            features = features[0]
        return self._maybe_upsample_features(features, out_lens)

    def _maybe_upsample_features(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply learned temporal upsampling, based on resolution."""
        B, T, D = features.shape
        R = self.resolution
        features = (
            self.upsample(features)
            .view(B, T, R, D)
            .reshape(B, T * R, D)
        )
        return features, lengths * R

    def _prepare_seg_targets(
        self, batch: dict[str, Any],
    ) -> None:
        """Convert sample-level timestamps to frame indices."""
        start_idx, end_idx = convert_pointstamps_to_frame_indices(
            batch["target_start"],
            batch["target_end"],
            self.effective_pbf,
        )
        batch["target_start_idx"] = start_idx
        batch["target_end_idx"] = end_idx

    def training_step(
        self, batch: dict[str, Any], batch_idx: int,
    ) -> torch.Tensor:
        """Dispatch sub-batches to the registered loss modules."""
        pr_batch = batch.get("recognition")
        seg_batch = batch.get("segmentation")
        loss = torch.tensor(0.0, device=self.device)

        if pr_batch is not None:
            feat, lens = self._encode(
                pr_batch["speech"], pr_batch["speech_length"],
            )
            loss = self._apply_losses(
                self.pr_losses, feat, lens, pr_batch,
                loss, prefix="train", on_step=True,
            )

        if seg_batch is not None:
            feat, lens = self._encode(
                seg_batch["speech"],
                seg_batch["speech_length"],
            )
            self._prepare_seg_targets(seg_batch)
            loss = self._apply_losses(
                self.seg_losses, feat, lens, seg_batch,
                loss, prefix="train", on_step=True,
            )

        self.log(
            "train/loss", loss.detach(),
            on_step=True, on_epoch=True,
        )
        return loss

    def _normalize_val_batch(
        self, batch: dict[str, Any],
    ) -> None:
        """Remap segmentation-only val keys to the unified schema.

        TODO(shikhar): Simplify from dataloader side.
        """
        if "text" not in batch and "target" in batch:
            batch["text"] = batch.pop("target")
            batch["text_length"] = batch.pop("target_length")
        batch.setdefault(
            "lang_sym", ["<eng>"] * batch["speech"].shape[0],
        )

    def validation_step(
        self, batch: Any, batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Run all losses on a validation batch."""
        # TODO(shikhar): Refactor to a single _step fn with mode=valid/train/test
        self._normalize_val_batch(batch)
        features, logit_len = self._encode(
            batch["speech"], batch["speech_length"],
        )
        zero = torch.tensor(0.0, device=self.device)
        self._apply_losses(
            self.pr_losses, features, logit_len, batch,
            zero, "val_seg", on_step=False,
        )
        if batch.get("target_start") is None:
            return
        self._prepare_seg_targets(batch)
        seg_total = self._apply_losses(
            self.seg_losses, features, logit_len, batch,
            zero, "val_seg", on_step=False,
        )
        self.val_seg_loss(seg_total.detach())
        self.log(
            "val_seg/seg_loss", seg_total.detach(),
            on_step=False, on_epoch=True,
        )

    def on_validation_epoch_end(self) -> None:
        """Track best validation segmentation loss."""
        if self.val_seg_loss.weight.item() == 0:
            return
        loss = self.val_seg_loss.compute()
        self.val_seg_loss_best(loss)
        self.log(
            "val_seg/loss_best",
            self.val_seg_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Log gradient norms."""
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self) -> dict[str, Any]:
        """Build optimizer and optional scheduler."""
        optimizer = self.hparams.optimizer(
            params=self.parameters(),
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_seg/rval",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def _apply_losses(
        self,
        losses: nn.ModuleDict,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: dict[str, Any],
        loss: torch.Tensor,
        prefix: str,
        on_step: bool,
    ) -> torch.Tensor:
        """Run each loss module and accumulate weighted loss."""
        for lm in losses.values():
            out = lm(features, feature_lens, batch, net=self.net)
            loss = loss + lm.weight * out["loss"]
            metrics = lm.eval_metrics(
                out, feature_lens, batch,
            )
            lm.log_output(
                self, prefix, out, metrics, on_step=on_step,
            )
        return loss


def _instantiate_losses(
    cfg: dict[str, Any] | None,
    inject: dict[str, Any],
) -> dict[str, nn.Module]:
    """Instantiate loss modules from Hydra config.

    Args:
        cfg: Mapping of ``{name: DictConfig}`` with ``_target_``
            keys.  May be ``None`` (returns empty dict).
        inject: Extra kwargs merged into each instantiate call
            (``encoder_dim``, ``effective_pbf``, ``audio_sr``).

    Returns:
        Dict of instantiated ``LossModule`` objects.
    """
    if not cfg:
        return {}
    return {
        name: instantiate(loss_cfg, **inject)
        for name, loss_cfg in cfg.items()
    }


if __name__ == "__main__":
    from src.model.powsm.builders import build_powsm

    net = build_powsm(vocab_size=100)
    model = SegmentRecognizeModel(
        net=net,
        optimizer=torch.optim.Adam,
        seg_losses={
            "bce": {
                "_target_": (
                    "src.recipe.segment_recognize"
                    ".layers.bce_boundary.BCEBoundaryLoss"
                ),
                "weight": 1.0,
            },
        },
        pr_losses={
            "ctc": {
                "_target_": (
                    "src.recipe.segment_recognize"
                    ".layers.ctc_recognition"
                    ".CTCRecognitionLoss"
                ),
                "weight": 1.0,
            },
        },
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SegmentRecognizeModel: {n_params} params")
    print(f"  seg_losses: {list(model.seg_losses.keys())}")
    print(f"  pr_losses: {list(model.pr_losses.keys())}")
