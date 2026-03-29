"""Joint phone recognition + segmentation model module.

Trains a shared encoder on both CTC phone recognition and BCE boundary
detection simultaneously.  PR loss is computed on every batch; seg loss
is computed only when timestamp annotations are present.

Usage:
    python -m src.recipe.joint.model_module
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from lightning.pytorch.utilities import grad_norm

from src.recipe.segmentation.segmentation_loss import (
    BoundaryLoss,
    SegmentationLoss,
)
from src.recipe.segmentation.model_module import (
    convert_pointstamps_to_frame_indices,
)
from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)
from src.recipe.segmentation.inference import _boundary_flags_to_units


class JointPRSegModel(LightningModule):
    """Joint phone recognition + segmentation Lightning module.

    Shares a single encoder between CTC-based phone recognition and
    BCE-based boundary detection.  Each training batch carries a
    ``task`` key ("pr" or "seg").  PR loss is always computed; seg
    loss is computed only when ``target_start`` is present.

    Args:
        net: Shared encoder module (XeusPRModel or PowSM).
        optimizer: Partial optimizer constructor.
        scheduler: Optional LR scheduler constructor.
        bce_weight: Weight of BCE boundary loss vs FA loss for seg
            task (1.0 = pure BCE, 0.0 = pure FA).
        pos_weight: Positive class weight for BCEWithLogitsLoss.
        resolution: Temporal upsample factor for seg head.
        audio_sr: Audio sample rate for timestamp conversions.
        seg_loss_weight: Multiplier for segmentation loss.
        pr_loss_weight: Multiplier for phone recognition loss.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        bce_weight: float = 1.0,
        pos_weight: float = 1.0,
        resolution: int = 1,
        audio_sr: int = 16000,
        seg_loss_weight: float = 1.0,
        pr_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.audio_sr = audio_sr
        self.bce_weight = bce_weight
        self.seg_loss_weight = seg_loss_weight
        self.pr_loss_weight = pr_loss_weight

        # Segmentation heads
        dim = self.net.encoder_output_size()
        self.resolution = resolution
        self.upsample = nn.Linear(dim, dim * resolution)
        if bce_weight > 0:
            self.boundary_head = nn.Linear(dim, 1)
            self.boundary_criterion = BoundaryLoss(pos_weight=pos_weight)
        if bce_weight < 1.0:
            self.seg_criterion = SegmentationLoss()

        self.evaluator = SegmentationEvaluator(tolerance_ms=20)

        # Metrics
        self.train_pr_loss = MeanMetric()
        self.train_seg_loss = MeanMetric()
        self.val_seg_loss = MeanMetric()
        self.val_pr_loss = MeanMetric()
        self.val_seg_loss_best = MinMetric()

    @property
    def effective_pbf(self) -> float:
        """Points-by-frames after upsampling."""
        return self.net.points_by_frames() / self.resolution

    # ── Encoder helpers ──────────────────────────────────────────────

    def _encode(
        self, speech: torch.Tensor, lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode speech and upsample for segmentation."""
        features, out_lens = self.net.encode(speech, lengths)
        if isinstance(features, tuple):
            features = features[0]
        return self._upsample_features(features, out_lens)

    def _upsample_features(
        self, features: torch.Tensor, lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = features.shape
        R = self.resolution
        features = (
            self.upsample(features).view(B, T, R, D).reshape(B, T * R, D)
        )
        return features, lengths * R

    # ── PR loss ──────────────────────────────────────────────────────

    def _pr_loss(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """CTC phone recognition loss from encoder features."""
        return self.net._calc_ctc_loss(
            features, feature_lens, text, text_lengths, **kwargs,
        )

    # ── Seg loss ─────────────────────────────────────────────────────

    def _seg_bce_loss(
        self,
        features: torch.Tensor,
        logit_len: torch.Tensor,
        target_start_idx: torch.Tensor,
        target_len: torch.Tensor,
    ) -> Dict:
        """BCE boundary detection loss."""
        boundary_logits = self.boundary_head(features).squeeze(-1)
        out = self.boundary_criterion(
            boundary_logits, logit_len, target_start_idx, target_len,
        )
        out["boundary_logits"] = boundary_logits
        return out

    def _seg_fa_loss(
        self,
        logits: torch.Tensor,
        logit_len: torch.Tensor,
        target: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        t_len: torch.Tensor,
    ) -> Dict:
        """Frame-wise CTC alignment loss for segmentation."""
        return self.seg_criterion(
            logits, logit_len, target, t_start, t_end, t_len,
        )

    def _boundary_metrics(
        self,
        boundary_logits: torch.Tensor,
        logit_len: torch.Tensor,
        target_start_idx: torch.Tensor,
        target_len: torch.Tensor,
    ) -> Dict[str, float]:
        """Boundary-level P/R/F1/Rval from BCE logits."""
        pbf = self.effective_pbf
        sr = self.audio_sr
        preds_dict, gt_dict = {}, {}
        for b in range(boundary_logits.shape[0]):
            vlen = int(logit_len[b])
            flags = (boundary_logits[b, :vlen] > 0).tolist()
            pred_units = _boundary_flags_to_units(flags, vlen, pbf, sr)
            n = int(target_len[b])
            starts = target_start_idx[b, :n].tolist()
            gt_units = [
                SegmentationUnit(
                    start=starts[i] * pbf / sr,
                    end=(starts[i + 1] if i + 1 < n else vlen)
                    * pbf / sr,
                    label=0,
                )
                for i in range(n)
            ]
            preds_dict[str(b)] = pred_units
            gt_dict[str(b)] = gt_units
        results = self.evaluator.evaluate_batch(preds_dict, gt_dict)
        return {
            k: self.evaluator._get_metric(results, k, 0.0)
            for k in ("precision", "recall", "f1", "rval")
        }

    # ── Training / validation steps ──────────────────────────────────

    def _compute_seg_loss(
        self,
        features: torch.Tensor,
        logit_len: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute segmentation loss and metrics from features."""
        target_start_idx, target_end_idx = (
            convert_pointstamps_to_frame_indices(
                batch["target_start"],
                batch["target_end"],
                self.effective_pbf,
            )
        )
        target_len = batch["text_length"]

        seg_loss = torch.tensor(0.0, device=features.device)
        bnd_metrics = {}

        if self.bce_weight > 0:
            bce_out = self._seg_bce_loss(
                features, logit_len, target_start_idx, target_len,
            )
            seg_loss = seg_loss + self.bce_weight * bce_out["loss"]
            with torch.no_grad():
                bnd_metrics = self._boundary_metrics(
                    bce_out["boundary_logits"].detach(),
                    logit_len, target_start_idx, target_len,
                )

        if self.bce_weight < 1.0:
            logits = self.net.ctc.ctc_lo(features)
            fa_out = self._seg_fa_loss(
                logits, logit_len, batch["text"],
                target_start_idx, target_end_idx, target_len,
            )
            seg_loss = seg_loss + (1 - self.bce_weight) * fa_out["loss"]

        return seg_loss, bnd_metrics

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int,
    ) -> torch.Tensor:
        """Dispatch to per-type sub-batches.

        Batch format: ``{"segmentation": sub_batch, "recognition": sub_batch}``
        where either value may be ``None``.
        """
        pr_batch = batch.get("recognition")
        seg_batch = batch.get("segmentation")
        loss = torch.tensor(0.0, device=self.device)

        if pr_batch is not None:
            pr_feat, pr_len = self._encode(
                pr_batch["speech"], pr_batch["speech_length"],
            )
            pr_loss, pr_stats = self._pr_loss(
                pr_feat, pr_len,
                pr_batch["text"], pr_batch["text_length"],
                lang_sym=pr_batch.get("lang_sym"),
            )
            loss = loss + self.pr_loss_weight * pr_loss
            self.train_pr_loss(pr_loss.detach())
            self.log(
                "train/pr_loss", self.train_pr_loss,
                on_step=True, on_epoch=True, prog_bar=True,
            )
            for k, v in pr_stats.items():
                self.log(f"train/{k}", v, on_step=True, on_epoch=True)

        if seg_batch is not None:
            seg_feat, seg_len = self._encode(
                seg_batch["speech"], seg_batch["speech_length"],
            )
            seg_loss, bnd_metrics = self._compute_seg_loss(
                seg_feat, seg_len, seg_batch,
            )
            loss = loss + self.seg_loss_weight * seg_loss
            self.train_seg_loss(seg_loss.detach())
            self.log(
                "train/seg_loss", self.train_seg_loss,
                on_step=True, on_epoch=True, prog_bar=True,
            )
            for k, v in bnd_metrics.items():
                self.log(f"train/{k}", v, on_step=True, on_epoch=True)

        self.log("train/loss", loss.detach(), on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Normalize seg batch keys (target -> text)
        if "text" not in batch and "target" in batch:
            batch["text"] = batch.pop("target")
            batch["text_length"] = batch.pop("target_length")
        B = batch["speech"].shape[0]
        batch.setdefault("lang_sym", ["<eng>"] * B)

        features, logit_len = self._encode(
            batch["speech"], batch["speech_length"],
        )

        # PR metrics
        pr_loss, pr_stats = self._pr_loss(
            features, logit_len,
            batch["text"], batch["text_length"],
            lang_sym=batch.get("lang_sym"),
        )
        self.log("val_seg/pr_loss", pr_loss.detach(),
                 on_step=False, on_epoch=True)

        # Seg metrics
        if batch.get("target_start") is not None:
            seg_loss, bnd_metrics = self._compute_seg_loss(
                features, logit_len, batch,
            )
            self.val_seg_loss(seg_loss.detach())
            self.log("val_seg/seg_loss", seg_loss.detach(),
                     on_step=False, on_epoch=True)
            for k, v in bnd_metrics.items():
                self.log(f"val_seg/{k}", v, on_step=False, on_epoch=True,
                         prog_bar=(k == "rval"))

    def on_validation_epoch_end(self) -> None:
        loss = self.val_seg_loss.compute()
        self.val_seg_loss_best(loss)
        self.log(
            "val_seg/loss_best",
            self.val_seg_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
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
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # python -m src.recipe.joint.model_module
    from src.model.powsm.builders import build_powsm

    net = build_powsm(vocab_size=100)
    model = JointPRSegModel(
        net=net,
        optimizer=torch.optim.Adam,
        bce_weight=1.0,
    )
    print(f"JointPRSegModel created: {sum(p.numel() for p in model.parameters())} params")
    print(f"  Encoder dim: {net.encoder_output_size()}")
    print(f"  Seg heads: boundary_head, upsample")
    print(f"  PR: uses net._calc_ctc_loss()")
