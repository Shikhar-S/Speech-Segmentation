"""Forced alignment model module.

Usage:
    python -m src.recipe.segmentation.model_module
"""

import pyarrow.parquet as pq  # before torch
from typing import Any, Dict, Tuple, Optional, List

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from lightning.pytorch.utilities import grad_norm

from src.recipe.segmentation.segmentation_loss import BoundaryLoss, SegmentationLoss
from src.recipe.segmentation.inference import (
    SegmentationInference,
    _boundary_flags_to_units,
)
from src.metrics.segmentation_evaluator import SegmentationEvaluator, SegmentationUnit
from src.utils import RankedLogger


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


class SegmentationModel(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        bce_weight: float = 0.0,
        pos_weight: float = 1.0,
        resolution: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.evaluator = SegmentationEvaluator(tolerance_ms=20)
        self.encoder_dim = self.net.encoder_output_size()
        self.criterion = SegmentationLoss()
        self.bce_weight = bce_weight
        assert resolution >= 1 and isinstance(resolution, int), (
            f"resolution must be a positive integer, got {resolution}"
        )
        self.resolution = resolution
        self.upsample = nn.Linear(self.encoder_dim, self.encoder_dim * resolution)
        if bce_weight > 0:
            self.boundary_head = nn.Linear(self.encoder_dim, 1)
            self.boundary_criterion = BoundaryLoss(pos_weight=pos_weight)

        self.test_data = {}
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

    @property
    def effective_pbf(self) -> float:
        """Points per frame after upsampling."""
        return self.net.points_by_frames() / self.resolution

    def _upsample_features(
        self, features: torch.Tensor, lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Upsample encoder features by self.resolution.

        Linear projects (B, T, D) → (B, T, D*R), then reshapes to
        (B, T*R, D) so embedding [a, b] becomes [a, a_1, b, b_1].
        """
        B, T, D = features.shape
        R = self.resolution
        # (B, T, D) → (B, T, D*R) → (B, T, R, D) → (B, T*R, D)
        features = self.upsample(features).view(B, T, R, D).reshape(B, T * R, D)
        return features, lengths * R

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return encoder features and their lengths."""
        features, lengths = self.net.encode(x, x_lengths)
        if isinstance(features, tuple):
            features = features[0]
        features, lengths = self._upsample_features(features, lengths)
        return features, lengths

    def _fa_loss(
        self,
        logits: torch.Tensor,
        logit_len: torch.Tensor,
        target: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        t_len: torch.Tensor,
    ) -> Dict:
        """Forced alignment loss from CTC logits."""
        return self.criterion(logits, logit_len, target, t_start, t_end, t_len)

    def _bce_loss(
        self,
        features: torch.Tensor,
        logit_len: torch.Tensor,
        t_start: torch.Tensor,
        t_len: torch.Tensor,
    ) -> Dict:
        """Boundary detection loss from encoder features."""
        boundary_logits = self.boundary_head(features).squeeze(-1)
        out = self.boundary_criterion(boundary_logits, logit_len, t_start, t_len)
        out["boundary_logits"] = boundary_logits
        return out

    def _boundary_metrics(
        self,
        boundary_logits: torch.Tensor,
        logit_len: torch.Tensor,
        target_start_idx: torch.Tensor,
        target_len: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute boundary-level P/R/F1/Rval using SegmentationEvaluator.

        Converts frame predictions and GT to SegmentationUnit segments,
        then delegates to the evaluator for proper boundary matching.
        """
        pbf = self.effective_pbf
        sr = self.net.sampling_rate
        preds_dict, gt_dict = {}, {}
        B = boundary_logits.shape[0]
        for b in range(B):
            vlen = int(logit_len[b])
            # Predicted boundaries: convert frame flags to segments
            flags = (boundary_logits[b, :vlen] > 0).tolist()
            pred_units = _boundary_flags_to_units(flags, vlen, pbf, sr)
            # GT boundaries: build segments from start frame indices
            n_phones = int(target_len[b])
            starts = target_start_idx[b, :n_phones].tolist()
            gt_units = []
            for i in range(n_phones):
                s = starts[i]
                e = starts[i + 1] if i + 1 < n_phones else vlen
                gt_units.append(SegmentationUnit(
                    start=s * pbf / sr, end=e * pbf / sr, label=0,
                ))
            key = str(b)
            preds_dict[key] = pred_units
            gt_dict[key] = gt_units
        results = self.evaluator.evaluate_batch(preds_dict, gt_dict)
        return {
            k: self.evaluator._get_metric(results, k, 0.0)
            for k in ("precision", "recall", "f1", "rval")
        }

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Shared step used by train/val/test."""
        speech = batch["speech"]
        lengths = batch["speech_length"]
        target = batch["target"]
        target_start = batch["target_start"]
        target_end = batch["target_end"]
        target_len = batch["target_length"]

        target_start_idx, target_end_idx = convert_pointstamps_to_frame_indices(
            target_start, target_end, self.effective_pbf,
        )

        # Guard: FA-only mode uses the public ctc_logits() API, which works for
        # all model types (not all nets expose ctc.ctc_lo directly).
        if self.bce_weight == 0.0:
            features, logit_len = self(speech, lengths)
            logits = self.net.ctc.ctc_lo(features)
            fa_out = self._fa_loss(logits, logit_len, target, target_start_idx, target_end_idx, target_len)
            return {
                "speech": speech, "speech_length": lengths,
                "loss": fa_out["loss"], "accuracy": fa_out["accuracy"],
                "fa_loss": fa_out["loss"], "bce_loss": None,
                "target": target, "target_start": target_start,
                "target_end": target_end, "target_length": target_len,
                "logits": logits.detach(),
            }

        # BCE or combined: encode once, apply both heads as needed.
        features, logit_len = self(speech, lengths)
        logits = None if self.bce_weight == 1.0 else self.net.ctc.ctc_lo(features)
        fa_out = {} if logits is None else self._fa_loss(
            logits, logit_len, target, target_start_idx, target_end_idx, target_len
        )
        bce_out = self._bce_loss(features, logit_len, target_start_idx, target_len)

        loss = (1 - self.bce_weight) * fa_out.get("loss", 0) + self.bce_weight * bce_out["loss"]
        result = {
            "speech": speech,
            "speech_length": lengths,
            "loss": loss,
            "accuracy": fa_out.get("accuracy", 0.0),
            "fa_loss": fa_out.get("loss"),
            "bce_loss": bce_out["loss"],
            "target": target,
            "target_start": target_start,
            "target_end": target_end,
            "target_length": target_len,
            "logits": logits.detach() if logits is not None else None,
        }
        with torch.no_grad():
            bnd_metrics = self._boundary_metrics(
                bce_out["boundary_logits"].detach(),
                logit_len, target_start_idx, target_len,
            )
        result.update(bnd_metrics)
        return result

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.test_loss.reset()
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.test_data = {}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        out = self.model_step(batch)
        self.train_loss(out["loss"].detach())
        self.log("train/accuracy", out["accuracy"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        if out["fa_loss"] is not None:
            self.log("train/fa_loss", out["fa_loss"].detach(), on_step=True, on_epoch=True)
        if out["bce_loss"] is not None:
            self.log("train/bce_loss", out["bce_loss"].detach(), on_step=True, on_epoch=True)
        for k in ("precision", "recall", "f1", "rval"):
            if k in out:
                self.log(f"train/{k}", out[k], on_step=True, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model_step(batch)
        self.val_loss(out["loss"].detach())
        self.log(
            "val/accuracy", out["accuracy"], on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for k in ("precision", "recall", "f1", "rval"):
            if k in out:
                self.log(f"val/{k}", out[k], on_step=False, on_epoch=True, prog_bar=(k == "f1"))

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def _build_gt_alignments(
        self, batch: Dict[str, torch.Tensor], key_prefix: str = ""
    ) -> Dict[str, List[SegmentationUnit]]:
        """Build ground truth alignments from target_start/target_end.
        # TODO(shikhar): Move this as a builder to forced alignment metric module.
        """
        target = batch["target"]
        target_start = batch["target_start"]
        target_end = batch["target_end"]
        target_length = batch["target_length"]

        gt = {}
        for b in range(target.size(0)):
            length = int(target_length[b].item())
            starts = target_start[b, :length]
            ends = target_end[b, :length]
            labels = target[b, :length]

            segs: List[SegmentationUnit] = []
            for s, e, lab in zip(starts, ends, labels):
                if s < 0 or e < s:
                    continue
                segs.append(
                    SegmentationUnit(
                        start=float(s.item()) / self.net.sampling_rate,
                        end=float(e.item()) / self.net.sampling_rate,
                        label=int(lab.item()),
                    )
                )
            gt[f"{key_prefix}{b}"] = segs
        return gt

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model_step(batch)
        self.test_loss(out["loss"].detach())
        self.log("test/accuracy", out["accuracy"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        KEY_PREFIX = f"batch_{batch_idx}_utt"
        gt_dict = self._build_gt_alignments(out, key_prefix=KEY_PREFIX)

        # CTC-based eval: FA + greedy (only when CTC head was trained)
        if self.bce_weight < 1.0:
            fa_preds = self._align(
                out["speech"], out["speech_length"],
                out["target"], out["target_length"],
            )
            greedy_preds = self._greedy_align(
                out["speech"], out["speech_length"],
            )
            for mode, preds in [("fa", fa_preds), ("greedy", greedy_preds)]:
                pred_dict = {f"{KEY_PREFIX}{i}": ali for i, ali in enumerate(preds)}
                self.test_data[mode]["predictions"].update(pred_dict)
                self.test_data[mode]["ground_truth"].update(gt_dict)

        # Boundary eval (only when boundary_head was trained)
        if self.bce_weight > 0.0:
            bnd_preds = self._boundary_align(
                out["speech"], out["speech_length"],
            )
            pred_dict = {f"{KEY_PREFIX}{i}": ali for i, ali in enumerate(bnd_preds)}
            self.test_data["boundary"]["predictions"].update(pred_dict)
            self.test_data["boundary"]["ground_truth"].update(gt_dict)

    def on_test_start(self):
        self.test_data = {
            mode: {"predictions": {}, "ground_truth": {}}
            for mode in ("fa", "greedy", "boundary")
        }

    def on_test_epoch_end(self):
        LOG_METRICS = [
            "f1", "precision", "recall", "pbe_median",
            "start_err_median", "end_err_median",
            "dur_err_median", "pred_dur_mean", "gt_dur_mean",
        ]
        for mode in ("fa", "greedy", "boundary"):
            preds = self.test_data[mode]["predictions"]
            if not preds:
                continue
            gt = self.test_data[mode]["ground_truth"]
            results = self.evaluator.evaluate_batch(preds, gt)
            for metric in LOG_METRICS:
                val = self.evaluator._get_metric(results, metric, 0.0)
                self.log(f"{mode}/{metric}", val)

    def _align(
        self, speech, speech_length, text, text_length
    ) -> List[List[SegmentationUnit]]:
        """Get forced alignments for a batch.

        Args:
            speech: (Batch, Length, ...)
            speech_length: (Batch,)
            text: (Batch, Length) tokenized
            text_length: (Batch,)
        Returns:
            List[List[AlignmentResult]]: per-utterance segment list.
        """
        predicted_alignments: List[List[SegmentationUnit]] = []
        for sp, splen, txt, txtlen in zip(speech, speech_length, text, text_length):
            sp, txt, splen_t, txtlen_t = SegmentationInference.prepare_inputs(
                sp, splen, txt, txtlen, device=self.device
            )
            align_label, _ = self.net.forced_align(sp, splen_t, txt, txtlen_t)
            labels = align_label.squeeze(0).detach().cpu().tolist()
            alignment_result = SegmentationInference.post_process_alignments(
                self.net,
                labels,
            )
            predicted_alignments.append(alignment_result)
        return predicted_alignments

    def _greedy_align(
        self, speech, speech_length,
    ) -> List[List[SegmentationUnit]]:
        """Greedy CTC decode for a batch (no text input needed)."""
        return [
            SegmentationInference.greedy_decode(
                self.net, sp, splen, device=self.device,
            )
            for sp, splen in zip(speech, speech_length)
        ]

    @torch.no_grad()
    def _boundary_align(
        self, speech, speech_length,
    ) -> List[List[SegmentationUnit]]:
        """Boundary detection alignment for a batch."""
        pbf = self.effective_pbf
        sr = self.net.sampling_rate
        results: List[List[SegmentationUnit]] = []
        for sp, splen in zip(speech, speech_length):
            sp_b = sp[:int(splen)].unsqueeze(0).to(self.device)
            splen_t = torch.as_tensor([int(splen)], device=self.device)
            features, logit_lens = self(sp_b, splen_t)
            probs = torch.sigmoid(
                self.boundary_head(features).squeeze(-1)
            ).squeeze(0)
            valid_len = int(logit_lens[0])
            flags = (probs[:valid_len] > 0.5).tolist()
            results.append(
                _boundary_flags_to_units(flags, valid_len, pbf, sr)
            )
        return results

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Any:
        speech = batch["speech"]
        speech_length = batch["speech_length"]
        target = batch["target"]
        target_length = batch["target_length"]
        return self._align(speech, speech_length, target, target_length)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
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


if __name__ == "__main__":
    from src.data.segmentation.segmentation_dataset import SegmentationDataModule

    MODEL = "w2v2ph"
    # MODEL = "powsm"
    HF_REPO = "changelinglab/buckeye-segment"

    if MODEL == "powsm":
        from src.model.powsm.powsm_model import build_powsm
        from src.model.powsm.token_id_converter import build_powsm_tokenizer

        tokenizer = build_powsm_tokenizer(
            work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
            hf_repo="espnet/powsm",
        )
        net = build_powsm(
            work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
            hf_repo="espnet/powsm",
        )
    elif MODEL == "w2v2ph":
        from src.model.wav2vec2phoneme.builders import (
            build_wav2vec2phoneme_model,
            build_wav2vec2phoneme_tokenizer,
        )

        tokenizer = build_wav2vec2phoneme_tokenizer(
            hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
        )
        net = build_wav2vec2phoneme_model(
            hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
        )

    model = SegmentationModel(
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    data_module = SegmentationDataModule(
        hf_repo=HF_REPO,
        tokenizer=tokenizer,
        batch_size=2,
        num_workers=1,
    )

    data_module.setup()

    print("Training step sanity check...")
    train_batch = next(iter(data_module.train_dataloader()))
    loss = model.training_step(train_batch, batch_idx=0)
    print(f"Training loss: {loss}")
    print("Model training step successful!")

    print("Validation step sanity check...")
    val_batch = next(iter(data_module.val_dataloader()))
    model.validation_step(val_batch, batch_idx=0)
    print("Model validation step successful!")

    print("Test step sanity check...")
    test_batch = next(iter(data_module.test_dataloader()))
    model.test_step(test_batch, batch_idx=0)
    print("Model test step successful!")
