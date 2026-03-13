"""Forced alignment model module.

Usage:
    python -m src.recipe.forced_alignment.model_module
"""

import pyarrow.parquet as pq  # before torch
from typing import Any, Dict, Tuple, Optional, List

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from lightning.pytorch.utilities import grad_norm

from src.recipe.forced_alignment.forced_alignment_loss import ForcedAlignmentLoss
from src.recipe.forced_alignment.inference import ForcedAlignmentInference
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


class ForcedAlignmentModel(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.evaluator = SegmentationEvaluator(tolerance_ms=20)
        self.encoder_dim = self.net.encoder_output_size()
        self.criterion = ForcedAlignmentLoss()

        self.test_data = {}
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return CTC logits and their lengths."""
        return self.net.ctc_logits(x, x_lengths)

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Shared step used by train/val/test."""
        speech = batch["speech"]
        lengths = batch["speech_length"]
        target = batch["target"]
        target_start = batch["target_start"]
        target_end = batch["target_end"]
        target_len = batch["target_length"]

        target_start_idx, target_end_idx = convert_pointstamps_to_frame_indices(
            target_start, target_end, self.net.points_by_frames()
        )

        logits, logit_len = self(speech, lengths)
        fa_loss = self.criterion(
            logits, logit_len, target, target_start_idx, target_end_idx, target_len
        )
        return {
            "speech": speech,
            "speech_length": lengths,
            "loss": fa_loss["loss"],
            "accuracy": fa_loss["accuracy"],
            "target": target,
            "target_start": target_start,
            "target_end": target_end,
            "target_length": batch["target_length"],
            "logits": logits.detach(),
        }

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
        self.log(
            "train/accuracy",
            out["accuracy"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return out["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model_step(batch)
        self.val_loss(out["loss"].detach())
        self.log(
            "val/accuracy", out["accuracy"], on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

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
        accuracy = out["accuracy"]

        self.log("test/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        pred_alignments = self._align(
            out["speech"],
            out["speech_length"],
            out["target"],
            out["target_length"],
        )
        KEY_PREFIX = f"batch_{batch_idx}_utt"
        pred_dict = {f"{KEY_PREFIX}{i}": ali for i, ali in enumerate(pred_alignments)}
        gt_dict = self._build_gt_alignments(out, key_prefix=KEY_PREFIX)
        # store for epoch
        if self.test_data:
            self.test_data["predictions"].update(pred_dict)
            self.test_data["ground_truth"].update(gt_dict)
        else:
            self.test_data = {"predictions": pred_dict, "ground_truth": gt_dict}

    def on_test_start(self):
        self.test_data = {}  # clear

    def on_test_epoch_end(self):
        fa_results = self.evaluator.evaluate_batch(
            self.test_data["predictions"], self.test_data["ground_truth"]
        )
        # Relevant metrics
        LOG_METRICS = [
            "f1",
            "precision",
            "recall",
            "pbe_median",
            "start_err_median",
            "end_err_median",
            "dur_err_median",
            "pred_dur_mean",
            "gt_dur_mean",
        ]
        for metric in LOG_METRICS:
            self.log(f"{metric}", self.evaluator._get_metric(fa_results, metric, 0.0))

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
            sp, txt, splen_t, txtlen_t = ForcedAlignmentInference.prepare_inputs(
                sp, splen, txt, txtlen, device=self.device
            )
            align_label, _ = self.net.forced_align(sp, splen_t, txt, txtlen_t)
            labels = align_label.squeeze(0).detach().cpu().tolist()
            alignment_result = ForcedAlignmentInference.post_process_alignments(
                self.net,
                labels,
            )
            predicted_alignments.append(alignment_result)
        return predicted_alignments

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
    from pathlib import Path
    from src.model.powsm.powsm_model import build_powsm
    from src.data.buckeye.common_datamodule import BuckeyeDataModule
    from src.model.powsm.token_id_converter import build_powsm_tokenizer
    from src.model.wav2vec2phoneme.builders import (
        build_wav2vec2phoneme_model,
        build_wav2vec2phoneme_tokenizer,
    )

    MODEL = "w2v2ph"
    # MODEL = "powsm"

    data_dir = "/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/buckeye_cache"
    buckeye_root = (
        "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/"
        "ipapack_plus/s2t1/dump/raw/test_buckeye/buckeye"
    )
    train_meta = Path(data_dir) / "train_metadata.json"
    val_meta = Path(data_dir) / "val_metadata.json"
    test_meta = Path(data_dir) / "test_metadata.json"

    if MODEL == "powsm":
        tokenizer = build_powsm_tokenizer(
            work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
            hf_repo="espnet/powsm",
        )
        net = build_powsm(
            work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
            hf_repo="espnet/powsm",
        )
    elif MODEL == "w2v2ph":

        tokenizer = build_wav2vec2phoneme_tokenizer(
            hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
        )
        net = build_wav2vec2phoneme_model(
            hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
        )

    model = ForcedAlignmentModel(
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    # print(model.net.points_by_frames(), "points by frame ratio")

    data_module = BuckeyeDataModule(
        buckeye_root="/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/buckeye",
        local_cache_path="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/buckeye_cache",
        model_tokenizer=tokenizer,
        batch_size=2,
        num_workers=1,
    )

    data_module.setup()
    # print("Model step sanity check...")
    # test_batch = next(iter(data_module.test_dataloader()))
    # preds = model.predict_step(test_batch, batch_idx=0)

    # for utt_idx, alignment in enumerate(preds):
    #     print(f"Utterance {utt_idx}:")
    #     for seg in alignment:
    #         print(seg.start, seg.end, seg.label)
    #         print(seg.start, seg.end)
    #         print("---")

    # print("Model predict step successful!")

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
