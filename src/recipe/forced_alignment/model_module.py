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
from src.metrics.forced_alignment import AlignmentEvaluator, ForceAlignedUnit


class ForcedAlignmentModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model"])

        self.net = model
        self.evaluator = AlignmentEvaluator(tolerance_ms=20)
        self.encoder_dim = self.net.encoder_output_size()
        self.criterion = ForcedAlignmentLoss(ignore_index=0)  # TODO: set ignore index

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

        # Convert target start and end to indices based on net's resolution
        frames2points = self.net.frames2points()
        target_start_idx = target_start * frames2points
        target_end_idx = target_end * frames2points  # P * frames/points = F

        logits, logit_len = self(speech, lengths)
        loss = self.criterion(
            logits, target, target_start_idx, target_end_idx, logit_len
        )
        return {
            "speech": speech,
            "speech_length": lengths,
            "loss": loss,
            "target": target,
            "target_start": target_start,
            "target_end": target_end,
            "target_length": batch["target_length"],
            "logits": logits.detach(),
            "logit_lengths": logit_len.detach(),
        }

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.test_loss.reset()
        self.val_loss.reset()
        self.val_loss_best.reset()

    def _fa_metrics(self, output) -> Dict[str, torch.Tensor]:
        """Compute simple token-level accuracy for monitoring."""
        predictions = output["logits"].argmax(dim=-1)
        accuracy = predictions == output["target"]
        mask = output["target_start"] != -1.0
        accuracy = accuracy.masked_select(mask).float().mean()
        return {"accuracy": accuracy}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        out = self.model_step(batch)
        self.train_loss(out["loss"])
        accuracy = self._fa_metrics(out)["accuracy"]

        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return out["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model_step(batch)
        self.val_loss(out["loss"])
        accuracy = self._fa_metrics(out)["accuracy"]

        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
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
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, List[ForceAlignedUnit]]:
        """Build ground truth alignments from target_start/target_end."""
        frames2points = self.net.frames2points()
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

            segs: List[ForceAlignedUnit] = []
            for s, e, lab in zip(starts, ends, labels):
                if s < 0 or e <= s:
                    continue
                segs.append(
                    ForceAlignedUnit(
                        start=float(s.item()) * frames2points / self.net.sampling_rate,
                        end=float(e.item()) * frames2points / self.net.sampling_rate,
                        label=int(lab.item()),
                    )
                )
            gt[f"utt_{b}"] = segs
        return gt

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model_step(batch)
        self.test_loss(out["loss"])
        accuracy = self._fa_metrics(out)["accuracy"]

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
        pred_dict = {f"utt_{i}": ali for i, ali in enumerate(pred_alignments)}
        gt_dict = self._build_gt_alignments(out)

        fa_results = self.evaluator.evaluate_batch(pred_dict, gt_dict)

        # Log a few key metrics if available
        if fa_results:
            self.log(
                "test/fa_f1",
                fa_results.get("mean_f1", 0.0),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/fa_boundary_err_ms",
                fa_results.get("mean_boundary_err_mean", 0.0),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "test/fa_dur_err_ms",
                fa_results.get("mean_dur_err_mean", 0.0),
                on_step=False,
                on_epoch=True,
            )

    def _align(
        self, speech, speech_length, text, text_length
    ) -> List[List[ForceAlignedUnit]]:
        """Get forced alignments for a batch.

        Args:
            speech: (Batch, Length, ...)
            speech_length: (Batch,)
            text: (Batch, Length) tokenized
            text_length: (Batch,)
        Returns:
            List[List[AlignmentResult]]: per-utterance segment list.
        """
        predicted_alignments: List[List[ForceAlignedUnit]] = []
        frames2points = self.net.frames2points()

        for sp, splen, txt, txtlen in zip(speech, speech_length, text, text_length):
            sp = sp[: int(splen)].unsqueeze(0)  # (1, T_s)
            txt = txt[: int(txtlen)].unsqueeze(0)  # (1, T_t)
            splen_t = torch.as_tensor([int(splen)], device=sp.device)
            txtlen_t = torch.as_tensor([int(txtlen)], device=sp.device)

            align_label, _ = self.net.forced_align(sp, splen_t, txt, txtlen_t)
            labels = align_label.squeeze(0).detach().cpu().tolist()
            if not labels:
                predicted_alignments.append([])
                continue

            alignment_result: List[ForceAlignedUnit] = []
            start_idx = 0
            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1]:
                    alignment_result.append(
                        ForceAlignedUnit(
                            start=start_idx * frames2points / self.net.sampling_rate,
                            end=i * frames2points / self.net.sampling_rate,
                            label=labels[i - 1],
                        )
                    )
                    start_idx = i

            alignment_result.append(
                ForceAlignedUnit(
                    start=start_idx * frames2points / self.net.sampling_rate,
                    end=len(labels) * frames2points / self.net.sampling_rate,
                    label=labels[-1],
                )
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
    from src.data.buckeye.forced_alignment import BuckeyeAlignment
    from src.model.powsm.token_id_converter import build_powsm_tokenizer
    from src.model.wav2vec2phoneme.builders import (
        build_wav2vec2phoneme_model,
        build_wav2vec2phoneme_tokenizer,
    )

    MODEL = "w2v2ph"

    data_dir = "/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/buckeye_cache"
    buckeye_root = (
        "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/"
        "ipapack_plus/s2t1/dump/raw/test_buckeye/buckeye"
    )
    train_meta = Path(data_dir) / "train_metadata.json"
    val_meta = Path(data_dir) / "val_metadata.json"
    test_meta = Path(data_dir) / "test_metadata.json"

    if MODEL == "powsm":
        model_tokenizer = build_powsm_tokenizer(
            work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
            hf_repo="espnet/powsm",
        )
        model = build_powsm(
            work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
            hf_repo="espnet/powsm",
        )
    elif MODEL == "w2v2ph":

        model_tokenizer = build_wav2vec2phoneme_tokenizer(
            hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
        )
        model = build_wav2vec2phoneme_model(
            hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
        )

    model = ForcedAlignmentModel(
        model=model,
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    data_module = BuckeyeAlignment(
        buckeye_root=buckeye_root,
        train_metadata=str(train_meta),
        val_metadata=str(val_meta),
        test_metadata=str(test_meta),
        model_tokenizer=model_tokenizer,
        batch_size=2,
        num_workers=1,
    )

    data_module.setup()
    print("Model step sanity check...")
    test_batch = next(iter(data_module.test_dataloader()))
    preds = model.predict_step(test_batch, batch_idx=0)

    for utt_idx, alignment in enumerate(preds):
        print(f"Utterance {utt_idx}:")
        for seg in alignment:
            print(seg.start, seg.end, seg.label)
            print(seg.start, seg.end)
            print("---")

    print("Model predict step successful!")
