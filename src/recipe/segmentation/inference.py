"""Forced alignment inference module.

Usage:
    python -m src.recipe.segmentation.inference
"""

import pyarrow.parquet as pq  # before torch
from typing import List

import torch
import torch.nn as nn
from src.metrics.segmentation_evaluator import SegmentationUnit, SegmentationEvaluator
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SegmentationInference:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.net = model
        self.device = device
        self.net.to(self.device)

    @staticmethod
    def post_process_alignments(net, labels) -> List[SegmentationUnit]:
        """Post-process frame-level labels into forced aligned units.
            Removes blanks.
        Args:
            net: The model used for alignment.
            labels: List of labels where labels[i] corresponds to the label for frame i.
        Returns:
            List[ForceAlignedUnit]: List of forced aligned units.
        """
        points_by_frames = net.points_by_frames()
        alignment_result: List[SegmentationUnit] = []
        if not labels:
            return alignment_result

        start_idx = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                if labels[i - 1] != net.get_blank_id():
                    # only add non-blank labels
                    alignment_result.append(
                        SegmentationUnit(
                            start=start_idx * points_by_frames / net.sampling_rate,
                            end=i * points_by_frames / net.sampling_rate,
                            label=labels[i - 1],
                        )
                    )
                start_idx = i  # do irrespective

        if labels[-1] != net.get_blank_id():
            alignment_result.append(
                SegmentationUnit(
                    start=start_idx * points_by_frames / net.sampling_rate,
                    end=len(labels) * points_by_frames / net.sampling_rate,
                    label=labels[-1],
                )
            )
        return alignment_result

    @staticmethod
    def prepare_inputs(
        speech: torch.Tensor,
        speech_length: torch.Tensor,
        target: torch.Tensor,
        target_length: torch.Tensor,
        device: str = "cpu",
        utt_id=None,
    ) -> dict:
        """Prepare inputs for forced alignment inference.

        Args:
            speech: (Length)
            speech_length: int
            target: (Length) tokenized
            target_length: int
            device: device to move tensors to
            utt_id: identifier for the utterance
        Returns:
            dict: Dictionary containing prepared inputs.
        """
        sp = speech[: int(speech_length)].unsqueeze(0).to(device)  # (1, T_s)
        txt = target[: int(target_length)].unsqueeze(0).to(device)  # (1, T_t)
        splen_t = torch.as_tensor([int(speech_length)], device=sp.device)
        txtlen_t = torch.as_tensor([int(target_length)], device=sp.device)
        log.info(
            f"Prepared inputs: uttid:{utt_id}, speech length: {splen_t.item()}, target length: {txtlen_t.item()}"
        )
        return sp, txt, splen_t, txtlen_t

    @torch.no_grad()
    def __call__(
        self,
        speech,
        speech_length,
        target,
        target_length,
        utt_id,
        *args,
        **kwargs,
    ) -> List[List[SegmentationUnit]]:
        """Get forced alignments for a single utterance.

        Args:
            speech: (Length)
            speech_length: int
            target: (Length) tokenized
            target_length: int
            utt_id: str, identifier for the utterance
        Returns:
            List[ForceAlignedUnit]: list of forced aligned units for the utterance.
        """
        sp, txt, splen_t, txtlen_t = SegmentationInference.prepare_inputs(
            speech,
            speech_length,
            target,
            target_length,
            device=self.device,
            utt_id=utt_id,
        )
        align_label, _ = self.net.forced_align(
            sp, splen_t, txt, txtlen_t, utt_id=utt_id
        )
        labels = align_label.squeeze(0).detach().cpu().tolist()
        alignment_result = SegmentationInference.post_process_alignments(
            self.net, labels
        )
        return alignment_result


if __name__ == "__main__":
    from src.model.wav2vec2phoneme.builders import (
        build_wav2vec2phoneme_model,
        build_wav2vec2phoneme_tokenizer,
    )
    from src.model.powsm.powsm_model import build_powsm
    from src.model.powsm.token_id_converter import build_powsm_tokenizer

    MODEL = "powsm"
    # MODEL = "w2v2ph"
    if MODEL == "powsm":
        model = build_powsm(
            work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
            hf_repo="espnet/powsm",
        )
        model_tokenizer = build_powsm_tokenizer(
            work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
            hf_repo="espnet/powsm",
        )
    elif MODEL == "w2v2ph":
        model = build_wav2vec2phoneme_model(
            "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
        )
        model_tokenizer = build_wav2vec2phoneme_tokenizer(
            "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
        )
    print("Point to frame for model:", model.points_by_frames())
    inference_module = SegmentationInference(model=model)

    # Dummy input
    # speech = torch.randn(16000 * 5)  # 5 seconds of audio at 16kHz
    # speech_length = torch.tensor(16000 * 5)
    # target = torch.randint(0, 100, (50,))  # random tokenized target
    # target_length = torch.tensor(50)
    from src.data.buckeye.common_datamodule import BuckeyeDataModule

    # Create dataloaders
    data_module = BuckeyeDataModule(
        buckeye_root="/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/buckeye",
        local_cache_path="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/buckeye_cache",
        model_tokenizer=model_tokenizer,
        batch_size=1,
        num_workers=1,
    )
    data_module.setup()

    # train_loader = data_module.train_dataloader()
    # val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    for batch in test_loader:
        speech = batch["speech"].squeeze(0)
        speech_length = batch["speech_length"].squeeze(0)
        target = batch["target"].squeeze(0)
        target_length = batch["target_length"].squeeze(0)
        target_start = batch["target_start"].squeeze(0).tolist()
        target_end = batch["target_end"].squeeze(0).tolist()
        # print(target_length)
        alignment = inference_module(
            speech=speech,
            speech_length=speech_length,
            target=target,
            target_length=target_length,
        )
        print(target)
        print(batch["target_text"])

        for j, unit in enumerate(alignment):
            gt_start = target_start[j] / 16000 if j < len(target_start) else "N/A"
            gt_end = target_end[j] / 16000 if j < len(target_end) else "N/A"
            print(
                f"  {unit} | {model_tokenizer.ids2tokens([unit.label])} | start: {gt_start} | end: {gt_end}"
            )
        print()
        evaluator = SegmentationEvaluator()
        metrics = evaluator.evaluate_batch(
            predictions={"identifier": alignment},
            ground_truth={
                "identifier": [
                    SegmentationUnit(start=ts / 16000, end=te / 16000, label=tl)
                    for ts, te, tl in zip(target_start, target_end, target.tolist())
                ]
            },
            skip_symbols={model_tokenizer.unk_symbol},
        )
        evaluator.pretty_print(metrics, verbosity=2)
        break
        # ignore id problems in training
        # filter blank/pad during FA evals
