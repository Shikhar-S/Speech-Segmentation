# python -m src.model.powsm.powsm_vr_model
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchaudio
from src.model.powsm.powsm_model import build_powsm
from src.model.powsm.powsm_ctc_model import build_powsm_ctc
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class PowsmVariablerateModel(nn.Module):
    def __init__(self, points_by_frames, powsm_model):
        super().__init__()
        self.powsm_model = powsm_model
        self.points_by_frames_ = points_by_frames
        self.sampling_ratio = powsm_model.points_by_frames() / points_by_frames
        # >1 means upsample, <1 means downsample
        encoder_size = powsm_model.encoder_output_size()
        log.info(
            f"Building PowsmVariablerateModel with points_by_frames={points_by_frames}, "
            f"sampling_ratio={self.sampling_ratio:.3f}, encoder_size={encoder_size}"
        )
        self.sampling_net = nn.Sequential(
            nn.Linear(encoder_size, int(encoder_size * self.sampling_ratio)),
            nn.LayerNorm(int(encoder_size * self.sampling_ratio)),
        )
        self.sampling_rate = self.powsm_model.sampling_rate

    def encoder_output_size(self):
        return self.powsm_model.encoder_output_size()

    def ctc_logits(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get CTC logits from encoder output

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - CTC logits: (Batch, Length, Vocab)
                - Encoder output lengths: (Batch,)
        """
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        logits = self.powsm_model.ctc.ctc_lo(encoder_out)
        return logits, encoder_out_lens

    def points_by_frames(self):
        return self.points_by_frames_

    def encode(self, speech, speech_lengths):
        feat, featlen = self.powsm_model.encode(speech, speech_lengths)
        feat = self.sampling_net(feat)
        feat = feat.view(
            feat.size(0), -1, int(feat.size(2) / self.sampling_ratio)
        ).contiguous()
        featlen = (featlen.float() * self.sampling_ratio).long()
        return feat, featlen

    @torch.no_grad()
    def forced_align(self, speech, speech_lengths, text, text_lengths, utt_id=None):
        """Calculate frame-wise alignment from CTC probabilities.
        Only works with batch size 1.

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            text: (Batch, Length)
            text_lengths: (Batch,)
            utt_id: Optional[str], utterance identifier for logging
        Returns:
            Tuple(tensor, tensor):
                - Label for each time step in the alignment path computed
                using forced alignment.
                - Log probability scores of the labels for each time
                step.
        """
        assert (
            self.powsm_model.ctc is not None
        ), "CTC is not used in this model. Cannot compute forced alignment."
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
        )
        batch_size = speech.shape[0]
        assert batch_size == 1, "Forced alignment needs batch size 1."

        # -1 is used as padding index in collate fn
        text = torch.where(text == -1, self.powsm_model.ignore_id, text)
        text = text[:, : text_lengths.max()]  # for data-parallel
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        log_probs = self.powsm_model.ctc.log_softmax(encoder_out)  # (B, Tmax, odim)
        assert log_probs.size(0) == 1, "Forced alignment needs batch size 1"
        assert not (text == self.blank_id).any(), "Target has blank tokens."
        if text_lengths.item() > encoder_out_lens.item():
            log.error(
                f"Target length {text_lengths.item()} is longer than "
                f"encoder output length {encoder_out_lens.item()}."
                f"Utterance id is :{utt_id}"
            )
        align_label, align_prob = torchaudio.functional.forced_align(
            log_probs, text, encoder_out_lens, text_lengths, blank=self.blank_id
        )
        return align_label, align_prob

    def get_blank_id(self):
        return self.powsm_model.get_blank_id()


def build_powsm_vr(
    work_dir: str,
    *args,
    hf_repo: Optional[str] = "espnet/powsm",
    force: bool = False,
    config_file: Optional[str] = None,
    model_file: Optional[str] = None,
    stats_file: Optional[str] = None,
    architecture: str = "encdec",
    **kwargs,
):
    if architecture == "enc":
        builder = build_powsm_ctc
    elif architecture == "encdec":
        builder = build_powsm
    model = builder(
        *args,
        work_dir=work_dir,
        hf_repo=hf_repo,
        force=force,
        config_file=config_file,
        model_file=model_file,
        stats_file=stats_file,
        **kwargs,
    )
    log.info(
        f"Wrapping Powsm model into PowsmVariablerateModel with "
        f"points_by_frames=320, architecture={architecture}"
    )
    return PowsmVariablerateModel(points_by_frames=320, powsm_model=model)


if __name__ == "__main__":
    # model = build_powsm_vr(work_dir="exp/cache/powsm_ctc", architecture="enc")
    model = build_powsm_vr(
        work_dir="exp/cache/powsm", hf_repo="espnet/powsm", architecture="encdec"
    )

    print(model)
