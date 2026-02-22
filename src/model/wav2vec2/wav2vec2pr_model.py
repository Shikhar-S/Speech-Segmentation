"""Wav2Vec2 Phone Recognition Model."""

from typing import Tuple, Union

import torch
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet_import.nets.e2e_asr_common import ErrorCalculator

from src.model.powsm.ctc import CTC
from src.model.wav2vec2.wav2vec2_model import Wav2Vec2Model


class Wav2Vec2PRModel(torch.nn.Module):
    """CTC model for phone recognition using Wav2Vec2 encoder."""

    def __init__(
        self,
        encoder: Wav2Vec2Model,
        ctc: CTC,
        token_list: Union[Tuple, list],
        ignore_id: int = -1,
        sym_blank: str = "<blank>",
        freeze_frontend: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.token_list = list(token_list)
        self.ignore_id = ignore_id
        assert sym_blank in token_list, "Blank symbol must be in token list."
        self.blank_id = token_list.index(sym_blank)
        self.freeze_frontend = freeze_frontend
        self.error_calculator = ErrorCalculator(
            token_list,
            kwargs.get("sym_space", "<space>"),
            sym_blank,
            report_cer=True,
            report_wer=False,
        )

    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        loss_ctc, stats = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )
        loss, stats, weight = force_gatherable(
            (loss_ctc, stats, speech.shape[0]), loss_ctc.device
        )
        return {"loss": loss, "stats": stats, "weight": weight}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.encode(speech, speech_lengths)

    def _calc_ctc_loss(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens):
        ys_pad = torch.where(ys_pad == -1, self.ignore_id, ys_pad)
        ys_pad = ys_pad[:, : ys_pad_lens.max()]
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        stats = {}
        if not self.training:
            with torch.no_grad():
                ys_hat = self.ctc.argmax(encoder_out).data
                stats["cer_ctc"] = self.error_calculator(
                    ys_hat.cpu(), ys_pad.cpu(), is_ctc=True
                )
        return loss_ctc, stats

    def ctc_logits(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        return self.ctc.ctc_lo(encoder_out), encoder_out_lens

    def encoder_output_size(self) -> int:
        return self.encoder.encoder_output_size()

    def get_blank_id(self) -> int:
        return self.blank_id

    def get_frontend(self):
        return self.encoder.model.wav2vec2.feature_extractor

    def get_trainable_parameters(self):
        trainable_params = {"head": [], "encoder": []}
        for n, p in self.named_parameters():
            if n.startswith("ctc"):
                trainable_params["head"].append(p)
            elif n.startswith("encoder.model.wav2vec2.encoder"):
                trainable_params["encoder"].append(p)
            elif n.startswith("encoder.model.wav2vec2.feature"):
                # feature_extractor and feature_projection
                if self.freeze_frontend:
                    p.requires_grad = False
                else:
                    trainable_params["encoder"].append(p)
            else:
                # freeze other parts:
                p.requires_grad = False
        return trainable_params
