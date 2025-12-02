"""A wrapper over icefall ZIPA-CRCTC implementation"""

import torch
import torch.nn as nn
from typing import Any, Tuple
import numpy as np

from lhotse.features.kaldi.extractors import Fbank


class ZipaCtcModel(nn.Module):

    def __init__(self, encoder, blank_id, sampling_rate=16000):
        super().__init__()
        self.encoder = encoder
        self.fbank = Fbank()
        self.blank_id = blank_id
        self.sampling_rate = sampling_rate

    @torch.no_grad()
    def points_by_frames(self) -> int:
        raise NotImplementedError("Implement points_by_frames method in zipa ctc")

    def forward(self, inputs) -> Any:
        pass
        # # TODO(shikhar): check with PR
        # # only called in phone recognition recipe when finetuning
        # simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, cr_loss = (
        #     self.encoder(**inputs)
        # )
        # return {
        #     "simple_loss": simple_loss,
        #     "pruned_loss": pruned_loss,
        #     "ctc_loss": ctc_loss,
        #     "attention_decoder_loss": attention_decoder_loss,
        #     "cr_loss": cr_loss,
        # }

    def _extract_feats(self, speech, speech_lengths):
        features = self.fbank.extract_batch(
            speech, lengths=speech_lengths, sampling_rate=self.sampling_rate
        )  # ragged
        feature_lens = torch.tensor([len(feature) for feature in features])
        if isinstance(features, np.ndarray) and features.ndim == 2:
            # bs=1
            features = [features]
        features = [torch.tensor(f, dtype=torch.float32) for f in features]
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        features = features.to(speech.device)
        feature_lens = feature_lens.to(speech.device)
        return features, feature_lens

    def encode(self, speech, speech_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, featlens = self._extract_feats(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.encoder.forward_encoder(feat, featlens)
        return encoder_out, encoder_out_lens

    def ctc_logits(self, speech, speech_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        logits = self.encoder.ctc_output(encoder_out)  # (N, T, C)
        return logits, encoder_out_lens

    def encoder_output_size(self) -> int:
        return self.encoder.encoder_dim

    @torch.no_grad()
    def forced_align(self, speech, speech_lengths, text, text_lengths, utt_id=None):
        raise NotImplementedError("Implement forced_align method in zipa ctc")

    def get_blank_id(self) -> int:
        return self.blank_id
