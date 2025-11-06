"""Wav2Vec2Phoneme model implementation using Hugging Face Transformers.
Functionalities:
1. TODO(shikhar): Model fine-tuning with ctc loss
2. Encoder output extraction via encode() method.
3. TODO(shikhar): CTC-Decoding via decode() method.
4. TODO(shikhar): Forced alignment via forced_align() method.

This file supports the following pretrained models:
- facebook/wav2vec2-lv-60-espeak-cv-ft
- facebook/wav2vec2-xlsr-53-espeak-cv-ft
- ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns

"facebook" models use phonemizer which needs espeak-ng
Build espeak-ng following https://github.com/espeak-ng/espeak-ng/blob/master/docs/building.md
Then export the following paths:
export PHONEMIZER_ESPEAK_LIBRARY="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/src/.libs/libespeak-ng.so.1.1.51"
export ESPEAK_DATA_PATH="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/espeak-ng-data"
Here the prefix is the path passed to ./configure --prefix=/usr during build.
Both of these exports are necessary.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import numpy as np


def preprocess_inputs_wav2vec2(
    preprocessor: Wav2Vec2Processor,
    speech: List[torch.Tensor] | torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare batched input for Wav2Vec2 model."""
    if isinstance(speech, torch.Tensor):
        speech = speech if speech.ndim == 1 else list(speech)
    batch = [x.detach().cpu().float().numpy().squeeze() for x in speech]

    inputs = preprocessor(
        batch,
        sampling_rate=preprocessor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return {
        "input_values": inputs.input_values.to(device),
        "attention_mask": inputs.attention_mask.to(device),
    }


class Wav2Vec2PhonemeModel(nn.Module):

    def __init__(self, hf_repo: str):
        """
        Args:
            hf_repo: one of the following pretrained models
                facebook/wav2vec2-lv-60-espeak-cv-ft
                facebook/wav2vec2-xlsr-53-espeak-cv-ft
                ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns
        """
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(hf_repo)
        self.model = Wav2Vec2ForCTC.from_pretrained(hf_repo)
        self.model_stride = np.prod(self.model.config.conv_stride)
        print(f"Model stride: {self.model_stride}")
        self.encoder_dim = self.model.config.output_hidden_size
        self.vocab_size = self.model.config.vocab_size

    def forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass compatible with PowsmModel interface"""
        # TODO(shikhar): Implement training with CTC loss
        encoder_out, encoder_out_lens = self.encode(inputs)
        return encoder_out, encoder_out_lens

    def encode(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder"""
        model_out = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        encoder_out = model_out.hidden_states[-1]
        encoder_out_lens = self.model._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(-1)
        )
        return encoder_out, encoder_out_lens

    def output_size(self) -> int:
        """Get output dimension"""
        return self.encoder_dim


def build_wav2vec2phoneme(
    hf_repo: str = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
):
    """Build Wav2Vec2Phoneme model

    Args:
        hf_repo: HuggingFace repository ID

    Returns:
        Wav2Vec2Phoneme model
    """
    model = Wav2Vec2PhonemeModel(hf_repo=hf_repo)
    logging.info(f"Wav2Vec2Phoneme model loaded from {hf_repo}")
    logging.info(f"Model vocab size: {model.vocab_size}")
    return model


if __name__ == "__main__":
    # Example usage
    model = build_wav2vec2phoneme("facebook/wav2vec2-lv-60-espeak-cv-ft")
    dummy_speech = [
        torch.randn(16000),
        torch.randn(8000),
    ]  # Batch of 2 samples, 1 sec, 0.5 sec at 16kHz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = preprocess_inputs_wav2vec2(model.processor, dummy_speech, device=device)
    encoder_out, encoder_out_lens = model.encode(inputs)
    print(f"Encoder output shape: {encoder_out.shape}")
    print(f"Encoder output lengths: {encoder_out_lens}")
