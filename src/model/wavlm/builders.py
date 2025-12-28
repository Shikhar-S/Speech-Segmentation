"""Builders for WavLM model in PhoneBench.

Usage:
    from src.model.wavlm.builders import build_wavlm_model

    model = build_wavlm_model(
        hf_repo="microsoft/wavlm-base",
        output_vocabsz=50,  # IPA vocab size for FA
    )
"""

from typing import Optional

from src.model.wavlm.wavlm_model import WavLMEncoderModel
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_wavlm_model(
    hf_repo: str = "microsoft/wavlm-base",
    output_vocabsz: Optional[int] = None,
    blank_id: int = 0,
    freeze_encoder: bool = True,
    encoder_layer: int = -1,
) -> WavLMEncoderModel:
    """Build WavLM encoder model for PhoneBench.

    Args:
        hf_repo: HuggingFace model ID (e.g., "microsoft/wavlm-base").
        output_vocabsz: If set, creates a CTC head with this vocab size.
            Required for forced alignment. Use IPATokenizer.vocab_size() for IPA.
        blank_id: Blank token ID for CTC (default 0, matching IPATokenizer).
        freeze_encoder: Whether to freeze encoder weights (default True).
        encoder_layer: Which encoder layer to use (-1 = last).

    Returns:
        WavLMEncoderModel instance.
    """
    model = WavLMEncoderModel(
        hf_repo=hf_repo,
        output_vocabsz=output_vocabsz,
        blank_id=blank_id,
        freeze_encoder=freeze_encoder,
        encoder_layer=encoder_layer,
    )
    log.info(f"WavLM model loaded from {hf_repo}")
    log.info(f"Encoder dim: {model.encoder_output_size()}")
    if output_vocabsz is not None:
        log.info(f"CTC head vocab size: {output_vocabsz}")
    return model
