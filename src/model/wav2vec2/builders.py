from src.model.wav2vec2.tokenizer import Wav2Vec2Tokenizer
from src.model.wav2vec2.wav2vec2_model import Wav2Vec2Model
from src.model.wav2vec2.wav2vec2_inference import Wav2Vec2Inference
from src.utils import RankedLogger
from typing import Optional

log = RankedLogger(__name__, rank_zero_only=True)


def build_wav2vec2_tokenizer(
    hf_repo: str = "facebook/mms-300m",
):
    """Build Wav2Vec2 tokenizer

    Args:
        hf_repo: HuggingFace repository ID

    Returns:
        Wav2Vec2 tokenizer
    """
    tokenizer = Wav2Vec2Tokenizer(hf_repo=hf_repo)
    log.info(f"Wav2Vec2 tokenizer loaded from {hf_repo}")
    return tokenizer


def build_wav2vec2_model(
    hf_repo: str = "facebook/mms-300m",
    output_vocabsz: Optional[int] = None,
    blank_id: int = 0,
    freeze_encoder: bool = False,
):
    """Build Wav2Vec2 model

    Args:
        hf_repo: HuggingFace repository ID
        output_vocabsz: Optional output vocabulary size
        blank_id: Blank token ID for CTC
        freeze_encoder: Whether to freeze the encoder layers

    Returns:
        Wav2Vec2 model
    """
    model = Wav2Vec2Model(
        hf_repo=hf_repo,
        output_vocabsz=output_vocabsz,
        blank_id=blank_id,
        freeze_encoder=freeze_encoder,
    )
    log.info(f"Wav2Vec2 model loaded from {hf_repo}")
    log.info(f"Model vocab size: {model.vocab_size}")
    return model


def build_wav2vec2_inference(
    hf_repo: str = "facebook/mms-300m",
    device: str = "cpu",
):
    """Build Wav2Vec2 inference module

    Returns:
        Wav2Vec2 inference module
    """
    model = build_wav2vec2_model(hf_repo=hf_repo)
    tokenizer = build_wav2vec2_tokenizer(hf_repo=hf_repo)
    inference_module = Wav2Vec2Inference(model, tokenizer, device=device)
    log.info("Wav2Vec2 inference module built")
    return inference_module


if __name__ == "__main__":
    # python -m src.model.wav2vec2.builders
    import torch

    model = build_wav2vec2_model(hf_repo="facebook/mms-300m")
    wav = torch.randn(1, 16000 * 5)
    inputs = {"speech": wav, "speech_lengths": torch.tensor([wav.shape[1]])}
    with torch.no_grad():
        outputs = model.encode(**inputs)
    print(outputs)
    print(f"Output shape: {outputs[0].shape}, Lengths shape: {outputs[1].shape}")
