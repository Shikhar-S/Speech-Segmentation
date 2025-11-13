from src.model.wav2vec2phoneme.tokenizer import Wav2Vec2PhonemeTokenizer
from src.model.wav2vec2phoneme.wav2vec2phoneme_model import Wav2Vec2PhonemeModel
import logging


def build_wav2vec2phoneme_tokenizer(
    hf_repo: str = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
):
    """Build Wav2Vec2Phoneme tokenizer

    Args:
        hf_repo: HuggingFace repository ID

    Returns:
        Wav2Vec2Phoneme tokenizer
    """
    tokenizer = Wav2Vec2PhonemeTokenizer(hf_repo=hf_repo)
    logging.info(f"Wav2Vec2Phoneme tokenizer loaded from {hf_repo}")
    return tokenizer


def build_wav2vec2phoneme_model(
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
