"""Stage 2 of the decoupled English cascade: forced-align recognizer phones.

Stage 1 (``distributed_inference`` with the masked XEUS recognizer) dumps
predicted phones per utterance. This runner reads that dump and forced-aligns
each utterance's *predicted* phones with per-utterance ``mfa align_one``, reusing
``MFASingleInference`` (``recognizer=None``).
"""

from typing import Any, List

from src.metrics.types import SegmentationUnit
from src.model.mfa.inference_baseline import build_mfa_single_inference
from src.model.mfa.mfa2_align import load_recog_phones


class MFAAlignFromDumpInference:
    """Align Stage-1 recognized phones with ``mfa align_one`` per utterance."""

    def __init__(self, recog_jsonl: str, **mfa_kwargs: Any):
        """Args:
        recog_jsonl: Glob of the Stage-1 recognition dump jsonl(s).
        **mfa_kwargs: Forwarded to ``build_mfa_single_inference`` (e.g.
            acoustic_model, dictionary, phone_normalizer, cache_dir, sr).
        """
        self.phones_by_uid = load_recog_phones(recog_jsonl)
        self.mfa = build_mfa_single_inference(
            recognizer=None, units="phones", **mfa_kwargs
        )  # recognizer is None here because we are not doing cascade dynamically

    def __call__(
        self, speech, speech_length, utt_id, **kwargs
    ) -> List[SegmentationUnit]:
        """Align one utterance's predicted phones; empty list if none."""
        phones = self.phones_by_uid.get(utt_id, [])
        if not phones:
            return []
        return self.mfa(
            speech,
            speech_length,
            text="",
            phones=phones,
            language=kwargs.get("language"),
        )


def build_mfa_align_from_dump(
    recog_jsonl: str, **mfa_kwargs: Any
) -> MFAAlignFromDumpInference:
    """Hydra entry point: instantiate MFAAlignFromDumpInference."""
    return MFAAlignFromDumpInference(recog_jsonl, **mfa_kwargs)
