"""PhoneticXEUS recognizer constrained to an MFA acoustic model's inventory.

PhoneticXEUS decodes over the full multilingual IPA vocab; a single phone outside
the aligner's inventory breaks that utterance's ``mfa align_one``. We mask
out-of-inventory vocab logits to -inf pre-argmax so every decode is alignable by
construction (blank is always kept). The allowed inventory is the dataset's
normalized GT phone set (``allowed_phones_file``, built by
scripts/build_pxeus_mfa_maskvocab.py); absent one, it falls back to the acoustic
model's full ``meta.json`` inventory.
"""

import json
from typing import List, Optional

import yaml

from src.model.mfa.inference_baseline import _PHONE_NORMALIZERS
from src.model.mfa.utils import (
    ensure_mfa_model,
    mfa_env,
    mfa_extracted_path,
)
from src.model.xeusphoneme.builders import build_xeus_pr_from_hf
from src.model.xeusphoneme.xeuspr_inference import XeusPRInference


def _acoustic_phone_set(acoustic_model: str, cache_dir: Optional[str]) -> set:
    """Phones the aligner accepts, read from the acoustic model's metadata.

    Newer MFA models store the inventory in ``meta.json``; older ones (e.g.
    tamil_cv, v2.0.0b9) store it in ``meta.yaml``.
    """
    env = mfa_env(cache_dir)
    ensure_mfa_model(acoustic_model, dictionary=None, env=env)
    model_dir = mfa_extracted_path(acoustic_model, env)
    meta_json = model_dir / "meta.json"
    if meta_json.exists():
        with open(meta_json) as f:
            return set(json.load(f)["phones"])
    with open(model_dir / "meta.yaml") as f:
        return set(yaml.safe_load(f)["phones"])


def out_of_inventory_token_ids(
    token_list: List[str], phone_set: set, normalizer: str
) -> List[int]:
    """Vocab ids to mask: a token is kept iff, mapped through ``normalizer``, it
    lands in ``phone_set``. <blank> is always kept; tokens that normalize to
    nothing are masked."""
    normalize = _PHONE_NORMALIZERS[normalizer]

    def norm(t: str) -> Optional[str]:
        mapped = normalize([t])
        return mapped[0] if mapped else None

    return [
        i
        for i, t in enumerate(token_list)
        if t != "<blank>" and norm(t) not in phone_set
    ]  # we mask sos, eos, unk!


def build_masked_xeus_recognizer(
    *,
    acoustic_model: str = "english_mfa",
    normalizer: str = "english_mfa",
    allowed_phones_file: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "float32",
    cache_dir: Optional[str] = None,
    **net_kwargs,
) -> XeusPRInference:
    """Build a PhoneticXEUS recognizer restricted to an MFA inventory.

    Args:
        acoustic_model: MFA model whose inventory bounds the output (fallback
            source if ``allowed_phones_file`` is None).
        normalizer: ``_PHONE_NORMALIZERS`` key; must match ``acoustic_model``.
        allowed_phones_file: JSON ``{"phones": [...]}`` of the dataset's
            normalized GT inventory; overrides the acoustic model's full set.
        device, dtype, cache_dir: Recognizer placement and MFA cache.
        **net_kwargs: Forwarded to ``build_xeus_pr_from_hf``.

    Returns:
        A ``XeusPRInference`` that masks out-of-inventory tokens pre-argmax.
    """
    if allowed_phones_file is not None:
        with open(allowed_phones_file) as f:
            phone_set = set(json.load(f)["phones"])
    else:
        phone_set = _acoustic_phone_set(acoustic_model, cache_dir)
    model = build_xeus_pr_from_hf(**net_kwargs)
    masked = out_of_inventory_token_ids(model.token_list, phone_set, normalizer)
    return XeusPRInference(
        model, device=device, dtype=dtype, masked_token_ids=masked
    )
