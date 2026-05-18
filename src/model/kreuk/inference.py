"""Kreuk UnsupSeg inference via the distributed_inference harness.

Loads original Kreuk .ckpt files directly — no conversion step needed.

Usage:
    python src/main.py experiment=inference/kreuk \
        inference.inference_runner.checkpoint=timit
"""

import importlib
import io
import pickle
import types
from types import SimpleNamespace
from typing import List

import logging
from pathlib import Path

import numpy as np
import torch

from src.metrics.segmentation_evaluator import SegmentationUnit
from src.model.kreuk.model import NextFrameClassifier
from src.model.kreuk.utils import (
    detect_peaks,
    max_min_norm,
    replicate_first_k_frames,
)

GITHUB_BASE = (
    "https://github.com/felixkreuk/UnsupSeg/raw/master/pretrained_models"
)

SAMPLE_RATE = 16000
FRAME_SHIFT_SAMPLES = 160

MODEL_SPEC_KEYS = [
    "z_dim",
    "z_proj",
    "z_proj_linear",
    "z_proj_dropout",
    "latent_dim",
    "cosine_coef",
    "pred_steps",
    "pred_offset",
    "n_negatives",
    "batch_shuffle",
]


# ── Legacy checkpoint loading ────────────────────────────────────────


class _LegacyDillUnpickler(pickle.Unpickler):
    """Deserialize dill-pickled defaultdict from Kreuk Solver.

    The original checkpoints were serialized with dill on Python 3.7.
    The CodeType bytecode is incompatible with 3.8+, so we intercept
    the dill helpers and reconstruct only the stored dict entries.
    """

    _DISPATCH = {
        ("dill._dill", "_load_type"): lambda name: (
            (lambda *a, **kw: None)
            if name == "CodeType"
            else getattr(types, name, type(None))
        ),
        ("dill._dill", "_create_function"): (
            lambda *a, **kw: lambda: {
                "prominence": None,
                "width": None,
                "distance": None,
            }
        ),
        ("dill._dill", "_get_attr"): getattr,
        ("dill._dill", "_import_module"): importlib.import_module,
    }

    def find_class(self, module, name):
        key = (module, name)
        if key in self._DISPATCH:
            return self._DISPATCH[key]
        if module == "solver" and name == "__dict__":
            return {}
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return lambda *a, **kw: None


def _load_peak_params(raw_bytes: bytes) -> dict:
    """Extract peak detection params from legacy dill-serialized bytes."""
    data = _LegacyDillUnpickler(io.BytesIO(raw_bytes)).load()
    params = dict(data["cpc_1"])
    params["prominence"] = float(params["prominence"])
    return params


def _load_kreuk_ckpt(path: str, device: str = "cpu"):
    """Load an original Kreuk .ckpt, return (model, peak_params)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = SimpleNamespace(
        **{k: dict(ckpt["hparams"])[k] for k in MODEL_SPEC_KEYS}
    )
    model = NextFrameClassifier(hp)
    weights = {
        k.replace("NFC.", ""): v for k, v in ckpt["state_dict"].items()
    }
    model.load_state_dict(weights)
    peak_params = _load_peak_params(ckpt["peak_detection_params"])
    return model.to(device).eval(), peak_params


# ── Inference class ──────────────────────────────────────────────────


class KreukInference:
    """Per-utterance Kreuk UnsupSeg inference."""

    def __init__(
        self,
        model: NextFrameClassifier,
        peak_params: dict,
        device: str = "cpu",
    ):
        self.model = model
        self.peak_params = peak_params
        self.device = device

    @torch.no_grad()
    def __call__(
        self, speech, speech_length=None, **kwargs
    ) -> List[SegmentationUnit]:
        """Run single-utterance segmentation.

        Args:
            speech: 1-D waveform tensor or numpy array (16 kHz).
            speech_length: Valid sample count.
            **kwargs: Ignored extras from dataset item.

        Returns:
            Contiguous unlabeled SegmentationUnit spans.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        if speech_length is not None:
            speech = speech[: int(speech_length)]
        audio = speech.float().unsqueeze(0).to(self.device)
        duration = audio.shape[1] / SAMPLE_RATE

        preds = self.model(audio)
        preds = preds[1][0]
        preds = replicate_first_k_frames(preds, k=1, dim=1)
        preds = 1 - max_min_norm(preds)
        peaks = detect_peaks(
            x=preds,
            lengths=[preds.shape[1]],
            prominence=self.peak_params["prominence"],
            width=self.peak_params["width"],
            distance=self.peak_params["distance"],
        )[0]

        boundaries = sorted(
            {0.0}
            | {idx * FRAME_SHIFT_SAMPLES / SAMPLE_RATE for idx in peaks}
            | {duration}
        )
        return [
            SegmentationUnit(
                start=boundaries[i],
                end=boundaries[i + 1],
                label=None,
            )
            for i in range(len(boundaries) - 1)
        ]


def _download_ckpt(filename: str, cache_dir: str) -> str:
    """Download a Kreuk checkpoint from GitHub if not already cached."""
    dest = Path(cache_dir) / filename
    if dest.exists():
        logging.info(f"Using cached checkpoint: {dest}")
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GITHUB_BASE}/{filename}"
    logging.info(f"Downloading {url} -> {dest}")
    torch.hub.download_url_to_file(url, str(dest))
    return str(dest)


def build_kreuk_inference(
    checkpoint: str,
    device: str = "cuda",
    cache_dir: str = ".",
) -> KreukInference:
    """Hydra entry point for Kreuk UnsupSeg inference.

    Args:
        checkpoint: Checkpoint filename stem
            (e.g. "timit_pretrained", "buckeye+_pretrained").
        device: Torch device.
        cache_dir: Local directory for caching downloaded checkpoints.
    """
    ckpt_path = _download_ckpt(f"{checkpoint}.ckpt", cache_dir)
    model, peak_params = _load_kreuk_ckpt(ckpt_path, device=device)
    return KreukInference(model, peak_params, device=device)
