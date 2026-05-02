"""Phonvec training/tuning utilities.

Holds every helper that ``src/recipe/phonvec/train.py`` calls so the
entrypoint stays a thin three-stage pipeline (load data → fit/load
Segmenter → grid-search + save artifact).

Sections:
  - Encoder feature extraction: ``build_per_phone_df`` (fit), ``collect_eval_inputs`` (tune).
  - GT/pred conversion: ``gt_units``, ``pred_frames_to_units``.
  - Grid utilities: ``iter_grid_points``, ``apply_grid_point``.
  - Pipeline stages: ``instantiate_fit_dataset``, ``instantiate_tune_dataset``,
    ``fit_or_load_segmenter``, ``run_grid_search``, ``save_phonvec_artifact``.
"""

import itertools
import os
from typing import Iterable, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)
from src.model.phonvec.model import Segmenter, SilenceHandler
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


# IPA labels treated as silence (ipa column NaN) — same set the legacy
# notebook / script uses when fitting PhonologicalVectors.
SILENCE_LABELS = {"h#", "pau", "ʔ̞"}


# ──────────────────────────────────────────────────────────────────────
# Encoder feature extraction
# ──────────────────────────────────────────────────────────────────────


# NOTE(shikhar): If end-to-end training,
# remember to remove the torch.no_grad here
@torch.no_grad()
def _extract_net_feats(net, speech, device, frame_shift):
    """Run the encoder forward on a single utterance
        with small padding on both ends and data in middle.

    Returns: (numpy (T_f, D), valid_len_in_frames).
    """
    if isinstance(speech, np.ndarray):
        speech = torch.from_numpy(speech)
    speech = speech.float()
    pad = (400 - frame_shift) // 2
    sp_padded = F.pad(speech, (pad, pad)).unsqueeze(0).to(device)
    feats, feat_lens = net.encode(
        sp_padded,
        torch.tensor([sp_padded.shape[-1]], device=device),
    )
    vlen = int(feat_lens[0])
    return feats[0, :vlen].cpu().numpy(), vlen


def build_per_phone_df(
    dataset,
    net,
    device: str,
    frame_shift: int,
    sr: int,
) -> pd.DataFrame:
    """Builds a dataframe by walking over each phone in each utterance in the dataset.
    A row corresponds to the phone with center-frame encoder features and has columns:
        feat: encoder features at the center frame of the current phone
        ipa: IPA label of the current phone (NaN if in SILENCE_LABELS)
        l_1: IPA label of the left-adjacent phone (None if no left neighbor)
        r_1: IPA label of the right-adjacent phone (None if no right neighbor)
        audio_path: the utt_id of the current phone, a backpointer to the original audio and metadata
        min: start time of the phone in seconds,
    This is the schema `Segmenter._fit_from_df` expects.
    """
    rows: List[dict] = []
    for i in tqdm(
        range(len(dataset)), desc="Building per-phone features", leave=False
    ):
        item = dataset[i]
        speech = item["speech"]
        speech_length = int(item["speech_length"])
        phones = list(item["phones"])
        phone_timestamps = list(item["phone_timestamps"])
        utt_id = item["utt_id"]

        if speech_length <= 0 or not phones:
            continue

        feats_np, vlen = _extract_net_feats(
            net,
            speech[:speech_length],
            device,
            frame_shift,
        )
        if vlen <= 0:
            continue

        for j, ((s_sec, e_sec), p) in enumerate(zip(phone_timestamps, phones)):
            mid_sec = 0.5 * (float(s_sec) + float(e_sec))
            mid_frame = int(round(mid_sec * sr / frame_shift))
            mid_frame = max(0, min(mid_frame, vlen - 1))
            l_1 = phones[j - 1] if j > 0 else None
            r_1 = phones[j + 1] if j + 1 < len(phones) else None
            ipa_label = float("nan") if p in SILENCE_LABELS else p
            rows.append(
                {
                    "feat": feats_np[mid_frame].copy(),
                    "ipa": ipa_label,
                    "l_1": l_1,
                    "r_1": r_1,
                    "audio_path": utt_id,
                    "min": float(s_sec),
                }
            )

    return pd.DataFrame(rows)


def collect_eval_inputs(
    dataset,
    net,
    device: str,
    frame_shift: int,
    sr: int,
    desc: str = "Caching eval features",
) -> List[dict]:
    """Cache (utt_id, net_feats, waveform_np, gt) per utterance.

    Used during tuning so each grid point only re-runs the cheap segmenter
    pipeline, not the encoder forward.
    """
    cache: List[dict] = []
    for i in tqdm(range(len(dataset)), desc=desc, leave=False):
        item = dataset[i]
        speech_length = int(item["speech_length"])
        sp = item["speech"][:speech_length]
        if isinstance(sp, torch.Tensor):
            sp_np = sp.float().cpu().numpy()
        else:
            sp_np = np.asarray(sp, dtype=np.float32)
        feats_np, vlen = _extract_net_feats(net, sp, device, frame_shift)
        cache.append(
            {
                "utt_id": item["utt_id"],
                "net_feats": feats_np,
                "waveform_np": sp_np,
                "phone_timestamps": list(item["phone_timestamps"]),
                "phones": list(item["phones"]),
                "vlen": vlen,
            }
        )
    return cache


# ──────────────────────────────────────────────────────────────────────
# GT / prediction conversion
# ──────────────────────────────────────────────────────────────────────


def pred_frames_to_units(
    pred_frames,
    vlen: int,
    frame_shift: int,
    sr: int,
) -> List[SegmentationUnit]:
    """Phonvec frame indices → contiguous SegmentationUnit time spans."""
    times = sorted({int(p) for p in pred_frames if 0 <= int(p) < vlen})
    if len(times) < 2:
        return []
    return [
        SegmentationUnit(
            start=times[i] * frame_shift / sr,
            end=times[i + 1] * frame_shift / sr,
        )
        for i in range(len(times) - 1)
    ]


def gt_units(
    phone_timestamps,
    phones,
    *,
    strip_outer_silences: bool,
) -> List[SegmentationUnit]:
    """Convert dataset phone_timestamps/phones to SegmentationUnits.

    The dataset transform (when registered) has already merged TIMIT
    closures into stops and collapsed adjacent silences. Outer-silence
    stripping is left as an evaluator-side decision.
    """
    pts = list(phone_timestamps)
    phs = list(phones)
    if strip_outer_silences and phs:
        if phs[0] in SILENCE_LABELS:
            pts, phs = pts[1:], phs[1:]
        if phs and phs[-1] in SILENCE_LABELS:
            pts, phs = pts[:-1], phs[:-1]
    return [SegmentationUnit(start=float(s), end=float(e)) for s, e in pts]


# ──────────────────────────────────────────────────────────────────────
# Grid utilities
# ──────────────────────────────────────────────────────────────────────

_GRID_KEYS = (
    "use_combined",
    "drop_k",
    "norm_method",
    "combined_prominence",
    "single_signal_prominence",
    "snap_silence",
    "snap_tolerance",
)


def iter_grid_points(grid_cfg: dict) -> Iterable[dict]:
    """Cartesian product over a dict of lists; yields one dict per point."""
    keys = list(grid_cfg.keys())
    for combo in itertools.product(*(grid_cfg[k] for k in keys)):
        yield dict(zip(keys, combo))


def apply_grid_point(default: dict, point: dict) -> dict:
    """Overlay a flat grid-point dict onto the structured Segmenter hparams."""
    h = dict(default)
    for k in _GRID_KEYS:
        if k in point:
            h[k] = point[k]
    return h


# ──────────────────────────────────────────────────────────────────────
# Pipeline stages (called from train.py)
# ──────────────────────────────────────────────────────────────────────


def instantiate_fit_dataset(cfg: DictConfig, *, skip: bool):
    """Build the fit datamodule's train_dataset, or return None when skipping."""
    if skip:
        return None
    log.info(f"Instantiating fit datamodule <{cfg.data._target_}>")
    dm = hydra.utils.instantiate(cfg.data)
    if hasattr(dm, "prepare_data"):
        dm.prepare_data()
    dm.setup(stage="fit")
    fit_ds = dm.train_dataset
    if fit_ds is None:
        raise RuntimeError(
            "Train split missing — required to fit PhonologicalVectors."
        )
    log.info(f"Fit dataset size: {len(fit_ds)}")
    return fit_ds


def instantiate_tune_dataset(pt_cfg: DictConfig):
    """Build the tune datamodule's tune_dataset.

    `phonvec_tune.tune_data` is required: a `SegmentationDataModule` config
    whose `hf_repo` has a registered `HF_REPO_SPLIT_TRANSFORMS` entry that
    produces a `tune` split. Same-dataset and cross-dataset cases both go
    through this path — there is no fallback.
    """
    assert pt_cfg.get("tune_data") is not None, (
        "phonvec_tune.tune_data must be set — point at a SegmentationDataModule "
        "whose hf_repo has a registered HF_REPO_SPLIT_TRANSFORMS entry that "
        "produces a 'tune' split."
    )
    log.info(f"Instantiating tune datamodule <{pt_cfg.tune_data._target_}>")
    tune_dm = hydra.utils.instantiate(pt_cfg.tune_data)
    if hasattr(tune_dm, "prepare_data"):
        tune_dm.prepare_data()
    tune_dm.setup(stage="fit")
    tune_ds = tune_dm.tune_dataset
    assert tune_ds is not None, (
        f"hf_repo={pt_cfg.tune_data.hf_repo} did not produce a 'tune' split — "
        "register a split_transform in dataset_processing_transforms.py."
    )
    log.info(f"Tune dataset size: {len(tune_ds)}")
    return tune_ds


def fit_or_load_segmenter(
    pt_cfg: DictConfig,
    fit_ds,
    net,
    device: str,
    frame_shift: int,
    sr: int,
    mel_frame_shift_ms: int,
) -> Tuple[Segmenter, Optional[dict]]:
    """
    Returns (Segmentation, saved_net_spec).
        Segmentation: either a newly fit Segmenter or one loaded from artifact.
        saved_net_spech: non-None when loaded from artifact, useful to restore
            saved net_spec rather than deriving from config
    """
    fit_artifact_path = pt_cfg.get("fit_artifact")
    if fit_artifact_path is not None:
        log.info(f"Loading pre-fit Segmenter from {fit_artifact_path}")
        artifact = torch.load(
            fit_artifact_path,
            map_location="cpu",
            weights_only=False,
        )
        return Segmenter.from_artifact(artifact), artifact["net"]

    # FIT
    log.info("Building per-phone DataFrame for fit...")
    train_df = build_per_phone_df(fit_ds, net, device, frame_shift, sr)
    log.info(f"Rows in the fit DataFrame: {len(train_df)}")

    log.info(f"Loading silence detector from {pt_cfg.silence_detector_path}")
    silence_handler = SilenceHandler(detector_path=pt_cfg.silence_detector_path)

    log.info("Fitting base Segmenter...")
    seg = Segmenter.fit(
        train_df,
        silence_handler,
        frame_shift=frame_shift,
        sr=sr,
        mel_frame_shift_ms=mel_frame_shift_ms,
    )
    return seg, None


def run_grid_search(
    base_seg: Segmenter,
    tune_cache: List[dict],
    pt_cfg: DictConfig,
    frame_shift: int,
    sr: int,
) -> Tuple[Segmenter, List[dict]]:
    """Run grid search over `pt_cfg.grid`; return (best_seg, all_results)."""
    evaluator = SegmentationEvaluator(
        tolerance_ms=int(pt_cfg.get("tolerance_ms", 20))
    )
    strip_outer = bool(pt_cfg.get("strip_outer_silences", True))
    grid_cfg = OmegaConf.to_container(pt_cfg.grid, resolve=True)
    default_hparams = base_seg.default_hparams()

    log.info("Running grid search...")
    results = []
    for point in iter_grid_points(grid_cfg):
        seg = base_seg.with_hparams(apply_grid_point(default_hparams, point))
        preds_dict, gt_dict = {}, {}
        for c in tune_cache:
            pred_frames = seg.segment(c["net_feats"], c["waveform_np"])
            preds_dict[c["utt_id"]] = pred_frames_to_units(
                pred_frames,
                c["vlen"],
                frame_shift,
                sr,
            )
            gt_dict[c["utt_id"]] = gt_units(
                c["phone_timestamps"],
                c["phones"],
                strip_outer_silences=strip_outer,
            )
        score = evaluator.evaluate_batch(preds_dict, gt_dict)
        rval = float(score.get("rval", 0.0))
        log.info(f"  {point} -> RV={rval:.4f}")
        results.append({"point": point, "rval": rval, "score": score})

    best = max(results, key=lambda r: r["rval"])
    log.info(f"Best grid point: {best['point']} (RV={best['rval']:.4f})")
    best_seg = base_seg.with_hparams(
        apply_grid_point(default_hparams, best["point"])
    )
    return best_seg, results


def save_phonvec_artifact(
    seg: Segmenter,
    saved_net_spec: Optional[dict],
    pt_cfg: DictConfig,
    results: List[dict],
    frame_shift: int,
    sr: int,
    mel_frame_shift_ms: int,
) -> str:
    """Save Segmenter + net spec + tune metrics to ``pt_cfg.out_artifact``.

    When ``saved_net_spec`` is provided (i.e. retuning a pre-fit artifact),
    its values are preserved verbatim; otherwise ``net_spec`` is derived
    from the current run's config.
    """
    net_spec = (
        dict(saved_net_spec)
        if saved_net_spec is not None
        else {
            "hf_repo": pt_cfg.net.get("hf_repo"),
            "encoder_layer": pt_cfg.net.get("encoder_layer", -1),
            "frame_shift": frame_shift,
            "sr": sr,
            "mel_frame_shift_ms": mel_frame_shift_ms,
        }
    )
    best = max(results, key=lambda r: r["rval"])
    artifact = seg.to_artifact(
        net_spec=net_spec,
        tune_metrics={
            "best": {"point": best["point"], "rval": best["rval"]},
            "grid": [{"point": r["point"], "rval": r["rval"]} for r in results],
        },
    )
    out_path = pt_cfg.out_artifact
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(artifact, out_path)
    log.info(f"Saved phonvec artifact to {out_path}")
    return out_path
