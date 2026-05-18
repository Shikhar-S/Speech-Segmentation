"""Phonvec training/tuning utilities.

Holds every helper that ``src/recipe/phonvec/train.py`` calls so the
entrypoint stays a four-stage pipeline (load data → fit/load Segmenter →
per-signal correlation tuning → combined grid-search + save artifact).

Sections:
  - Encoder feature extraction: ``build_per_phone_df`` (fit),
    ``collect_eval_inputs`` (tune).
  - GT/pred conversion: ``gt_units``, ``pred_frames_to_units``.
  - Grid utilities: ``iter_grid_points``.
  - Per-signal tuning: ``run_signal_tuning``.
  - Combined grid search: ``run_grid_search``.
  - Pipeline stages: ``instantiate_fit_dataset``,
    ``instantiate_tune_dataset``, ``fit_or_load_segmenter``,
    ``save_phonvec_artifact``.
"""

import hashlib
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

from joblib import Memory
from scipy.signal import find_peaks

from src.core.ipa_utils import IPA_SILENCE_LABELS


from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)
from src.model.phonvec.model import (
    PhonologicalVectors,
    Segmenter,
    SilenceHandler,
    _combine_stacked,
    _normalize_signal,
    _shift_signal,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

_cache_path = os.environ.get("CACHE_DIR")
_feat_memory = (
    Memory(
        location=os.path.join(_cache_path, "phonvec_feats"),
        verbose=0,
    )
    if _cache_path
    else None
)
if _feat_memory is None:
    print("Warning: CACHE_DIR is not set. Proceeding without caching.")


def _maybe_cache(**kwargs):
    """``@_feat_memory.cache`` when CACHE_DIR is set, no-op otherwise."""
    def decorator(fn):
        if _feat_memory is not None:
            return _feat_memory.cache(fn, **kwargs)
        return fn
    return decorator


def make_cache_id(
    net_cfg: DictConfig,
    hf_repo: str,
    split: str,
) -> str:
    """Deterministic hash identifying a (model, dataset-split) pair."""
    net_yaml = OmegaConf.to_yaml(net_cfg, resolve=True)
    raw = f"{net_yaml}|{hf_repo}|{split}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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


@_maybe_cache(ignore=["dataset", "net", "device"])
def build_per_phone_df(
    cache_id: str,
    dataset,
    net,
    device: str,
    frame_shift: int,
    sr: int,
) -> pd.DataFrame:
    """Build per-phone center-frame features from encoder output.

    Each row has columns: feat, ipa, l_1, r_1, audio_path, min.
    This is the schema ``Segmenter._fit_from_df`` expects.

    Args:
        cache_id: Opaque key for joblib caching (not used in logic).
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
            ipa_label = float("nan") if p in IPA_SILENCE_LABELS else p
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


@_maybe_cache(ignore=["train_df"])
def _fit_components(cache_id: str, train_df) -> dict:
    """Fit PhonologicalVectors + cross-position regressors.

    Returns a dict suitable for ``Segmenter(..., _from_components=)``,
    minus the ``silence_handler`` (which is not data-dependent).

    Args:
        cache_id: Opaque key for joblib caching (not used in logic).
    """
    pv_train_df = train_df[~train_df.ipa.isna()]
    vocab = pv_train_df.ipa.unique().tolist()

    pv_ipa = PhonologicalVectors(pv_train_df, vocab, group_col="ipa")
    pv_l1 = PhonologicalVectors(pv_train_df, vocab, group_col="l_1")
    pv_r1 = PhonologicalVectors(pv_train_df, vocab, group_col="r_1")

    df_sorted = pv_train_df.sort_values(["audio_path", "min"])
    same_utt = (
        df_sorted.audio_path.values[:-1]
        == df_sorted.audio_path.values[1:]
    )
    prev_feats = np.stack(df_sorted.feat.values[:-1][same_utt])
    curr_feats = np.stack(df_sorted.feat.values[1:][same_utt])

    proj_ipa_curr = pv_ipa.project_raw(curr_feats)
    proj_ipa_prev = pv_ipa.project_raw(prev_feats)
    proj_r1_prev = pv_r1.project_raw(prev_feats)
    proj_l1_next = pv_l1.project_raw(curr_feats)

    W_r1_to_ipa = np.linalg.lstsq(
        proj_r1_prev, proj_ipa_curr, rcond=None,
    )[0]
    W_l1_to_ipa = np.linalg.lstsq(
        proj_l1_next, proj_ipa_prev, rcond=None,
    )[0]

    return {
        "pv_ipa": pv_ipa,
        "pv_l1": pv_l1,
        "pv_r1": pv_r1,
        "W_r1_to_ipa": W_r1_to_ipa,
        "W_l1_to_ipa": W_l1_to_ipa,
    }


@_maybe_cache(ignore=["dataset", "net", "device", "desc"])
def collect_eval_inputs(
    cache_id: str,
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

    Args:
        cache_id: Opaque key for joblib caching (not used in logic).
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
        if phs[0] in IPA_SILENCE_LABELS:
            pts, phs = pts[1:], phs[1:]
        if phs and phs[-1] in IPA_SILENCE_LABELS:
            pts, phs = pts[:-1], phs[:-1]
    return [SegmentationUnit(start=float(s), end=float(e)) for s, e in pts]


# ──────────────────────────────────────────────────────────────────────
# Grid utilities
# ──────────────────────────────────────────────────────────────────────

def iter_grid_points(grid_cfg: dict) -> Iterable[dict]:
    """Cartesian product over a dict of lists; yields one dict per point."""
    keys = list(grid_cfg.keys())
    for combo in itertools.product(*(grid_cfg[k] for k in keys)):
        yield dict(zip(keys, combo))


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
        "register a split_transform in dataset_splitting_transforms.py."
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
    fit_cache_id: str = "",
) -> Tuple[Segmenter, Optional[dict]]:
    """Returns (Segmenter, saved_net_spec).

    Args:
        fit_cache_id: Passed to ``build_per_phone_df`` for joblib caching.
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
    train_df = build_per_phone_df(
        fit_cache_id, fit_ds, net, device, frame_shift, sr
    )
    log.info(f"Rows in the fit DataFrame: {len(train_df)}")

    log.info(f"Loading silence detector from {pt_cfg.silence_detector_path}")
    silence_handler = SilenceHandler(detector_path=pt_cfg.silence_detector_path)

    log.info("Fitting PhonologicalVectors + regressors...")
    components = _fit_components(fit_cache_id, train_df)

    seg = Segmenter(
        frame_shift=frame_shift,
        sr=sr,
        mel_frame_shift_ms=mel_frame_shift_ms,
        _from_components={**components, "silence_handler": silence_handler},
        hparams={},
    )
    return seg, None


def _boundary_target(
    phone_timestamps,
    phones,
    n_frames: int,
    frame_shift: int,
    sr: int,
    strip_outer_silences: bool,
) -> np.ndarray:
    """Binary boundary-target vector at the encoder frame rate."""
    units = gt_units(
        phone_timestamps, phones,
        strip_outer_silences=strip_outer_silences,
    )
    boundary_secs = set()
    for u in units:
        boundary_secs.add(u.start)
        boundary_secs.add(u.end)
    target = np.zeros(n_frames)
    for t in boundary_secs:
        idx = int(round(t * sr / frame_shift))
        if 0 <= idx < n_frames:
            target[idx] = 1.0
    return target


def run_signal_tuning(
    seg: Segmenter,
    tune_cache: List[dict],
    signal_grid: dict,
    shift_values: List[int],
    frame_shift: int,
    sr: int,
    strip_outer_silences: bool = True,
) -> dict:
    """Per-signal correlation analysis.

    For each signal, grid-searches kwargs and shift values to find the
    configuration that maximises mean Pearson correlation with a binary
    boundary target.

    Args:
        signal_grid: ``{signal_name: {kwarg_name: [vals], ...}, ...}``
        shift_values: shift offsets to try (e.g. ``[-2,-1,0,1,2]``).

    Returns:
        ``{signal_name: {"kwargs": dict, "shift": int,
        "correlation": float}}``
    """
    log.info("Preparing projections for signal tuning...")
    prepped = []
    for c in tqdm(
        tune_cache, desc="Preparing tune data", leave=False,
    ):
        net_feats = c["net_feats"]
        prepped.append({
            "proj_ipa": seg.pv_ipa.project_raw(net_feats),
            "proj_r1": seg.pv_r1.project_raw(net_feats),
            "proj_l1": seg.pv_l1.project_raw(net_feats),
            "waveform_np": c["waveform_np"],
            "boundary_target": _boundary_target(
                c["phone_timestamps"], c["phones"],
                len(net_feats), frame_shift, sr,
                strip_outer_silences,
            ),
        })

    log.info("Running per-signal correlation analysis...")
    results = {}
    for signal_name, kwarg_ranges in signal_grid.items():
        best_corr = -np.inf
        best_config = None

        for kwargs in iter_grid_points(kwarg_ranges):
            corrs_by_shift = {s: [] for s in shift_values}

            for p in prepped:
                sig = seg._signal(
                    signal_name,
                    p["proj_ipa"], p["proj_r1"],
                    p["proj_l1"], p["waveform_np"],
                    kwargs=kwargs,
                )
                normed = _normalize_signal(sig, "minmax")

                for shift in shift_values:
                    shifted = _shift_signal(normed, shift)
                    finite = np.isfinite(shifted)
                    if finite.sum() < 2:
                        continue
                    s = shifted[finite]
                    t = p["boundary_target"][finite]
                    s_std, t_std = s.std(), t.std()
                    if s_std > 0 and t_std > 0:
                        corr = float(np.mean(
                            (s - s.mean()) / s_std
                            * (t - t.mean()) / t_std
                        ))
                        corrs_by_shift[shift].append(corr)

            for shift in shift_values:
                if not corrs_by_shift[shift]:
                    continue
                mean_corr = float(np.mean(corrs_by_shift[shift]))
                if mean_corr > best_corr:
                    best_corr = mean_corr
                    best_config = {
                        "kwargs": dict(kwargs),
                        "shift": shift,
                        "correlation": mean_corr,
                    }

        if best_config is not None:
            results[signal_name] = best_config
            log.info(
                f"  {signal_name}: corr={best_config['correlation']:.4f}"
                f"  kwargs={best_config['kwargs']}"
                f"  shift={best_config['shift']}"
            )

    return results


def _precompute_signals(seg, tune_cache, signal_configs):
    """Pre-compute shifted signals and silence masks for all utterances.

    Uses the best per-signal kwargs/shifts from ``run_signal_tuning``.

    Args:
        signal_configs: ``{name: {"kwargs": dict, "shift": int, ...}}``
    """
    precomputed = []
    for c in tqdm(
        tune_cache, desc="Pre-computing signals", leave=False,
    ):
        net_feats = c["net_feats"]
        waveform_np = c["waveform_np"]
        proj_ipa = seg.pv_ipa.project_raw(net_feats)
        proj_r1 = seg.pv_r1.project_raw(net_feats)
        proj_l1 = seg.pv_l1.project_raw(net_feats)

        shifted = {}
        for name, cfg in signal_configs.items():
            sig = seg._signal(
                name, proj_ipa, proj_r1, proj_l1,
                waveform_np, kwargs=cfg["kwargs"],
            )
            shifted[name] = _shift_signal(sig, cfg["shift"])

        silence_mask = seg.silence_handler.predict_silence_mask(
            net_feats,
        )
        precomputed.append({
            "shifted_signals": shifted,
            "silence_mask": silence_mask,
        })
    return precomputed


def run_grid_search(
    base_seg: Segmenter,
    tune_cache: List[dict],
    signal_configs: dict,
    pt_cfg: DictConfig,
    frame_shift: int,
    sr: int,
) -> Tuple[Segmenter, List[dict]]:
    """Grid search over combined hparams; return (best_seg, results).

    Args:
        signal_configs: output of ``run_signal_tuning``.
    """
    evaluator = SegmentationEvaluator(
        tolerance_ms=int(pt_cfg.get("tolerance_ms", 20))
    )
    strip_outer = bool(pt_cfg.get("strip_outer_silences", True))
    grid_cfg = OmegaConf.to_container(pt_cfg.grid, resolve=True)

    precomputed = _precompute_signals(
        base_seg, tune_cache, signal_configs,
    )

    gt_dict = {}
    for c in tune_cache:
        gt_dict[c["utt_id"]] = gt_units(
            c["phone_timestamps"],
            c["phones"],
            strip_outer_silences=strip_outer,
        )

    log.info("Running combined grid search...")
    results = []
    for point in iter_grid_points(grid_cfg):
        norm = point["norm_method"]
        drop_k = point["drop_k"]
        prominence = point["prominence"]
        snap_silence = point["snap_silence"]
        snap_tolerance = point.get("snap_tolerance", 2)
        min_corr = point.get("min_correlation", 0.0)

        active = [
            n for n, c in signal_configs.items()
            if c["correlation"] >= min_corr
        ]
        if len(active) < 2:
            continue

        preds_dict = {}
        for c, pc in zip(tune_cache, precomputed):
            components = [
                _normalize_signal(
                    pc["shifted_signals"][name].copy(), norm,
                )
                for name in active
            ]
            stacked = np.stack(components, axis=0)
            eff_drop = min(drop_k, len(active) - 1)
            if eff_drop > 0:
                stacked = np.sort(stacked, axis=0)[eff_drop:]
            signal = _combine_stacked(stacked, norm)

            preds = find_peaks(signal, prominence=prominence)[0]
            if snap_silence:
                preds = base_seg.silence_handler.handle_silence(
                    preds=preds,
                    silence_mask=pc["silence_mask"],
                    snap_tolerance=snap_tolerance,
                )
            preds_dict[c["utt_id"]] = pred_frames_to_units(
                preds, c["vlen"], frame_shift, sr,
            )
        score = evaluator.evaluate_batch(preds_dict, gt_dict)
        rval = float(score.get("rval", 0.0))
        log.info(
            f"  {point} -> RV={rval:.4f} ({len(active)} signals)"
        )
        results.append({
            "point": point, "rval": rval,
            "score": score, "active_signals": active,
        })

    best = max(results, key=lambda r: r["rval"])
    log.info(
        f"Best: {best['point']} (RV={best['rval']:.4f})"
    )

    min_corr = best["point"].get("min_correlation", 0.0)
    active = [
        n for n, c in signal_configs.items()
        if c["correlation"] >= min_corr
    ]
    hparams = {
        "combined_signals": active,
        "signal_kwargs": {
            n: signal_configs[n]["kwargs"] for n in active
        },
        "signal_shifts": {
            n: signal_configs[n]["shift"] for n in active
        },
        "signal_correlations": {
            n: signal_configs[n]["correlation"]
            for n in signal_configs
        },
        "drop_k": best["point"]["drop_k"],
        "norm_method": best["point"]["norm_method"],
        "prominence": best["point"]["prominence"],
        "snap_silence": best["point"]["snap_silence"],
        "snap_tolerance": best["point"].get("snap_tolerance", 2),
    }
    best_seg = base_seg.with_hparams(hparams)
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
