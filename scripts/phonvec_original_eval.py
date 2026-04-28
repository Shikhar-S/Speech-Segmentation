"""Reproduce the final TIMIT test set evaluation from segment_phones.ipynb.

Uses the Segmenter class from segment_phones.py with the best configuration:
  - norm_method: min (subtract nanmin)
  - drop_k: 2
  - signals: frame_delta, fwd_delta, bwd_delta, fwd_contrast, bwd_contrast, mel_svf
  - prominence: 0.001

Expected output (from notebook):
  TIMIT_test  RV=0.869  (snapped=0.877)  P=0.885 R=0.826
"""
import warnings
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")

import argparse
import csv
import hashlib
import os
import pickle

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from src.model.phonvec.model import SR, FRAME_SHIFT, Segmenter


# --- PrecisionRecallMetric (from Strgar & Harwath) ---

class PrecisionRecallMetric:
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.eps = 1e-8
        self.data = []

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        os = recall / (precision + self.eps) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os**2)
        r2 = (-os + recall - 1) / (np.sqrt(2))
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2
        return {"precision": precision, "recall": recall, "f1": f1, "rval": rval, "over_seg": os}

    def update(self, seg, pos_pred):
        self.data.append((seg, pos_pred))

    def get_counts(self, gt, pred):
        match_counter = 0
        used_idxs = []
        matches = {i: [] for i in range(len(pred))}
        for i, yhat_i in enumerate(pred):
            dists = np.abs(gt - yhat_i)
            idxs = np.argsort(dists)
            for idx in idxs:
                if dists[idx] <= self.tolerance:
                    matches[i].append(idx)

        for m, vs in matches.items():
            vs = sorted(vs)
            for v in vs:
                if v not in used_idxs:
                    used_idxs.append(v)
                    match_counter += 1
                    break
        return match_counter

    def compute(self):
        n_gts, n_preds, p_count, r_count = 0, 0, 0, 0
        for y, yhat in self.data:
            n_gts += len(y)
            n_preds += len(yhat)
            p_count += self.get_counts(y, yhat)
            r_count += self.get_counts(yhat, y)
        return self.get_metrics(p_count, r_count, n_preds, n_gts)

    def compute_single(self, gt, pred):
        p_count = self.get_counts(gt, pred)
        r_count = self.get_counts(pred, gt)
        return self.get_metrics(p_count, r_count, len(pred), len(gt))


# --- TIMIT loading ---

_CLOSURE_TO_STOP = {
    "bcl": "b", "dcl": "d", "gcl": "g",
    "pcl": "p", "tcl": "t", "kcl": "k",
}
_SILENCE_LABELS = {"h#", "pau", "epi"}


def _simplify_phn_segments(segments):
    merged = []
    i = 0
    while i < len(segments):
        start, end, label = segments[i]
        if label in _CLOSURE_TO_STOP and i + 1 < len(segments):
            next_start, next_end, next_label = segments[i + 1]
            expected = _CLOSURE_TO_STOP[label]
            is_affricate = (label == "dcl" and next_label == "jh") or \
                           (label == "tcl" and next_label == "ch")
            if next_label == expected or is_affricate:
                merged.append((start, next_end, next_label))
                i += 2
                continue
        merged.append((start, end, label))
        i += 1

    collapsed = []
    for start, end, label in merged:
        if label in _SILENCE_LABELS:
            if collapsed and collapsed[-1][2] in _SILENCE_LABELS:
                prev_start, _, prev_label = collapsed[-1]
                collapsed[-1] = (prev_start, end, prev_label)
                continue
        collapsed.append((start, end, label))
    return collapsed


def _load_timit_sample(path):
    audio, sr = librosa.load(path, sr=SR, mono=True)
    phn_path = os.path.splitext(path)[0] + ".PHN"
    segments = []
    with open(phn_path) as f:
        for line in f:
            start, end, label = line.strip().split()
            segments.append((int(start), int(end), label))

    segments = _simplify_phn_segments(segments)
    inner = segments[1:-1]
    boundary_samples = np.array([inner[0][0], inner[0][1]] + [end for _, end, _ in inner[1:]])
    return audio, boundary_samples, segments


def _boundary_frames(audio, boundary_samples, frame_shift=FRAME_SHIFT):
    n_frames = len(audio) // frame_shift
    boundary_frames = np.array([round(s / frame_shift) for s in boundary_samples])
    return np.clip(boundary_frames, 0, n_frames - 1)


# --- WavLM feature extraction ---

def _extract_wavlm_feats(audio, audio_path, cache_dir="feats/cache"):
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha1(audio_path.encode("utf-8")).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"wavlm_feats_{digest}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = AutoModel.from_pretrained("microsoft/wavlm-large")
    model.eval()

    pad = (400 - FRAME_SHIFT) // 2
    audio_padded = np.pad(audio, (pad, pad), mode="constant")
    inputs = processor(raw_speech=[audio_padded], sampling_rate=SR, padding=False, return_tensors="pt")
    with torch.no_grad():
        feats = model(**inputs).last_hidden_state[0].cpu().numpy()

    with open(cache_path, "wb") as f:
        pickle.dump(feats, f)
    return feats


def main():
    # python -m scripts.phonvec_original_eval --csv exp/runs/phonological_vectors/test.csv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", metavar="PATH",
                        help="Write per-utterance predicted segments and metrics to a CSV file")
    parser.add_argument("--snapped-only", action="store_true",
                        help="Only evaluate snapped boundaries (skip raw)")
    args = parser.parse_args()

    print("Loading TIMIT data...")
    CACHE_DIR="/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/phonological_vectors/cache"
    timit_pickled_file = '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/data/phonological_vector/timit-wavlm-large-24-center-featslice.pkl'
    silence_detector_path = '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/data/phonological_vector/logistic_regression_silence_detector.joblib'
    timit_all_df = pd.read_pickle(timit_pickled_file)
    timit_train_df = timit_all_df[timit_all_df.split == "train"]
    timit_test_df = timit_all_df[timit_all_df.split == "test"]
    timit_test_paths = timit_test_df["audio_path"].unique()
    print(f"Test utterances: {len(timit_test_paths)}")

    print("Building segmenter from training data...")
    segmenter = Segmenter(timit_train_df, silence_detector_path)

    print("Evaluating on TIMIT test set...")
    m_raw = None if args.snapped_only else PrecisionRecallMetric(tolerance=1)
    m_snapped = PrecisionRecallMetric(tolerance=1)
    csv_rows = []

    for path in tqdm(timit_test_paths, desc="Segmenting"):
        audio, boundary_samples, segments = _load_timit_sample(path)
        ref = _boundary_frames(audio, boundary_samples)
        wavlm_feats = _extract_wavlm_feats(audio, path, cache_dir=CACHE_DIR)

        preds_snapped = segmenter.segment(wavlm_feats, audio, use_combined=True, snap_silence=True)
        m_snapped.update(ref, preds_snapped)

        if m_raw is not None:
            preds_raw = segmenter.segment(wavlm_feats, audio, use_combined=True, snap_silence=False)
            m_raw.update(ref, preds_raw)

        if args.csv:
            metrics = m_snapped.compute_single(ref, preds_snapped)
            csv_rows.append({
                "audio_path": path,
                "n_ref": len(ref),
                "n_pred": len(preds_snapped),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "rval": round(metrics["rval"], 4),
                "over_seg": round(metrics["over_seg"], 4),
                "ref_frames": " ".join(str(int(b)) for b in ref),
                "pred_frames": " ".join(str(int(b)) for b in preds_snapped),
            })

    r_snapped = m_snapped.compute()

    if m_raw is not None:
        r_raw = m_raw.compute()
        print(f"\n{'Metric':<12s} {'Raw':>8s} {'Snapped':>8s}")
        print("-" * 30)
        for name in ("precision", "recall", "f1", "rval"):
            print(f"{name:<12s} {r_raw[name]:>8.3f} {r_snapped[name]:>8.3f}")
        print(f"{'over_seg':<12s} {r_raw['over_seg']*100:>7.1f}% {r_snapped['over_seg']*100:>7.1f}%")
    else:
        print(f"\n{'Metric':<12s} {'Snapped':>8s}")
        print("-" * 22)
        for name in ("precision", "recall", "f1", "rval"):
            print(f"{name:<12s} {r_snapped[name]:>8.3f}")
        print(f"{'over_seg':<12s} {r_snapped['over_seg']*100:>7.1f}%")

    print(f"\nExpected: TIMIT_test RV=0.869 (snapped=0.877) P=0.885 R=0.826")

    if args.csv:
        fieldnames = ["audio_path", "n_ref", "n_pred",
                      "precision", "recall", "f1", "rval", "over_seg",
                      "ref_frames", "pred_frames"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote {len(csv_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()