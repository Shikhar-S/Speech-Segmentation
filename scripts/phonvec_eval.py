"""Evaluate `Segmenter` (src/model/phonvec/model.py) on a segmentation dataset.

Extracts WavLM-large layer-24 features per utterance, runs the rule-based
phonvec `Segmenter.segment(...)` to predict boundary frame indices, and
compares predicted boundaries against ground-truth phone timestamps using
`SegmentationEvaluator`.

Usage:
    python -m scripts.phonvec_eval \
        --train-df  /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/data/phonological_vector/timit-wavlm-large-24-center-featslice.pkl \
        --silence-detector /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/data/phonological_vector/logistic_regression_silence_detector.joblib \
        --hf-repo changelinglab/timit-segment \
        --split test \
        --out /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/phonological_vectors/timit_segments.jsonl \
        --limit 5
    
    
    python -m scripts.phonvec_eval \
        --train-df  /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/data/phonological_vector/voxangeles-wavlm-large-24-center-featslice.pkl \
        --silence-detector /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/data/phonological_vector/logistic_regression_silence_detector.joblib \
        --hf-repo changelinglab/voxangeles-segment \
        --out /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/phonological_vectors/voxangeles_segments.jsonl \
        --split test
    
    python -m scripts.phonvec_eval \
        --train-df  /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/data/phonological_vector/voxangeles-wavlm-large-24-center-featslice.pkl \
        --silence-detector /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/data/phonological_vector/logistic_regression_silence_detector.joblib \
        --hf-repo changelinglab/buckeye-segment \
        --out /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/phonological_vectors/buckeye_segments.jsonl \
        --split test

The train-df must contain columns feat/ipa/l_1/r_1/audio_path/min as
documented in `Segmenter.__init__`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
import librosa

from src.data.segmentation.segmentation_dataset import build_segmentation_dataset
from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)
from src.model.phonvec.model import FRAME_SHIFT, SR, Segmenter
from src.model.wavlm.builders import build_wavlm_model
from src.recipe.common.boundary_utils import boundaries_to_units


class _DummyTokenizer:
    """HACKY(shikhar): Satisfies SegmentationDataset's tokenizer contract.
    
    Phonvec does not consume phone IDs; we only need something that returns
    a list of ints of the right length so the dataset's `__getitem__` runs.
    """

    def tokens2ids(self, phones):
        return list(range(len(phones)))


def _extract_wavlm_feats(wavlm, speech, device):
    """Run WavLM forward on one utterance. Returns (T_f, D) numpy + valid_len."""
    with torch.no_grad():
        feats, feat_lens = wavlm.encode(
            speech.unsqueeze(0).to(device),
            torch.tensor([speech.shape[0]], device=device),
        )
    vlen = int(feat_lens[0])
    return feats[0, :vlen].cpu().numpy(), vlen


def _pred_frames_to_units(pred_frames, vlen):
    """Convert phonvec's frame-index array into a SegmentationUnit list."""
    flags = [False] * vlen
    for p in pred_frames:
        pi = int(p)
        if 0 <= pi < vlen:
            flags[pi] = True
    return boundaries_to_units(flags, vlen, FRAME_SHIFT, SR)


def _gt_units(phone_timestamps):
    """Dataset item's `phone_timestamps` (seconds) → SegmentationUnit list."""
    return [
        SegmentationUnit(start=float(s), end=float(e))
        for s, e in phone_timestamps
    ]


def _maybe_dump_jsonl(out_path, preds_dict, gt_dict):
    if not out_path:
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for utt_id, pred_units in preds_dict.items():
            f.write(
                json.dumps(
                    {
                        "utt_id": utt_id,
                        "pred": [(u.start, u.end) for u in pred_units],
                        "gt": [(u.start, u.end) for u in gt_dict[utt_id]],
                    }
                )
                + "\n"
            )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-df", required=True, help="Path to pickled TIMIT train-phone DataFrame.")
    p.add_argument("--silence-detector", required=True, help="Path to silence-detector .joblib.")
    p.add_argument("--hf-repo", required=True, help="HuggingFace repo id for the eval dataset.")
    p.add_argument("--split", default="test", help="Dataset split to evaluate on.")
    p.add_argument("--wavlm-repo", default="microsoft/wavlm-large", help="WavLM HF repo id.")
    p.add_argument("--wavlm-layer", type=int, default=24, help="WavLM hidden-states layer index.")
    p.add_argument("--device", default="cuda", help="Torch device for WavLM forward.")
    p.add_argument("--tolerance-ms", type=int, default=20, help="Boundary match tolerance in ms.")
    p.add_argument("--cache-dir", default="exp/cache/hf", help="HuggingFace cache dir.")
    p.add_argument("--limit", type=int, default=None, help="Optional: only evaluate first N utterances.")
    p.add_argument("--out", default=None, help="Optional JSONL dump of per-utt predictions.")
    return p.parse_args()


def main():
    args = parse_args()

    for path in (args.train_df, args.silence_detector):
        if not os.path.isfile(path):
            sys.exit(f"Missing asset: {path}")

    print(f"[phonvec] loading training DataFrame from {args.train_df}")
    train_df = pd.read_pickle(args.train_df)
    print(f"[phonvec] initializing Segmenter (this fits phonological vectors)")
    segmenter = Segmenter(
        train_df, silence_detector_path=args.silence_detector,
    )

    print(f"[phonvec] loading {args.wavlm_repo} @ layer={args.wavlm_layer}")
    wavlm = (
        build_wavlm_model(
            hf_repo=args.wavlm_repo,
            encoder_layer=args.wavlm_layer,
            cache_dir=args.cache_dir,
        )
        .to(args.device)
        .eval()
    )

    feat_dim = wavlm.encoder_output_size()
    train_feat_dim = len(train_df[~train_df.feat.isna()].iloc[0].feat)
    assert feat_dim == train_feat_dim, (
        f"WavLM output dim {feat_dim} != train-df feat dim {train_feat_dim}; "
        "asset mismatch — re-extract train-df with the same WavLM variant/layer."
    )

    print(f"[phonvec] loading {args.hf_repo} [{args.split}]")
    dataset = build_segmentation_dataset(
        hf_repo=args.hf_repo,
        split=args.split,
        tokenizer=_DummyTokenizer(), # Hacky but works because phonvec model does not need ids
        cache_dir=args.cache_dir,
    )
    n_items = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f"[phonvec] evaluating {n_items} utterances")

    preds_dict, gt_dict = {}, {}
    for i in tqdm(range(n_items), desc="phonvec"):
        item = dataset[i]
        utt_id = item["utt_id"]
        speech = item["speech"]
        # resample to 32khz
        if SR!=16000:
            speech = torch.from_numpy(librosa.resample(speech.numpy(), orig_sr=16000, target_sr=SR)).float()
        wavlm_feats, vlen = _extract_wavlm_feats(
            wavlm, speech, args.device,
        )
        pred_frames = segmenter.segment(
            wavlm_feats, speech.numpy(),
        )
        preds_dict[utt_id] = _pred_frames_to_units(pred_frames, vlen)
        gt_dict[utt_id] = _gt_units(item["phone_timestamps"])

    evaluator = SegmentationEvaluator(tolerance_ms=args.tolerance_ms)
    results = evaluator.evaluate_batch(preds_dict, gt_dict)
    if not results:
        sys.exit("No utterances evaluated successfully.")
    evaluator.pretty_print(results)
    _maybe_dump_jsonl(args.out, preds_dict, gt_dict)


if __name__ == "__main__":
    main()
