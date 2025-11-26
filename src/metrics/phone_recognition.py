"""Evaluate phone recognition output using panphon feature-based metrics.

Usage:
    python -m src.metrics.phone_recognition \
        --prediction_file /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_evals/runs/20251115_193559/l2arctic_perceived_powsm_out.json
"""

import string
import unicodedata
from dataclasses import dataclass
from typing import Dict, Tuple, Any
from tqdm import tqdm

import panphon.distance
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class PhoneRecognitionSummary:
    """Aggregate metrics for a phone recognition experiment."""

    PFER: float
    FER: float
    FED: float
    PER: float
    N: int  # number of utterances
    phones: int  # total number of reference phones


class PhoneRecognitionEvaluator:
    """
    Evaluates phone recognition output using panphon feature-based metrics.

      * PER (phone error rate, %)
      * FER (feature error rate, %)
      * FED (total feature edit distance)
      * PFER (phone feature error rate averaged per utterance)
      * per-utterance metrics

    Assumes `test_data` is a dict:
        { utt_id: {"prediction": str, "transcription": str, ...}, ... }
    """

    def __init__(self, normalize_ipa: bool = True):
        self.normalize_ipa = normalize_ipa
        self.dst = panphon.distance.Distance()

    @staticmethod
    def clean_text(s: str) -> str:
        """Normalize IPA text: remove spaces/punct, NFC->NFD, fix 'g'→'ɡ'."""
        s = s.replace(" ", "").translate(str.maketrans("", "", string.punctuation))
        s = unicodedata.normalize("NFD", s)
        return s.replace("g", "ɡ").strip()

    def _prepare(self, text: str) -> str:
        return self.clean_text(text) if self.normalize_ipa else text

    def _compute_utterance_metrics(
        self, hyp: str, ref: str
    ) -> Tuple[Dict[str, float], int, int, int, int]:
        """
        Compute metrics for a single utterance.

        Returns:
            (metrics_dict, pfer, fed, per_errors, n_phones)
        """
        hyp = self._prepare(hyp)
        ref = self._prepare(ref)

        # Phone feature distances
        pfer = self.dst.hamming_feature_edit_distance(hyp, ref)
        fed = self.dst.feature_edit_distance(hyp, ref)

        # PER via min_edit_distance over IPA segments
        hyp_segs = self.dst.fm.ipa_segs(hyp)
        ref_segs = self.dst.fm.ipa_segs(ref)
        n_phones = len(ref_segs)

        per_errors = self.dst.min_edit_distance(
            lambda v: 1,  # deletion cost
            lambda v: 1,  # insertion cost
            lambda x, y: 0 if x == y else 1,  # substitution cost
            [[]],  # inventory (kept as in original script)
            hyp_segs,
            ref_segs,
        )

        metrics = {
            "pfer": float(pfer),
            "fed": float(fed),
            "per": float(per_errors / n_phones * 100) if n_phones > 0 else 0.0,
            "fer": float(fed / n_phones * 100) if n_phones > 0 else 0.0,
        }
        return metrics, pfer, fed, per_errors, n_phones

    def evaluate(
        self, test_data: Dict[str, Dict[str, Any]]
    ) -> Tuple[PhoneRecognitionSummary, Dict[str, Dict[str, float]]]:
        """
        Evaluate a full dataset.

        Args:
            test_data: mapping from utt_id -> {"prediction": ..., "transcription": ...}

        Returns:
            summary: PhoneRecognitionSummary (aggregate metrics)
            instance_metrics: per-utterance metrics, same keys as original script:
                             {utt_id: {"pfer":..., "fed":..., "per":..., "fer":...}}
        """
        if not test_data:
            empty_summary = PhoneRecognitionSummary(
                PFER=0.0, FER=0.0, FED=0.0, PER=0.0, N=0, phones=0
            )
            return empty_summary, {}

        instance_metrics: Dict[str, Dict[str, float]] = {}

        pfer_sum = 0.0
        fed_sum = 0.0
        per_err_sum = 0.0
        phones_sum = 0
        n_utts = 0

        for utt_id, sample in tqdm(
            test_data.items(), total=len(test_data), desc="Evaluating"
        ):
            hyp = sample.get("prediction", "")
            ref = sample.get("transcription", "")

            metrics, pfer, fed, per_errors, n_phones = self._compute_utterance_metrics(
                hyp, ref
            )

            instance_metrics[utt_id] = metrics
            pfer_sum += pfer
            fed_sum += fed
            per_err_sum += per_errors
            phones_sum += n_phones
            n_utts += 1

        summary = PhoneRecognitionSummary(
            PFER=pfer_sum / n_utts if n_utts > 0 else 0.0,
            FER=(fed_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            FED=fed_sum,
            PER=(per_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            N=n_utts,
            phones=phones_sum,
        )

        return summary, instance_metrics

    def pretty_print(
        self,
        summary: PhoneRecognitionSummary,
        model_name: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        """Simple ASCII summary, no verbosity levels."""
        title_parts = []
        if model_name:
            title_parts.append(model_name)
        if dataset_name:
            title_parts.append(f"on {dataset_name}")

        title = " ".join(title_parts) if title_parts else "Phone Recognition Results"

        print("\n" + title)
        print("=" * max(len(title), 30))

        rows = [
            ["Metric", "Value"],
            ["-" * 15, "-" * 20],
            ["Utterances (N)", f"{summary.N}"],
            ["Total Phones", f"{summary.phones}"],
            ["PFER (avg per utt)", f"{summary.PFER:.4f}"],
            ["FER (%)", f"{summary.FER:.2f}"],
            ["FED (total)", f"{summary.FED:.2f}"],
            ["PER (%)", f"{summary.PER:.2f}"],
        ]

        col_widths = [
            max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))
        ]
        for row in rows:
            print(" | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)))
        print()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", required=True)
    args = parser.parse_args()

    def _load_predictions(pred_file: str) -> Dict[str, Dict[str, str]]:
        with open(pred_file, "r") as f:
            data = json.load(f)
        return {
            item["passthrough"]["key"]: {
                "prediction": item["pred"][0]["processed_transcript"],
                "transcription": item["passthrough"]["text"],
            }
            for _, item in data.items()
        }

    # Load predictions
    test_data = _load_predictions(args.prediction_file)
    log.info(f"Loaded predictions for {len(test_data)} utterances.")

    # Evaluate
    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
    summary, instance_metrics = evaluator.evaluate(test_data)
    evaluator.pretty_print(summary, model_name="dummy-model", dataset_name="dummy-set")
