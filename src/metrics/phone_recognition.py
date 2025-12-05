"""Evaluate phone recognition output using panphon feature-based metrics.

Usage:
    python -m src.metrics.phone_recognition \
        --prediction_file something.json \
        --noisy_pr # for noisy phone recognition
"""

import string
import unicodedata
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Union
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
    SUB: float
    INS: float
    DEL: float
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

    def _compute_sid_metrics(self, hyp: str, ref: str) -> Tuple[int, int, int]:
        """Calculates substitution, insertion, deletion rates on phones."""
        sub_errors = ins_errors = del_errors = 0
        # dp
        Hlen = len(hyp) + 1
        Rlen = len(ref) + 1
        D = [[0] * Rlen for _ in range(Hlen)]
        for hi in range(Hlen):
            D[hi][0] = hi
        for rj in range(Rlen):
            D[0][rj] = rj
        for hi in range(1, Hlen):
            for rj in range(1, Rlen):
                cost = 0 if hyp[hi - 1] == ref[rj - 1] else 1
                D[hi][rj] = min(
                    D[hi - 1][rj] + 1,
                    D[hi][rj - 1] + 1,
                    D[hi - 1][rj - 1] + cost,
                )
        # backtrack
        hi = Hlen - 1
        rj = Rlen - 1
        while hi > 0 or rj > 0:
            if (
                hi > 0
                and rj > 0
                and D[hi][rj] == D[hi - 1][rj - 1]
                and hyp[hi - 1] == ref[rj - 1]
            ):
                hi -= 1
                rj -= 1
            elif hi > 0 and rj > 0 and D[hi][rj] == D[hi - 1][rj - 1] + 1:
                sub_errors += 1
                hi -= 1
                rj -= 1
            elif rj > 0 and D[hi][rj] == D[hi][rj - 1] + 1:
                del_errors += 1
                rj -= 1
            else:
                ins_errors += 1
                hi -= 1
        return sub_errors, ins_errors, del_errors

    def _compute_utterance_metrics(
        self, hyp: str, ref: str
    ) -> Tuple[Dict[str, Union[int, float]]]:
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
            [[]],  # start
            hyp_segs,
            ref_segs,
        )
        sub_errors, ins_errors, del_errors = self._compute_sid_metrics(
            hyp_segs, ref_segs
        )

        metrics = {
            "pfer": float(pfer),
            "fed": float(fed),
            "per": float(per_errors / n_phones * 100) if n_phones > 0 else 0.0,
            "fer": float(fed / n_phones * 100) if n_phones > 0 else 0.0,
        }
        out = {
            "metrics": metrics,
            "pfer": pfer,
            "fed": fed,
            "per_errors": per_errors,
            "sub_errors": sub_errors,
            "ins_errors": ins_errors,
            "del_errors": del_errors,
            "n_phones": n_phones,
        }
        return out

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
                PFER=0.0,
                FER=0.0,
                FED=0.0,
                PER=0.0,
                N=0,
                phones=0,
                SUB=0.0,
                INS=0.0,
                DEL=0.0,
            )
            return empty_summary, {}

        instance_metrics: Dict[str, Dict[str, float]] = {}

        pfer_sum = 0.0
        fed_sum = 0.0
        per_err_sum = 0.0
        phones_sum = 0
        n_utts = 0
        sub_err_sum = 0
        ins_err_sum = 0
        del_err_sum = 0

        for utt_id, sample in tqdm(
            test_data.items(), total=len(test_data), desc="Evaluating"
        ):
            hyp = sample.get("prediction", "")
            ref = sample.get("transcription", "")

            out = self._compute_utterance_metrics(hyp, ref)

            instance_metrics[utt_id] = out["metrics"]
            pfer_sum += out["pfer"]
            fed_sum += out["fed"]
            per_err_sum += out["per_errors"]
            phones_sum += out["n_phones"]
            sub_err_sum += out["sub_errors"]
            ins_err_sum += out["ins_errors"]
            del_err_sum += out["del_errors"]
            n_utts += 1

        summary = PhoneRecognitionSummary(
            PFER=pfer_sum / n_utts if n_utts > 0 else 0.0,
            FER=(fed_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            FED=fed_sum,
            PER=(per_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            SUB=(sub_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            INS=(ins_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            DEL=(del_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            N=n_utts,
            phones=phones_sum,
        )

        return summary, instance_metrics

    def pretty_print(
        self,
        summary: PhoneRecognitionSummary,
    ) -> None:
        """Simple ASCII summary, no verbosity levels."""
        title = "Phone Recognition Results"
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
            ["SUB (%)", f"{summary.SUB:.2f}"],
            ["INS (%)", f"{summary.INS:.2f}"],
            ["DEL (%)", f"{summary.DEL:.2f}"],
        ]

        col_widths = [
            max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))
        ]
        for row in rows:
            print(" | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)))
        print()

    def write_to_csv(
        self, summary: PhoneRecognitionSummary, evalname: str, output_file: str
    ) -> None:
        """Write summary metrics to a CSV file."""
        import csv

        with open(output_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "eval_name",
                    "N",
                    "Total Phones",
                    "PFER",
                    "FER (%)",
                    "FED",
                    "PER (%)",
                    "SUB (%)",
                    "INS (%)",
                    "DEL (%)",
                ]
            )
            writer.writerow(
                [
                    evalname,
                    summary.N,
                    summary.phones,
                    f"{summary.PFER:.4f}",
                    f"{summary.FER:.2f}",
                    f"{summary.FED:.2f}",
                    f"{summary.PER:.2f}",
                    f"{summary.SUB:.2f}",
                    f"{summary.INS:.2f}",
                    f"{summary.DEL:.2f}",
                ]
            )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument(
        "--gt_field",
        type=str,
        default="masked_phones",
        help="Field name for ground truth transcription in the prediction file",
    )
    parser.add_argument(
        "--pred_field",
        type=str,
        default="processed_transcript",
        help="Field name for predicted transcription in the prediction file",
    )
    parser.add_argument(
        "--key_field",
        type=str,
        default="utt_id",
        help="Field name for utterance ID in the prediction file",
    )
    parser.add_argument(
        "--noisy_pr",
        action="store_true",
        help="Whether to evaluate noisy phone recognition",
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="File to write results to"
    )
    parser.add_argument("--evaluation_name", type=str, help="name for the evaluation")
    args = parser.parse_args()

    def _load_predictions(pred_file: str) -> Dict[str, Dict[str, str]]:
        with open(pred_file, "r") as f:
            data = json.load(f)
        D = {
            item["passthrough"][args.key_field]: {
                "prediction": item["pred"][0][args.pred_field],
                "transcription": (
                    item["passthrough"][args.gt_field]
                    if not args.noisy_pr
                    else "".join(
                        [
                            n
                            for n in item["passthrough"]["masked_phones"]
                            if n != "[NOISE]"
                        ]
                    )
                ),
            }
            for _, item in data.items()
        }
        return D

    # Load predictions
    test_data = _load_predictions(args.prediction_file)
    log.info(f"Loaded predictions for {len(test_data)} utterances.")

    # Evaluate
    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
    summary, instance_metrics = evaluator.evaluate(test_data)
    evaluator.pretty_print(summary)

    # Write results to file
    if args.output_file:
        assert args.evaluation_name is not None, "Please provide --evaluation_name"
        evaluator.write_to_csv(summary, args.evaluation_name, args.output_file)
        log.info(f"Wrote results to {args.output_file}")
