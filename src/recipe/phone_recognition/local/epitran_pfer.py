"""
Compute phone recognition metrics (PER / FER) for Epitran outputs.

Usage:

    python -m src.recipe.phone_recognition.local.epitran_pfer \
        --ref /path/to/text.good \
        --hyp exp/data/epitran_outputs/buckeye.epitran

With CSV writing:

    python -m src.recipe.phone_recognition.local.epitran_pfer \
        --ref /path/to/text.good \
        --hyp exp/data/epitran_outputs/buckeye.epitran \
        --evaluation_name epitran-buckeye \
        --output_file exp/runs/ipapack_ctc/results-epitran.csv
"""

import argparse
from src.metrics.phone_recognition import PhoneRecognitionEvaluator


def read_kaldi_text(path):
    """Read Kaldi-style text file: utt_id phone1 phone2 ..."""
    data = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            utt_id = parts[0]
            phones = "".join(parts[1:])  # join IPA without spaces
            data[utt_id] = phones
    return data


def main(args):
    ref = read_kaldi_text(args.ref)
    hyp = read_kaldi_text(args.hyp)

    # Build evaluator input format
    test_data = {
        utt: {
            "prediction": hyp.get(utt, ""),
            "transcription": ref[utt],
        }
        for utt in ref
        if utt in hyp
    }

    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
    summary, _ = evaluator.evaluate(test_data, compute_inventory=False)

    # ---- Print summary ----
    print("=" * 40)
    print(f"N utterances: {summary.N}")
    print(f"Total phones: {summary.phones}")
    print(f"PER (%): {summary.PER:.2f}")
    print(f"FER (%): {summary.FER:.2f}")
    print("=" * 40)

    # ---- Optional CSV writing ----
    if args.output_file:
        if not args.evaluation_name:
            raise ValueError("Must provide --evaluation_name when using --output_file")

        evaluator.write_to_csv(
            summary=summary,
            evalname=args.evaluation_name,
            output_file=args.output_file,
            language="combined",
        )

        print(f"Appended results to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Epitran outputs against reference (PER / FER)."
    )

    parser.add_argument(
        "--ref", required=True, help="Path to reference Kaldi text file"
    )
    parser.add_argument(
        "--hyp",
        required=True,
        help="Path to hypothesis Kaldi text file (Epitran outputs)",
    )
    parser.add_argument(
        "--evaluation_name",
        type=str,
        help="Name for this evaluation (required if writing to CSV)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="CSV file to append results to",
    )

    args = parser.parse_args()
    main(args)
