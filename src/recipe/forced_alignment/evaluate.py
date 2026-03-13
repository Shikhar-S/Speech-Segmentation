"""Script to read the dumped predictions for forced alignment and evaluate and print the metrics.

Usage:
    python -m src.recipe.forced_alignment.evaluate \
        --prediction_file <path_to_prediction_file>
"""

import argparse
import os
import json
from tqdm import tqdm

from src.metrics.segmentation_evaluator import SegmentationEvaluator, SegmentationUnit


def load_fa_predictions(file_path):
    with open(file_path, "r") as f:
        predictions = json.load(f)
    return predictions


def evaluate_fa_predictions(predictions):
    print("Evaluating forced alignment predictions")
    evaluator = SegmentationEvaluator(tolerance_ms=20)
    fa_predictions = {}
    ground_truth = {}
    for example_id, pred in tqdm(
        predictions.items(), desc="Evaluating FA predictions", total=len(predictions)
    ):
        fa_boundaries = [
            SegmentationUnit(
                unit["start_time"],
                unit["end_time"],
                unit["label"],
            )
            for unit in pred["predicted_units"]
        ]
        gt_boundaries = [
            SegmentationUnit(
                unit["start_time"],
                unit["end_time"],
                unit["label"],
            )
            for unit in pred["ground_truth_units"]
        ]
        fa_predictions[example_id] = fa_boundaries
        ground_truth[example_id] = gt_boundaries

    # Evaluate
    metrics = evaluator.evaluate_batch(fa_predictions, ground_truth)
    print("\nEvaluation metrics:")
    evaluator.pretty_print(metrics, verbosity=2)
    predictions = load_fa_predictions(args.prediction_file)
    evaluate_fa_predictions(predictions)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate forced alignment predictions."
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        help="Path to the JSON file containing predictions.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.prediction_file):
        print(f"Error: File {args.prediction_file} does not exist.")
        return

    predictions = load_fa_predictions(args.prediction_file)
    evaluate_fa_predictions(predictions)


if __name__ == "__main__":
    main()
