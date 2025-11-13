import unittest
import numpy as np
from src.recipe.forced_alignment.metrics import AlignmentEvaluator, ForcedAlignmentData


def make_results(boundaries, labels=None):
    """Helper to convert (start, end) tuples into AlignmentResult list."""
    if labels is None:
        # Use simple integer labels if none are provided
        labels = list(range(len(boundaries)))
    return [
        ForcedAlignmentData(start=s, end=e, label=labels[i])
        for i, (s, e) in enumerate(boundaries)
    ]


class TestAlignmentEvaluator(unittest.TestCase):
    """Test cases for AlignmentEvaluator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = AlignmentEvaluator(tolerance_ms=20)
        np.random.seed(42)

    def test_perfect_alignment(self):
        """Test evaluation with perfect alignment."""
        boundaries = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)])
        results = self.evaluator.evaluate_boundaries(boundaries, boundaries)

        self.assertEqual(results["f1"], 1.0)
        self.assertEqual(results["precision"], 1.0)
        self.assertEqual(results["recall"], 1.0)
        self.assertAlmostEqual(results["pbe_mean"], 0.0)
        self.assertAlmostEqual(results["start_err_mean"], 0.0)
        self.assertAlmostEqual(results["end_err_mean"], 0.0)

    def test_small_errors_within_tolerance(self):
        """Test with errors within tolerance threshold."""
        gt_boundaries = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)])
        # Add 10ms error (within 20ms tolerance)
        pred_boundaries = make_results([(0.01, 0.11), (0.11, 0.21), (0.21, 0.31)])

        results = self.evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        self.assertEqual(results["f1"], 1.0)  # All within tolerance
        self.assertAlmostEqual(results["start_err_mean"], 10.0, delta=0.1)
        self.assertAlmostEqual(results["end_err_mean"], 10.0, delta=0.1)
        self.assertAlmostEqual(results["pbe_mean"], 10.0, delta=0.1)

    def test_errors_outside_tolerance(self):
        """Test with errors outside tolerance threshold."""
        gt_boundaries = make_results([(0.0, 0.1), (0.1, 0.2)])
        # Add 30ms error (outside 20ms tolerance)
        pred_boundaries = make_results([(0.03, 0.13), (0.13, 0.23)])

        results = self.evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        self.assertEqual(results["f1"], 0.0)  # All outside tolerance
        self.assertAlmostEqual(results["start_err_mean"], 30.0, delta=0.1)
        self.assertAlmostEqual(results["end_err_mean"], 30.0, delta=0.1)

    def test_mixed_errors(self):
        """Test with mix of errors within and outside tolerance."""
        gt_boundaries = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)])
        pred_boundaries = make_results(
            [
                (0.01, 0.11),  # 10ms error - within tolerance
                (0.13, 0.23),  # 30ms error - outside tolerance
                (0.205, 0.305),  # 5ms error - within tolerance
            ]
        )

        results = self.evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        # 2 out of 3 within tolerance
        self.assertAlmostEqual(results["f1"], 2 / 3, delta=0.01)

    def test_duration_errors(self):
        """Test duration error calculation."""
        # 100ms and 200ms durations
        gt_boundaries = make_results([(0.0, 0.1), (0.2, 0.4)])
        # 120ms and 150ms durations
        pred_boundaries = make_results([(0.0, 0.12), (0.2, 0.35)])

        results = self.evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        # Duration errors: 20ms and 50ms -> mean = 35ms
        self.assertAlmostEqual(results["dur_err_mean"], 35.0, delta=0.1)
        self.assertAlmostEqual(results["gt_dur_mean"], 150.0, delta=0.1)  # (100+200)/2
        self.assertAlmostEqual(
            results["pred_dur_mean"], 135.0, delta=0.1
        )  # (120+150)/2

    def test_symbol_wise_analysis(self):
        """Test per-symbol error analysis."""
        gt_boundaries = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)])
        pred_boundaries = make_results(
            [(0.01, 0.11), (0.11, 0.21), (0.21, 0.31), (0.31, 0.41)]
        )
        symbols = ["AH", "T", "AH", "T"]

        results = self.evaluator.evaluate_boundaries(
            pred_boundaries, gt_boundaries, symbols
        )

        self.assertIn("symbol_errors", results)
        symbol_stats = results["symbol_errors"]

        # Check that we have stats for both symbols
        self.assertIn("AH", symbol_stats)
        self.assertIn("T", symbol_stats)

        # Each symbol appears twice
        self.assertEqual(symbol_stats["AH"]["count"], 2)
        self.assertEqual(symbol_stats["T"]["count"], 2)

        # All errors should be similar (10ms start, 10ms end)
        self.assertAlmostEqual(symbol_stats["AH"]["pbe_mean"], 10.0, delta=0.1)
        self.assertAlmostEqual(symbol_stats["T"]["pbe_mean"], 10.0, delta=0.1)

    def test_mismatched_lengths(self):
        """Test handling of mismatched prediction and ground truth lengths."""
        gt_boundaries = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)])
        pred_boundaries = make_results([(0.0, 0.1), (0.1, 0.2)])  # Missing last phoneme

        results = self.evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        # Should process only 2 phonemes
        self.assertEqual(results["n"], 2)

    def test_empty_input(self):
        """Test with empty input."""
        results = self.evaluator.evaluate_boundaries([], [])
        self.assertEqual(results, {})

    def test_percentiles(self):
        """Test percentile calculations."""
        # Create data with known distribution
        n = 100
        gt_boundaries = make_results([(i * 0.1, (i + 1) * 0.1) for i in range(n)])
        # Add varying errors from 0 to 50ms
        errors = np.linspace(0, 0.05, n)
        pred_boundaries = make_results(
            [
                (s + errors[i], e + errors[i])
                for i, (s, e) in enumerate([(i * 0.1, (i + 1) * 0.1) for i in range(n)])
            ]
        )

        results = self.evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        # Check that percentiles are calculated
        self.assertIn("pbe_p5", results)
        self.assertIn("pbe_p95", results)

        # P5 should be close to 0ms, P95 close to 50ms
        self.assertLess(results["pbe_p5"], 5.0)
        self.assertGreater(results["pbe_p95"], 45.0)

    def test_batch_evaluation(self):
        """Test batch evaluation."""
        batch_pred = {
            "seg1": make_results([(0.01, 0.11), (0.11, 0.21)]),
            "seg2": make_results([(0.02, 0.12), (0.12, 0.22)]),
        }
        batch_gt = {
            "seg1": make_results([(0.0, 0.1), (0.1, 0.2)]),
            "seg2": make_results([(0.0, 0.1), (0.1, 0.2)]),
        }

        results = self.evaluator.evaluate_batch(batch_pred, batch_gt)

        self.assertEqual(results["total_segments"], 2)
        self.assertEqual(results["total_samples"], 4)  # 2 segments * 2 phonemes
        self.assertIn("mean_f1", results)

    def test_batch_with_symbols(self):
        """Test batch evaluation with symbol analysis."""
        batch_pred = {
            "seg1": make_results([(0.01, 0.11), (0.11, 0.21)]),
            "seg2": make_results([(0.02, 0.12), (0.12, 0.22)]),
        }
        batch_gt = {
            "seg1": make_results([(0.0, 0.1), (0.1, 0.2)]),
            "seg2": make_results([(0.0, 0.1), (0.1, 0.2)]),
        }
        batch_symbols = {
            "seg1": ["AH", "T"],
            "seg2": ["AH", "K"],
        }

        results = self.evaluator.evaluate_batch(batch_pred, batch_gt, batch_symbols)

        self.assertIn("symbol_errors", results)
        self.assertEqual(results["symbol_errors"]["AH"]["count"], 2)
        self.assertEqual(results["symbol_errors"]["T"]["count"], 1)
        self.assertEqual(results["symbol_errors"]["K"]["count"], 1)

    def test_pretty_print_minimal(self):
        """Test minimal verbosity printing."""
        boundaries = make_results([(0.0, 0.1), (0.1, 0.2)])
        results = self.evaluator.evaluate_boundaries(boundaries, boundaries)

        # Should not raise any exceptions
        self.evaluator.pretty_print(results, verbosity=0)

    def test_pretty_print_all_levels(self):
        """Test all verbosity levels."""
        gt_boundaries = make_results([(i * 0.1, (i + 1) * 0.1) for i in range(10)])
        pred_boundaries = make_results(
            [
                (s + 0.01, e + 0.01)
                for s, e in [(i * 0.1, (i + 1) * 0.1) for i in range(10)]
            ]
        )
        symbols = ["AH", "T"] * 5

        results = self.evaluator.evaluate_boundaries(
            pred_boundaries, gt_boundaries, symbols
        )

        # Test all verbosity levels - should not raise exceptions
        for v in range(4):
            self.evaluator.pretty_print(results, verbosity=v)

    def test_statistics_calculation(self):
        """Test statistical calculations are correct."""
        # Create data with known statistics
        gt_boundaries = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)])
        # Errors: 0ms, 10ms, 20ms -> mean=10, std=8.16
        pred_boundaries = make_results([(0.0, 0.1), (0.11, 0.21), (0.22, 0.32)])

        results = self.evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        self.assertAlmostEqual(results["start_err_mean"], 10.0, delta=0.1)
        self.assertAlmostEqual(results["start_err_std"], 8.16, delta=0.1)
        self.assertAlmostEqual(results["start_err_median"], 10.0, delta=0.1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_negative_times(self):
        """Test handling of negative timestamps."""
        evaluator = AlignmentEvaluator()
        gt_boundaries = make_results([(-0.1, 0.0), (0.0, 0.1)])
        pred_boundaries = make_results([(-0.09, 0.01), (0.01, 0.11)])

        results = evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        # Should handle negative times correctly
        self.assertAlmostEqual(results["start_err_mean"], 10.0, delta=0.1)

    def test_zero_duration_phonemes(self):
        """Test handling of zero-duration phonemes."""
        evaluator = AlignmentEvaluator()
        gt_boundaries = make_results(
            [(0.0, 0.0), (0.1, 0.2)]
        )  # First phoneme has 0 duration
        pred_boundaries = make_results([(0.0, 0.01), (0.1, 0.2)])

        results = evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        # Should handle zero durations
        self.assertIn("dur_err_mean", results)

    def test_overlapping_boundaries(self):
        """Test with overlapping phoneme boundaries."""
        evaluator = AlignmentEvaluator()
        # Overlapping boundaries (end > next start)
        gt_boundaries = make_results([(0.0, 0.15), (0.1, 0.2)])
        pred_boundaries = make_results([(0.0, 0.15), (0.1, 0.2)])

        results = evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        # Should still compute metrics despite overlap
        self.assertEqual(results["f1"], 1.0)

    def test_very_long_phonemes(self):
        """Test with unusually long phonemes."""
        evaluator = AlignmentEvaluator()
        gt_boundaries = make_results([(0.0, 2.0), (2.0, 2.1)])  # 2 second phoneme
        pred_boundaries = make_results([(0.01, 2.01), (2.01, 2.11)])

        results = evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        self.assertAlmostEqual(
            results["gt_dur_mean"], 1050.0, delta=0.1
        )  # (2000+100)/2

    def test_unicode_symbols(self):
        """Test with Unicode phoneme symbols."""
        evaluator = AlignmentEvaluator()
        gt_boundaries = make_results([(0.0, 0.1), (0.1, 0.2)])
        pred_boundaries = make_results([(0.01, 0.11), (0.11, 0.21)])
        symbols = ["😀", "中文"]

        results = evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries, symbols)

        self.assertIn("symbol_errors", results)
        self.assertIn("😀", results["symbol_errors"])
        self.assertIn("中文", results["symbol_errors"])


class TestTablePrinting(unittest.TestCase):
    """Test ASCII table formatting."""

    def test_table_with_different_widths(self):
        """Test table printing with varying column widths."""
        evaluator = AlignmentEvaluator()
        rows = [
            ["Short", "Value"],
            ["Very Long Metric Name", "123.456"],
            ["X", "1"],
        ]

        # Should handle varying widths without error
        evaluator._print_table(rows)

    def test_empty_table(self):
        """Test printing empty table."""
        evaluator = AlignmentEvaluator()
        evaluator._print_table([])  # Should not raise error


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
