import unittest
import numpy as np
from src.metrics.segmentation_evaluator import SegmentationEvaluator, SegmentationUnit


def make_results(boundaries, labels=None):
    """Helper to convert (start, end) tuples into AlignmentResult list."""
    if labels is None:
        # Use simple integer labels if none are provided
        labels = list(range(len(boundaries)))
    return [
        SegmentationUnit(start=s, end=e, label=labels[i])
        for i, (s, e) in enumerate(boundaries)
    ]


class TestAlignmentEvaluator(unittest.TestCase):
    """Test cases for AlignmentEvaluator class (forced mode)."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = SegmentationEvaluator(tolerance_ms=20, forced=True)
        np.random.seed(42)

    def test_perfect_alignment(self):
        """Test evaluation with perfect alignment."""
        boundaries = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)])
        results = self.evaluator.evaluate_boundaries(boundaries, boundaries)

        self.assertAlmostEqual(results["f1"], 1.0, delta=1e-5)
        self.assertAlmostEqual(results["precision"], 1.0, delta=1e-5)
        self.assertAlmostEqual(results["recall"], 1.0, delta=1e-5)
        self.assertAlmostEqual(results["pbe_mean"], 0.0)
        self.assertAlmostEqual(results["start_err_mean"], 0.0)
        self.assertAlmostEqual(results["end_err_mean"], 0.0)

    def test_small_errors_within_tolerance(self):
        """Test with errors within tolerance threshold."""
        gt_boundaries = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)])
        # Add 10ms error (within 20ms tolerance)
        pred_boundaries = make_results([(0.01, 0.11), (0.11, 0.21), (0.21, 0.31)])

        results = self.evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        self.assertAlmostEqual(results["f1"], 1.0, delta=1e-5)  # All within tolerance
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
        self.assertAlmostEqual(results["f1"], 0.75, delta=0.01)

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

        # Should process only 2 phonemes for error metrics
        self.assertEqual(results["n"], 2)
        # pred_times=[0.0, 0.1], gt_times=[0.0, 0.1, 0.2]; both preds match, gt[2] unmatched
        self.assertEqual(results["n"], 2)
        self.assertAlmostEqual(results["precision"], 1.0, delta=0.001)
        self.assertAlmostEqual(results["recall"], 0.75, delta=0.001)
        self.assertAlmostEqual(results["f1"], 0.857142857, delta=0.001)

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
        self.evaluator.pretty_print(results)

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

        # Should not raise exceptions
        self.evaluator.pretty_print(results)

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

    def test_forced_mode_has_rval(self):
        """Forced mode should include rval alongside other keys."""
        boundaries = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)])
        results = self.evaluator.evaluate_boundaries(boundaries, boundaries)

        self.assertIn("rval", results)
        self.assertAlmostEqual(results["rval"], 1.0, delta=0.01)

    def test_forced_mode_rval_present_in_batch(self):
        """Batch forced mode results should include mean_rval."""
        batch_pred = {"seg1": make_results([(0.01, 0.11), (0.11, 0.21)])}
        batch_gt = {"seg1": make_results([(0.0, 0.1), (0.1, 0.2)])}
        results = self.evaluator.evaluate_batch(batch_pred, batch_gt)

        self.assertIn("mean_rval", results)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_negative_times(self):
        """Test handling of negative timestamps."""
        evaluator = SegmentationEvaluator(forced=True)
        gt_boundaries = make_results([(-0.1, 0.0), (0.0, 0.1)])
        pred_boundaries = make_results([(-0.09, 0.01), (0.01, 0.11)])

        results = evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        # Should handle negative times correctly
        self.assertAlmostEqual(results["start_err_mean"], 10.0, delta=0.1)

    def test_zero_duration_phonemes(self):
        """Test handling of zero-duration phonemes."""
        evaluator = SegmentationEvaluator(forced=True)
        gt_boundaries = make_results(
            [(0.0, 0.0), (0.1, 0.2)]
        )  # First phoneme has 0 duration
        pred_boundaries = make_results([(0.0, 0.01), (0.1, 0.2)])

        results = evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        # Should handle zero durations
        self.assertIn("dur_err_mean", results)

    def test_overlapping_boundaries(self):
        """Test with overlapping phoneme boundaries."""
        evaluator = SegmentationEvaluator(forced=True)
        # Overlapping boundaries (end > next start)
        gt_boundaries = make_results([(0.0, 0.15), (0.1, 0.2)])
        pred_boundaries = make_results([(0.0, 0.15), (0.1, 0.2)])

        results = evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        # Should still compute metrics despite overlap
        self.assertAlmostEqual(results["f1"], 1.0, delta=1e-5)

    def test_very_long_phonemes(self):
        """Test with unusually long phonemes."""
        evaluator = SegmentationEvaluator(forced=True)
        gt_boundaries = make_results([(0.0, 2.0), (2.0, 2.1)])  # 2 second phoneme
        pred_boundaries = make_results([(0.01, 2.01), (2.01, 2.11)])

        results = evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries)

        self.assertAlmostEqual(
            results["gt_dur_mean"], 1050.0, delta=0.1
        )  # (2000+100)/2

    def test_unicode_symbols(self):
        """Test with Unicode phoneme symbols."""
        evaluator = SegmentationEvaluator(forced=True)
        gt_boundaries = make_results([(0.0, 0.1), (0.1, 0.2)])
        pred_boundaries = make_results([(0.01, 0.11), (0.11, 0.21)])
        symbols = ["😀", "中文"]

        results = evaluator.evaluate_boundaries(pred_boundaries, gt_boundaries, symbols)

        self.assertIn("symbol_errors", results)
        self.assertIn("😀", results["symbol_errors"])
        self.assertIn("中文", results["symbol_errors"])


class TestFreeModeEvaluator(unittest.TestCase):
    """Tests for free mode (forced=False, default)."""

    def setUp(self):
        self.evaluator = SegmentationEvaluator(tolerance_ms=20)  # forced=False default

    def test_free_mode_is_default(self):
        """SegmentationEvaluator() should default to free mode."""
        self.assertFalse(self.evaluator.forced)

    def test_free_mode_different_lengths_no_raise(self):
        """Free mode should accept pred/GT lists of different length."""
        gt = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)])
        pred = make_results([(0.0, 0.15), (0.15, 0.3), (0.3, 0.4)])  # 3 vs 4

        # Should not raise
        results = self.evaluator.evaluate_boundaries(pred, gt)
        self.assertIn("f1", results)

    def test_free_mode_returns_boundary_keys_only(self):
        """Free mode results should contain boundary metrics and no per-phone stats."""
        gt = make_results([(0.0, 0.1), (0.1, 0.2)])
        pred = make_results([(0.005, 0.105), (0.105, 0.205)])

        results = self.evaluator.evaluate_boundaries(pred, gt)

        self.assertIn("n_pred", results)
        self.assertIn("n_gt", results)
        self.assertIn("precision", results)
        self.assertIn("recall", results)
        self.assertIn("f1", results)
        self.assertIn("rval", results)
        # Per-phone stats should NOT be present
        self.assertNotIn("start_err_mean", results)
        self.assertNotIn("pbe_mean", results)
        self.assertNotIn("n", results)

    def test_free_mode_perfect_match(self):
        """Identical pred and GT should give P=R=F1=Rval=1.0 in free mode."""
        segs = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)])
        results = self.evaluator.evaluate_boundaries(segs, segs)

        self.assertAlmostEqual(results["precision"], 1.0, delta=1e-5)
        self.assertAlmostEqual(results["recall"], 1.0, delta=1e-5)
        self.assertAlmostEqual(results["f1"], 1.0, delta=1e-5)
        self.assertAlmostEqual(results["rval"], 1.0, delta=1e-3)

    def test_free_mode_toy_boundary_values(self):
        """Verify boundary P/R/F1 for a toy example with known expected values.

        GT boundaries: 0.0, 0.1, 0.2 (from 2 segments: [0,0.1], [0.1,0.2])
        Pred boundaries: 0.0, 0.15, 0.3 (from 2 segments: [0,0.15], [0.15,0.3])

        tolerance = 20ms = 0.02s
        precision: 0.0 → nearest GT=0.0, dist=0 ✓; 0.15 → nearest=0.1, dist=0.05 ✗;
                   0.3 → nearest=0.2, dist=0.1 ✗ → precision=1/3
        recall:    0.0 → nearest pred=0.0 ✓; 0.1 → nearest=0.15, dist=0.05 ✗;
                   0.2 → nearest=0.15, dist=0.05 ✗ → recall=1/3
        """
        gt = make_results([(0.0, 0.1), (0.1, 0.2)])
        pred = make_results([(0.0, 0.15), (0.15, 0.3)])

        results = self.evaluator.evaluate_boundaries(pred, gt)

        self.assertAlmostEqual(results["precision"], 1 / 3, delta=0.01)
        self.assertAlmostEqual(results["recall"], 1 / 3, delta=0.01)

    def test_free_mode_rval_range(self):
        """R-value should be in a reasonable range [-1, 1]."""
        gt = make_results([(0.0, 0.1), (0.1, 0.3), (0.3, 0.5)])
        pred = make_results([(0.05, 0.2), (0.2, 0.4)])

        results = self.evaluator.evaluate_boundaries(pred, gt)

        self.assertGreaterEqual(results["rval"], -1.0)
        self.assertLessEqual(results["rval"], 1.0)

    def test_free_mode_empty_returns_empty(self):
        """Empty input should return {}."""
        results = self.evaluator.evaluate_boundaries([], [])
        self.assertEqual(results, {})

    def test_free_mode_n_pred_n_gt(self):
        """n_pred and n_gt should reflect segment counts, not boundary counts."""
        gt = make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)])  # 3 segments
        pred = make_results([(0.0, 0.2), (0.2, 0.3)])  # 2 segments

        results = self.evaluator.evaluate_boundaries(pred, gt)

        self.assertEqual(results["n_pred"], 2)
        self.assertEqual(results["n_gt"], 3)

    def test_free_mode_batch_no_raise(self):
        """Free mode batch evaluation should handle different-length segments."""
        batch_pred = {
            "seg1": make_results([(0.0, 0.2), (0.2, 0.4)]),  # 2 segs
            "seg2": make_results([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)]),  # 3 segs
        }
        batch_gt = {
            "seg1": make_results(
                [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
            ),  # 4 segs
            "seg2": make_results([(0.0, 0.15), (0.15, 0.3)]),  # 2 segs
        }

        results = self.evaluator.evaluate_batch(batch_pred, batch_gt)

        self.assertIn("mean_f1", results)
        self.assertIn("mean_rval", results)
        self.assertEqual(results["total_segments"], 2)
        # total_samples uses n_gt: 4 + 2 = 6
        self.assertEqual(results["total_samples"], 6)

    def test_free_mode_pretty_print_no_per_phone_rows(self):
        """pretty_print in free mode should not raise even without per-phone stats."""
        gt = make_results([(0.0, 0.1), (0.1, 0.2)])
        pred = make_results([(0.005, 0.105), (0.105, 0.205)])
        results = self.evaluator.evaluate_boundaries(pred, gt)

        # Should not raise
        self.evaluator.pretty_print(results)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
