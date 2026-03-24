"""Tests for SegmentationLoss and BoundaryLoss."""

import torch
import torch.nn as nn
import pytest
from src.recipe.segmentation.segmentation_loss import BoundaryLoss, SegmentationLoss
import torch.nn.functional as F


# -------------------------------------------------------------------
# GROUND TRUTH REFERENCE IMPLEMENTATION
# -------------------------------------------------------------------
class ReferenceSegmentationLoop(torch.nn.Module):
    """
    Slow, explicit loop-based implementation to establish Ground Truth.
    """

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets, target_start, target_end, input_lengths=None):
        """
        Args:
            logits: (B, S, C)
            targets: (B, T)
            target_start: (B, T)
            target_end: (B, T)
            input_lengths: (B,) - Valid audio frames
        """
        B, S, C = logits.size()
        T = targets.size(1)

        # 1. Construct Dense Probability Targets (Soft Labels)
        # G[b, s, c] = Probability of class c at frame s
        G = torch.zeros(B, S, C, device=logits.device)

        for b in range(B):
            valid_len = input_lengths[b] if input_lengths is not None else S

            for t in range(T):
                tgt_idx = targets[b, t].item()

                # Skip ignore_index
                if tgt_idx == self.ignore_index:
                    continue

                t_start = target_start[b, t].item()
                t_end = target_end[b, t].item()

                # Bounds check against Sequence Length (S) and Input Length (valid_len)
                if t_start >= valid_len:
                    continue

                actual_end = min(t_end, valid_len - 1)

                if actual_end < t_start:
                    continue

                # Additive assignment (handles overlaps)
                for s in range(t_start, actual_end + 1):
                    G[b, s, tgt_idx] += 1.0

        # 2. Normalize G to create valid probability distribution
        # If a frame has multiple targets, we average them (same as Sparse impl)
        G_sum = G.sum(dim=2, keepdim=True)
        G = G / G_sum.clamp_min(1.0)  # Avoid div by zero

        # 3. Compute Loss Manually (to ensure exact match with Sparse LogSoftmax)
        log_probs = F.log_softmax(logits, dim=-1)

        # Cross Entropy = - sum(P * log(Q))
        # sum over classes (dim 2)
        per_frame_loss = -(G * log_probs).sum(dim=2)

        # 4. Masking
        # Valid frames are those where G_sum > 0 (at least one target covered it)
        valid_positions = (G_sum.squeeze(-1) > 0).float()

        # The loss is summed over frames, then divided by count of valid frames
        loss = (
            per_frame_loss * valid_positions
        ).sum() / valid_positions.sum().clamp_min(1.0)

        return loss


# -------------------------------------------------------------------
# PYTEST FIXTURES AND RUNNER
# -------------------------------------------------------------------


@pytest.fixture
def loss_fn():
    """The optimized implementation under test."""
    # Ensure this imports your corrected class
    return SegmentationLoss(ignore_index=-100)


@pytest.fixture
def loss_fn_ref():
    """The ground truth reference."""
    return ReferenceSegmentationLoop(ignore_index=-100)


def run_comparison(
    loss_fn, loss_fn_ref, logits, targets, t_start, t_end, input_lengths=None
):
    """
    Helper to run both losses and assert equality.
    Adapts the arguments to fit the specific signatures.
    """

    # --- 1. Reference Pass ---
    logits_ref = logits.clone().detach().requires_grad_(True)
    loss_r = loss_fn_ref(logits_ref, targets, t_start, t_end, input_lengths)

    # --- 2. Sparse Pass (New Signature) ---
    logits_sparse = logits.clone().detach().requires_grad_(True)

    # Construct t_lens (Assume full length if not provided in test)
    # In these basic tests, we assume t_lens encompasses all targets provided
    t_lens = None  # The sparse implementation handles None as "All targets valid"

    # Ensure input_lengths exists for the sparse call
    if input_lengths is None:
        logit_lens = torch.tensor(
            [logits.size(1)] * logits.size(0), device=logits.device
        )
    else:
        logit_lens = input_lengths

    # Call Sparse Implementation
    # NOTE: New signature is (logits, logit_lens, targets, t_start, t_end, t_lens)
    output = loss_fn(
        logits=logits_sparse,
        logit_lens=logit_lens,
        targets=targets,
        t_start=t_start,
        t_end=t_end,
        t_lens=t_lens,
    )

    loss_s = output["loss"]

    # --- 3. Comparison ---

    # Check Loss Value
    # Slightly relaxed tolerance for floating point differences in summation order
    assert torch.isclose(
        loss_r, loss_s, atol=1e-5
    ), f"Loss mismatch! Ref: {loss_r.item():.6f}, Sparse: {loss_s.item():.6f}"

    # Check Gradients
    if loss_r.item() != 0.0:
        loss_r.backward()
        loss_s.backward()

        assert torch.allclose(
            logits_ref.grad, logits_sparse.grad, atol=1e-5
        ), "Gradient mismatch! Backprop logic differs."


# -------------------------------------------------------------------
# TEST CASES
# -------------------------------------------------------------------


def test_basic_alignment(loss_fn, loss_fn_ref):
    """Standard case: distinct segments, no overlaps."""
    B, S, C = 2, 10, 5
    logits = torch.randn(B, S, C)
    targets = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
    start = torch.tensor([[0, 5], [1, 6]], dtype=torch.long)
    end = torch.tensor([[4, 9], [5, 8]], dtype=torch.long)
    lens = torch.tensor([10, 10], dtype=torch.long)

    run_comparison(loss_fn, loss_fn_ref, logits, targets, start, end, lens)


def test_heavy_overlap_soft_labels(loss_fn, loss_fn_ref):
    """Test frames covered by multiple targets (mixture)."""
    B, S, C = 1, 10, 5
    logits = torch.randn(B, S, C)
    targets = torch.tensor([[2, 3]], dtype=torch.long)
    start = torch.tensor([[0, 4]], dtype=torch.long)
    end = torch.tensor([[8, 9]], dtype=torch.long)
    lens = torch.tensor([10], dtype=torch.long)

    run_comparison(loss_fn, loss_fn_ref, logits, targets, start, end, lens)


def test_duplicate_class_overlap(loss_fn, loss_fn_ref):
    """Test two targets defining the SAME class on SAME frames."""
    B, S, C = 1, 5, 4
    logits = torch.randn(B, S, C)
    targets = torch.tensor([[1, 1]], dtype=torch.long)
    start = torch.tensor([[0, 0]], dtype=torch.long)
    end = torch.tensor([[4, 4]], dtype=torch.long)
    lens = torch.tensor([5], dtype=torch.long)

    run_comparison(loss_fn, loss_fn_ref, logits, targets, start, end, lens)


def test_padding_handling(loss_fn, loss_fn_ref):
    """Test that targets defined outside logit_lens are ignored."""
    B, S, C = 1, 10, 5
    logits = torch.randn(B, S, C)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    start = torch.tensor([[0, 8]], dtype=torch.long)
    end = torch.tensor([[4, 9]], dtype=torch.long)

    # Valid audio only goes to 6. Target 2 (at 8-9) should be masked out.
    lens = torch.tensor([6], dtype=torch.long)

    run_comparison(loss_fn, loss_fn_ref, logits, targets, start, end, lens)


def test_ignore_index(loss_fn, loss_fn_ref):
    """Test that specific target IDs are ignored."""
    B, S, C = 1, 5, 4
    logits = torch.randn(B, S, C)
    targets = torch.tensor([[1, -100]], dtype=torch.long)
    start = torch.tensor([[0, 2]], dtype=torch.long)
    end = torch.tensor([[1, 4]], dtype=torch.long)
    lens = torch.tensor([5], dtype=torch.long)

    run_comparison(loss_fn, loss_fn_ref, logits, targets, start, end, lens)


def test_gaps_in_alignment(loss_fn, loss_fn_ref):
    """Test frames with NO targets covering them."""
    B, S, C = 1, 10, 4
    logits = torch.randn(B, S, C)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    start = torch.tensor([[0, 6]], dtype=torch.long)
    end = torch.tensor([[2, 9]], dtype=torch.long)
    lens = torch.tensor([10], dtype=torch.long)

    run_comparison(loss_fn, loss_fn_ref, logits, targets, start, end, lens)


def test_all_frames_masked(loss_fn, loss_fn_ref):
    """Test when no frames are valid (should return 0)."""
    B, S, C = 1, 5, 4
    logits = torch.randn(B, S, C)
    targets = torch.tensor([[1]], dtype=torch.long)
    start = torch.tensor([[0]], dtype=torch.long)
    end = torch.tensor([[4]], dtype=torch.long)
    # Audio length 0 -> No valid frames
    lens = torch.tensor([0], dtype=torch.long)

    run_comparison(loss_fn, loss_fn_ref, logits, targets, start, end, lens)


# ===================================================================
# BOUNDARY LOSS TESTS
# ===================================================================


def _reference_boundary_loss(
    boundary_logits: torch.Tensor,
    logit_lens: torch.Tensor,
    target_start_idx: torch.Tensor,
    t_lens=None,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Loop-based reference for BoundaryLoss."""
    B, S = boundary_logits.shape
    T = target_start_idx.shape[1]
    labels = torch.zeros(B, S)
    for b in range(B):
        n = int(t_lens[b]) if t_lens is not None else T
        for t in range(n):
            idx = int(target_start_idx[b, t])
            if 0 <= idx < S:
                labels[b, idx] = 1.0
    bce = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor([pos_weight])
    )
    frame_mask = torch.arange(S).unsqueeze(0) < logit_lens.unsqueeze(1)
    return (bce(boundary_logits, labels) * frame_mask).sum() / frame_mask.sum().clamp_min(1)


def run_boundary_comparison(loss_fn, logits, logit_lens, t_start, t_lens=None):
    ref = _reference_boundary_loss(logits.detach(), logit_lens, t_start, t_lens)
    logits_s = logits.clone().detach().requires_grad_(True)
    out = loss_fn(logits_s, logit_lens, t_start, t_lens)
    assert torch.isclose(
        out["loss"], ref, atol=1e-5
    ), f"Loss mismatch: ref={ref.item():.6f} got={out['loss'].item():.6f}"
    out["loss"].backward()
    assert logits_s.grad is not None


@pytest.fixture
def boundary_loss_fn():
    return BoundaryLoss(pos_weight=1.0)


def test_boundary_basic(boundary_loss_fn):
    """Single phone per utterance: boundary at frame 0."""
    B, S = 2, 10
    logits = torch.randn(B, S)
    t_start = torch.zeros(B, 1, dtype=torch.long)
    logit_lens = torch.tensor([10, 8], dtype=torch.long)
    run_boundary_comparison(boundary_loss_fn, logits, logit_lens, t_start)


def test_boundary_multi_phone(boundary_loss_fn):
    """Three phones per utterance; boundaries at frames 0, 4, 8."""
    B, S = 1, 12
    logits = torch.randn(B, S)
    t_start = torch.tensor([[0, 4, 8]], dtype=torch.long)
    logit_lens = torch.tensor([12], dtype=torch.long)
    run_boundary_comparison(boundary_loss_fn, logits, logit_lens, t_start)


def test_boundary_padding(boundary_loss_fn):
    """logit_lens < S: frames beyond valid length must not contribute."""
    B, S = 1, 12
    logits = torch.randn(B, S)
    t_start = torch.tensor([[0, 4, 9]], dtype=torch.long)
    # Only 7 valid frames; boundary at 9 is outside valid range
    logit_lens = torch.tensor([7], dtype=torch.long)
    run_boundary_comparison(boundary_loss_fn, logits, logit_lens, t_start)


def test_boundary_t_lens(boundary_loss_fn):
    """Batched: second item has fewer valid phones than padding implies."""
    B, S = 2, 10
    logits = torch.randn(B, S)
    t_start = torch.tensor([[0, 4, 8], [0, 6, 0]], dtype=torch.long)
    logit_lens = torch.tensor([10, 10], dtype=torch.long)
    # Second item only has 2 valid phones
    t_lens = torch.tensor([3, 2], dtype=torch.long)
    run_boundary_comparison(boundary_loss_fn, logits, logit_lens, t_start, t_lens)


def test_boundary_out_of_range(boundary_loss_fn):
    """target_start_idx at exactly S-1 (valid) and S (clamped/masked out)."""
    B, S = 1, 8
    logits = torch.randn(B, S)
    # Frame S-1=7 is valid; frame S=8 is out of range and should be ignored
    t_start = torch.tensor([[7, 8]], dtype=torch.long)
    logit_lens = torch.tensor([S], dtype=torch.long)
    t_lens = torch.tensor([2], dtype=torch.long)
    run_boundary_comparison(boundary_loss_fn, logits, logit_lens, t_start, t_lens)


def test_boundary_zero_valid_frames(boundary_loss_fn):
    """logit_lens=0: no frames valid, loss must be 0."""
    B, S = 1, 5
    logits = torch.randn(B, S)
    t_start = torch.tensor([[0, 2]], dtype=torch.long)
    logit_lens = torch.tensor([0], dtype=torch.long)
    out = boundary_loss_fn(logits, logit_lens, t_start)
    assert out["loss"].item() == 0.0


def test_boundary_pos_weight():
    """pos_weight > 1.0 must produce strictly larger loss on inputs with boundaries."""
    B, S = 1, 6
    logits = torch.zeros(B, S)  # zero logits -> p=0.5 everywhere
    t_start = torch.tensor([[0, 3]], dtype=torch.long)
    logit_lens = torch.tensor([S], dtype=torch.long)

    loss_default = BoundaryLoss(pos_weight=1.0)(logits, logit_lens, t_start)["loss"]
    loss_weighted = BoundaryLoss(pos_weight=5.0)(logits, logit_lens, t_start)["loss"]
    assert loss_weighted > loss_default


def test_boundary_gradient(boundary_loss_fn):
    """Backward pass must not raise and must produce finite gradients."""
    B, S = 2, 8
    logits = torch.randn(B, S, requires_grad=True)
    t_start = torch.tensor([[0, 4], [1, 5]], dtype=torch.long)
    logit_lens = torch.tensor([8, 7], dtype=torch.long)
    out = boundary_loss_fn(logits, logit_lens, t_start)
    out["loss"].backward()
    assert logits.grad is not None
    assert logits.grad.isfinite().all()
