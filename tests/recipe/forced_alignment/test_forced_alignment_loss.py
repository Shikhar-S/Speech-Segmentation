"""Tests for ForcedAlignLoss."""

import torch
import pytest
from src.recipe.forced_alignment.forced_alignment_loss import ForcedAlignmentLoss


# Helper function to manually compute cross-entropy
def manual_ce_loss(logits, targets, ignore_index=-1):
    """
    Computes manual cross-entropy loss, averages over valid frames.
    logits: (Batch, Length, Class)
    targets: (Batch, Length)
    """
    bs, ilen, vocab_sz = logits.size()
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    # (B, I)
    gathered_log_probs = torch.gather(
        log_probs, 2, targets.unsqueeze(2).clamp(min=0)  # clamp for ignore_index
    ).squeeze(2)

    valid_mask = targets != ignore_index

    # Mask out invalid frames
    loss = -gathered_log_probs.where(valid_mask, 0.0)

    total_loss = loss.sum()
    num_valid = valid_mask.sum().float()

    if num_valid == 0:
        return torch.tensor(0.0)

    return total_loss / num_valid


@pytest.fixture
def loss_fn():
    return ForcedAlignmentLoss(ignore_index=-1)


def test_basic_case(loss_fn):
    """Tests a single batch item with full alignment."""
    torch.manual_seed(0)
    # (B, I, V)
    logits = torch.randn(1, 5, 4)
    # (B, T)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    # (B, T)
    target_start = torch.tensor([[0, 2]], dtype=torch.long)
    target_end = torch.tensor([[1, 4]], dtype=torch.long)
    # (B,)
    input_lengths = torch.tensor([5], dtype=torch.long)

    loss = loss_fn(logits, targets, target_start, target_end, input_lengths)

    # Expected frame-level targets: [1, 1, 2, 2, 2]
    expected_targets = torch.tensor([[1, 1, 2, 2, 2]], dtype=torch.long)
    expected_loss = manual_ce_loss(logits, expected_targets)

    assert torch.allclose(loss, expected_loss)


def test_batch_case(loss_fn):
    """Tests a batch of 2 items."""
    torch.manual_seed(1)
    # (B, I, V)
    logits = torch.randn(2, 5, 4)
    # (B, T)
    targets = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
    # (B, T)
    target_start = torch.tensor([[0, 3], [0, 2]], dtype=torch.long)
    target_end = torch.tensor([[2, 4], [1, 4]], dtype=torch.long)
    # (B,)
    input_lengths = torch.tensor([5, 5], dtype=torch.long)

    loss = loss_fn(logits, targets, target_start, target_end, input_lengths)

    # Item 1 expected: [1, 1, 1, 2, 2]
    # Item 2 expected: [3, 3, 0, 0, 0]
    expected_targets = torch.tensor(
        [[1, 1, 1, 2, 2], [3, 3, 0, 0, 0]], dtype=torch.long
    )
    expected_loss = manual_ce_loss(logits, expected_targets)

    assert torch.allclose(loss, expected_loss)


def test_input_length_padding(loss_fn):
    """Tests that frames beyond input_lengths are ignored."""
    torch.manual_seed(2)
    # (B, I, V)
    logits = torch.randn(1, 5, 4)
    # (B, T)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    # (B, T)
    target_start = torch.tensor([[0, 2]], dtype=torch.long)
    target_end = torch.tensor([[1, 4]], dtype=torch.long)

    # Only first 3 frames are valid
    input_lengths = torch.tensor([3], dtype=torch.long)

    loss = loss_fn(logits, targets, target_start, target_end, input_lengths)

    # Frame-level targets: [1, 1, 2, 2, 2]
    # Valid mask (from length): [T, T, T, F, F]
    # Final targets for loss: [1, 1, 2, -1, -1]
    expected_targets = torch.tensor([[1, 1, 2, -1, -1]], dtype=torch.long)
    expected_loss = manual_ce_loss(logits, expected_targets)

    assert torch.allclose(loss, expected_loss)


def test_target_ignore_index(loss_fn):
    """Tests that targets with ignore_index are ignored."""
    torch.manual_seed(3)
    # (B, I, V)
    logits = torch.randn(1, 5, 4)
    # (B, T)
    targets = torch.tensor([[1, -1]], dtype=torch.long)  # -1 is ignore_index
    # (B, T)
    target_start = torch.tensor([[0, 2]], dtype=torch.long)
    target_end = torch.tensor([[1, 4]], dtype=torch.long)
    # (B,)
    input_lengths = torch.tensor([5], dtype=torch.long)

    loss = loss_fn(logits, targets, target_start, target_end, input_lengths)

    # Expected frame-level targets: [1, 1, -1, -1, -1]
    expected_targets = torch.tensor([[1, 1, -1, -1, -1]], dtype=torch.long)
    expected_loss = manual_ce_loss(logits, expected_targets)

    assert torch.allclose(loss, expected_loss)


def test_gap_case(loss_fn):
    """Tests that frames with no alignment (gaps) are ignored."""
    torch.manual_seed(4)
    # (B, I, V)
    logits = torch.randn(1, 5, 4)
    # (B, T)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    # (B, T)
    target_start = torch.tensor([[0, 3]], dtype=torch.long)
    target_end = torch.tensor([[1, 4]], dtype=torch.long)  # Frame 2 is a gap
    # (B,)
    input_lengths = torch.tensor([5], dtype=torch.long)

    loss = loss_fn(logits, targets, target_start, target_end, input_lengths)

    # Expected frame-level targets: [1, 1, -1, 2, 2]
    expected_targets = torch.tensor([[1, 1, -1, 2, 2]], dtype=torch.long)
    expected_loss = manual_ce_loss(logits, expected_targets)

    assert torch.allclose(loss, expected_loss)


def test_combined_case(loss_fn):
    """Tests batching with gaps, padding, and ignore_index."""
    torch.manual_seed(5)
    # (B, I, V)
    logits = torch.randn(2, 5, 4)
    # (B, T)
    targets = torch.tensor([[1, -1], [3, 0]], dtype=torch.long)
    # (B, T)
    target_start = torch.tensor([[0, 3], [0, 4]], dtype=torch.long)
    target_end = torch.tensor([[1, 4], [2, 4]], dtype=torch.long)
    # (B,)
    input_lengths = torch.tensor([5, 4], dtype=torch.long)  # 2nd item has padding

    loss = loss_fn(logits, targets, target_start, target_end, input_lengths)

    # Item 1:
    #   Alignments: [1, 1, -1, -1, -1] (frame 2=gap, 3-4=ignore_index)
    #   Length Mask: [T, T, T, T, T]
    #   Final: [1, 1, -1, -1, -1]
    # Item 2:
    #   Alignments: [3, 3, 3, -1, 0] (frame 3=gap)
    #   Length Mask: [T, T, T, T, F]
    #   Final: [3, 3, 3, -1, -1]
    expected_targets = torch.tensor(
        [[1, 1, -1, -1, -1], [3, 3, 3, -1, -1]], dtype=torch.long
    )
    expected_loss = manual_ce_loss(logits, expected_targets)

    assert torch.allclose(loss, expected_loss)


def test_no_valid_frames(loss_fn):
    """Tests case with no valid frames (e.g., all ignored or length 0)."""
    torch.manual_seed(6)
    logits = torch.randn(2, 5, 4)

    # Case 1: All targets ignored
    targets = torch.tensor([[-1, -1], [-1, -1]], dtype=torch.long)
    target_start = torch.tensor([[0, 2], [0, 2]], dtype=torch.long)
    target_end = torch.tensor([[1, 4], [1, 4]], dtype=torch.long)
    input_lengths = torch.tensor([5, 5], dtype=torch.long)

    loss1 = loss_fn(logits, targets, target_start, target_end, input_lengths)
    assert torch.allclose(loss1, torch.tensor(0.0))

    # Case 2: All input lengths are 0
    targets = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
    target_start = torch.tensor([[0, 2], [0, 2]], dtype=torch.long)
    target_end = torch.tensor([[1, 4], [1, 4]], dtype=torch.long)
    input_lengths = torch.tensor([0, 0], dtype=torch.long)  # Lengths are 0

    loss2 = loss_fn(logits, targets, target_start, target_end, input_lengths)
    assert torch.allclose(loss2, torch.tensor(0.0))
