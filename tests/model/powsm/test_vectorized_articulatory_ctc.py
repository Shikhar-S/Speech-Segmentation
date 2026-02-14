"""Tests that VectorizedArticulatoryCTC matches ArticulatoryCTC loss values."""

import pytest
import torch

from src.model.powsm.articulatory_ctc import ArticulatoryCTC
from src.model.powsm.vectorized_articulatory_ctc import VectorizedArticulatoryCTC
from src.model.xeusphoneme.builders import matrix_to_neighbor_lists


def _make_neighbors_and_models(V=5, beta=10.0, topk=2, seed=42):
    torch.manual_seed(seed)
    dist = torch.zeros(V, V)
    for i in range(V):
        dist[i, i] = 0.0
    dist[2, 4] = dist[4, 2] = 0.1
    for i in range(V):
        for j in range(V):
            if i != j and dist[i, j] == 0:
                dist[i, j] = 10.0
    nids, ndists = matrix_to_neighbor_lists(dist, topk=topk, blank_id=0)
    neighbors = {"global": (nids, ndists)}
    original = ArticulatoryCTC(
        neighbors_by_lang=neighbors,
        beta=beta,
        topk=topk,
        normalize=True,
    )
    vectorized = VectorizedArticulatoryCTC(
        neighbors_by_lang=neighbors,
        beta=beta,
        topk=topk,
        normalize=True,
    )
    return original, vectorized, V


def test_vectorized_matches_original_single_utt():
    """Single utterance: losses should match within tolerance."""
    original, vectorized, V = _make_neighbors_and_models()
    B, T = 1, 10
    nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)
    ys_pad = torch.tensor([[1, 2, 3]])
    hlens = torch.tensor([T])
    ylens = torch.tensor([3])

    loss_orig = original(nnet_output, ys_pad, hlens, ylens)
    loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)

    assert loss_orig.shape == loss_vec.shape
    assert torch.isfinite(loss_orig) == torch.isfinite(loss_vec)
    if torch.isfinite(loss_orig).all():
        torch.testing.assert_close(loss_orig, loss_vec, rtol=1e-4, atol=1e-5)


def test_vectorized_matches_original_batch():
    """Batch of utterances: losses should match per utterance."""
    original, vectorized, V = _make_neighbors_and_models()
    B, T = 4, 12
    nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)
    ys_pad = torch.tensor([
        [1, 2, 3, 0, 0],
        [2, 2, 4, 3, 0],
        [1, 2, 0, 0, 0],
        [1, 2, 3, 4, 0],
    ])
    hlens = torch.tensor([12, 10, 8, 11])
    ylens = torch.tensor([3, 4, 2, 4])

    loss_orig = original(nnet_output, ys_pad, hlens, ylens)
    loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)

    assert loss_orig.shape == loss_vec.shape
    for i in range(loss_orig.numel()):
        o, v = loss_orig[i].item(), loss_vec[i].item()
        assert (torch.isfinite(loss_orig[i]) == torch.isfinite(loss_vec[i])), (
            f"Utterance {i}: orig finite={torch.isfinite(loss_orig[i])} vec finite={torch.isfinite(loss_vec[i])}"
        )
        if torch.isfinite(loss_orig[i]):
            assert abs(o - v) < 1e-4 * (abs(o) + 1) + 1e-5, (
                f"Utterance {i}: orig={o} vec={v}"
            )


def test_vectorized_matches_original_empty_transcript():
    """Transcript with only blanks: both should handle (e.g. trivial acceptor)."""
    original, vectorized, V = _make_neighbors_and_models()
    B, T = 1, 5
    nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)
    ys_pad = torch.tensor([[0, 0, 0]])
    hlens = torch.tensor([T])
    ylens = torch.tensor([3])

    loss_orig = original(nnet_output, ys_pad, hlens, ylens)
    loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)

    assert loss_orig.shape == loss_vec.shape
    assert torch.isfinite(loss_orig) == torch.isfinite(loss_vec)
    if torch.isfinite(loss_orig).all():
        torch.testing.assert_close(loss_orig, loss_vec, rtol=1e-3, atol=1e-4)


def test_vectorized_matches_original_with_label_smoothing():
    """With label_smoothing > 0, losses should still be close."""
    torch.manual_seed(123)
    V = 5
    dist = torch.zeros(V, V)
    dist[2, 4] = dist[4, 2] = 0.1
    for i in range(V):
        for j in range(V):
            if i != j and dist[i, j] == 0:
                dist[i, j] = 10.0
    nids, ndists = matrix_to_neighbor_lists(dist, topk=2, blank_id=0)
    neighbors = {"global": (nids, ndists)}
    original = ArticulatoryCTC(
        neighbors_by_lang=neighbors,
        beta=10.0,
        topk=2,
        normalize=True,
        label_smoothing=0.1,
    )
    vectorized = VectorizedArticulatoryCTC(
        neighbors_by_lang=neighbors,
        beta=10.0,
        topk=2,
        normalize=True,
        label_smoothing=0.1,
    )
    B, T = 2, 8
    nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)
    ys_pad = torch.tensor([[1, 2, 3, 0], [1, 4, 3, 0]])
    hlens = torch.tensor([8, 8])
    ylens = torch.tensor([3, 3])

    loss_orig = original(nnet_output, ys_pad, hlens, ylens)
    loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)

    assert loss_orig.shape == loss_vec.shape
    assert torch.isfinite(loss_orig).all() == torch.isfinite(loss_vec).all()
    if torch.isfinite(loss_orig).all():
        torch.testing.assert_close(loss_orig, loss_vec, rtol=1e-3, atol=1e-4)


def test_vectorized_matches_original_normalize_false():
    """With normalize=False, scores are raw logits; losses should be close."""
    original, vectorized, V = _make_neighbors_and_models()
    # Rebuild with normalize=False
    dist = torch.zeros(V, V)
    dist[2, 4] = dist[4, 2] = 0.1
    for i in range(V):
        for j in range(V):
            if i != j and dist[i, j] == 0:
                dist[i, j] = 10.0
    nids, ndists = matrix_to_neighbor_lists(dist, topk=2, blank_id=0)
    neighbors = {"global": (nids, ndists)}
    original = ArticulatoryCTC(
        neighbors_by_lang=neighbors,
        beta=10.0,
        topk=2,
        normalize=False,
    )
    vectorized = VectorizedArticulatoryCTC(
        neighbors_by_lang=neighbors,
        beta=10.0,
        topk=2,
        normalize=False,
    )
    B, T = 1, 6
    nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)
    ys_pad = torch.tensor([[1, 2, 3]])
    hlens = torch.tensor([T])
    ylens = torch.tensor([3])

    loss_orig = original(nnet_output, ys_pad, hlens, ylens)
    loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)

    assert loss_orig.shape == loss_vec.shape
    if torch.isfinite(loss_orig).all():
        torch.testing.assert_close(loss_orig, loss_vec, rtol=1e-4, atol=1e-5)
