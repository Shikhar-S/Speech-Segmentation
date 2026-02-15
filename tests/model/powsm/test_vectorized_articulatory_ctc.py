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


def _make_custom_neighbors(V=7, topk=3, blank_id=0, seed=0):
    """
    Make a distance matrix with a few close pairs, rest far,
    then convert to neighbor lists. Useful for stress tests.
    """
    g = torch.Generator().manual_seed(seed)
    dist = torch.full((V, V), 10.0)
    dist.fill_diagonal_(0.0)

    # Inject a few small distances (symmetric)
    pairs = [(1, 3, 0.2), (2, 5, 0.3), (4, 6, 0.15)]
    for i, j, d in pairs:
        if i < V and j < V:
            dist[i, j] = d
            dist[j, i] = d

    nids, ndists = matrix_to_neighbor_lists(dist, topk=topk, blank_id=blank_id)
    return {"global": (nids, ndists)}, V


def _make_models(neighbors, V, beta=10.0, topk=3, normalize=True, modified_topo=False, label_smoothing=0.0):
    original = ArticulatoryCTC(
        neighbors_by_lang=neighbors,
        beta=beta,
        topk=topk,
        normalize=normalize,
        modified_topo=modified_topo,
        label_smoothing=label_smoothing,
    )
    vectorized = VectorizedArticulatoryCTC(
        neighbors_by_lang=neighbors,
        beta=beta,
        topk=topk,
        normalize=normalize,
        modified_topo=modified_topo,
        label_smoothing=label_smoothing,
    )
    return original, vectorized


def _assert_close_per_utt(loss_orig, loss_vec, rtol=1e-4, atol=1e-5):
    assert loss_orig.shape == loss_vec.shape
    assert torch.isfinite(loss_orig).all() == torch.isfinite(loss_vec).all()
    if torch.isfinite(loss_orig).all():
        torch.testing.assert_close(loss_orig, loss_vec, rtol=rtol, atol=atol)


def test_vectorized_matches_original_repeated_tokens_min_hlens_filter():
    """
    Repeated tokens increase minimum required frames in find_minimum_hlens.
    Ensure both implementations:
      1) drop invalid samples identically
      2) return the same per-utt losses for valid ones
    """
    neighbors, V = _make_custom_neighbors(V=6, topk=2, seed=1)
    original, vectorized = _make_models(neighbors, V, beta=10.0, topk=2)

    # Construct batch where some transcripts with repeats become invalid for small hlens
    B, T = 4, 6
    nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)

    ys_pad = torch.tensor([
        [1, 1, 1, 0, 0],  # repeats -> min_hlen bigger
        [2, 3, 0, 0, 0],
        [4, 4, 5, 0, 0],  # repeats
        [1, 2, 3, 0, 0],
    ])
    ylens = torch.tensor([3, 2, 3, 3])

    # Small hlens to force invalidation for some
    hlens = torch.tensor([3, 6, 4, 6])

    loss_orig = original(nnet_output, ys_pad, hlens, ylens)
    loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)

    # If one invalidates, the other must invalidate in the same places (they both return vector in original order)
    assert loss_orig.shape == loss_vec.shape
    assert (torch.isfinite(loss_orig) == torch.isfinite(loss_vec)).all()

    # Compare only finite entries
    mask = torch.isfinite(loss_orig)
    if mask.any():
        torch.testing.assert_close(loss_orig[mask], loss_vec[mask], rtol=1e-4, atol=1e-5)


def test_vectorized_matches_original_modified_topo_true():
    """Exercise k2.ctc_topo(modified=True) path; ensure equivalence."""
    neighbors, V = _make_custom_neighbors(V=7, topk=3, seed=2)
    original, vectorized = _make_models(neighbors, V, beta=8.0, topk=3, modified_topo=True)

    B, T = 3, 9
    nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)
    ys_pad = torch.tensor([
        [1, 3, 2, 0, 0],
        [2, 2, 5, 0, 0],
        [4, 6, 1, 0, 0],
    ])
    ylens = torch.tensor([3, 3, 3])
    hlens = torch.tensor([9, 8, 9])

    loss_orig = original(nnet_output, ys_pad, hlens, ylens)
    loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)
    _assert_close_per_utt(loss_orig, loss_vec, rtol=1e-3, atol=1e-4)


def test_vectorized_matches_original_extreme_beta_limits():
    """
    beta very small -> nearly uniform over candidates (after softmax)
    beta very large -> almost always self candidate
    Both should still match.
    """
    neighbors, V = _make_custom_neighbors(V=6, topk=2, seed=3)

    B, T = 2, 10
    ys_pad = torch.tensor([[1, 3, 0], [2, 5, 0]])
    ylens = torch.tensor([2, 2])
    hlens = torch.tensor([10, 9])

    for beta in [0.01, 1.0, 200.0]:
        original, vectorized = _make_models(neighbors, V, beta=beta, topk=2, normalize=True)
        nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)
        loss_orig = original(nnet_output, ys_pad, hlens, ylens)
        loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)
        _assert_close_per_utt(loss_orig, loss_vec, rtol=1e-3, atol=1e-4)


def test_vectorized_matches_original_longer_transcript_stress():
    """
    Stress a longer transcript to shake out any arc sorting / construction bugs.
    Keep V small but L bigger.
    """
    neighbors, V = _make_custom_neighbors(V=8, topk=3, seed=4)
    original, vectorized = _make_models(neighbors, V, beta=10.0, topk=3)

    B, T = 1, 40
    nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)

    # L=20 transcript (no blanks) with some repeats
    y = [1, 2, 2, 3, 4, 1, 5, 6, 6, 2, 3, 7, 4, 4, 5, 1, 2, 3, 6, 7]
    ys_pad = torch.tensor([y])
    ylens = torch.tensor([len(y)])
    hlens = torch.tensor([T])

    loss_orig = original(nnet_output, ys_pad, hlens, ylens)
    loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)
    _assert_close_per_utt(loss_orig, loss_vec, rtol=1e-3, atol=1e-4)


def test_vectorized_matches_original_device_cuda_if_available():
    """
    Ensure no hidden CPU-only ops remain in vectorized build (e.g., accidental .cpu() usage for sorting).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    neighbors, V = _make_custom_neighbors(V=6, topk=2, seed=5)
    original, vectorized = _make_models(neighbors, V, beta=10.0, topk=2)

    device = torch.device("cuda")
    original = original.to(device)
    vectorized = vectorized.to(device)

    B, T = 3, 12
    nnet_output = torch.randn(B, T, V, device=device).log_softmax(dim=-1)
    ys_pad = torch.tensor([[1, 2, 3, 0], [2, 2, 5, 0], [1, 0, 0, 0]], device=device)
    ylens = torch.tensor([3, 3, 1], device=device)
    hlens = torch.tensor([12, 10, 9], device=device)

    loss_orig = original(nnet_output, ys_pad, hlens, ylens)
    loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)
    _assert_close_per_utt(loss_orig, loss_vec, rtol=1e-3, atol=1e-4)


def test_vectorized_matches_original_under_autocast_fp16_if_cuda():
    """
    If you ever run training under AMP, DenseFsaVec scores can be fp16/bf16.
    This test ensures vectorized graph building isn't sensitive to autocast.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    neighbors, V = _make_custom_neighbors(V=6, topk=2, seed=6)
    original, vectorized = _make_models(neighbors, V, beta=10.0, topk=2)

    device = torch.device("cuda")
    original = original.to(device)
    vectorized = vectorized.to(device)

    B, T = 2, 10
    ys_pad = torch.tensor([[1, 2, 3, 0], [2, 5, 0, 0]], device=device)
    ylens = torch.tensor([3, 2], device=device)
    hlens = torch.tensor([10, 10], device=device)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        nnet_output = torch.randn(B, T, V, device=device).log_softmax(dim=-1)
        loss_orig = original(nnet_output, ys_pad, hlens, ylens)
        loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens)

    # allow slightly looser tol due to fp16 numerics
    _assert_close_per_utt(loss_orig, loss_vec, rtol=3e-3, atol=3e-4)


def test_vectorized_matches_original_lang_routing_global_fallback():
    """
    Vectorized takes lang_per_utt; ensure unknown langs fall back to 'global'
    in the same way as original.
    """
    neighbors, V = _make_custom_neighbors(V=6, topk=2, seed=7)

    original = ArticulatoryCTC(neighbors_by_lang=neighbors, beta=10.0, topk=2, normalize=True)
    vectorized = VectorizedArticulatoryCTC(neighbors_by_lang=neighbors, beta=10.0, topk=2, normalize=True)

    B, T = 3, 9
    nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)
    ys_pad = torch.tensor([[1, 2, 0], [2, 5, 0], [1, 3, 4]])
    ylens = torch.tensor([2, 2, 3])
    hlens = torch.tensor([9, 8, 9])

    # Include unknown lang token; both should fall back to global
    lang_per_utt = ["xx_UNKNOWN", "global", "yy_UNKNOWN"]

    loss_orig = original(nnet_output, ys_pad, hlens, ylens, lang_per_utt=lang_per_utt)
    loss_vec = vectorized(nnet_output, ys_pad, hlens, ylens, lang_per_utt=lang_per_utt)

    _assert_close_per_utt(loss_orig, loss_vec, rtol=1e-4, atol=1e-5)


def test_vectorized_matches_original_determinism_same_seed_same_inputs():
    """
    No randomness in graph construction: repeated calls should match exactly
    (given same inputs). This is useful to catch hidden nondeterminism in sorting.
    """
    neighbors, V = _make_custom_neighbors(V=6, topk=2, seed=8)
    original, vectorized = _make_models(neighbors, V, beta=10.0, topk=2)

    torch.manual_seed(999)
    B, T = 2, 10
    nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)
    ys_pad = torch.tensor([[1, 2, 3, 0], [2, 2, 5, 0]])
    ylens = torch.tensor([3, 3])
    hlens = torch.tensor([10, 10])

    loss_vec_1 = vectorized(nnet_output, ys_pad, hlens, ylens)
    loss_vec_2 = vectorized(nnet_output, ys_pad, hlens, ylens)
    torch.testing.assert_close(loss_vec_1, loss_vec_2, rtol=0.0, atol=0.0)

    loss_orig_1 = original(nnet_output, ys_pad, hlens, ylens)
    loss_orig_2 = original(nnet_output, ys_pad, hlens, ylens)
    torch.testing.assert_close(loss_orig_1, loss_orig_2, rtol=0.0, atol=0.0)

    # And cross-compare
    _assert_close_per_utt(loss_orig_1, loss_vec_1, rtol=1e-4, atol=1e-5)
    

import time
import pytest
import torch


def _assert_cuda_io(model, nnet_output, ys_pad, hlens, ylens):
    assert nnet_output.is_cuda, "nnet_output must be on CUDA"
    assert ys_pad.is_cuda, "ys_pad must be on CUDA"
    assert hlens.is_cuda, "hlens must be on CUDA"
    assert ylens.is_cuda, "ylens must be on CUDA"


def _assert_topo_on_cuda(model):
    # After at least one forward call, _topo should be cached on the same device used
    assert getattr(model, "_topo", None) is not None, "Expected model._topo to be set after warmup"
    assert str(model._topo.device).startswith("cuda"), f"Expected topo on CUDA, got {model._topo.device}"


@torch.no_grad()
def _time_forward_cuda(model, nnet_output, ys_pad, hlens, ylens, *, iters=50, warmup=10, lang_per_utt=None):
    # Warmup
    for _ in range(warmup):
        out = model(nnet_output, ys_pad, hlens, ylens, lang_per_utt=lang_per_utt)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    for _ in range(iters):
        out = model(nnet_output, ys_pad, hlens, ylens, lang_per_utt=lang_per_utt)
    ender.record()
    torch.cuda.synchronize()

    ms = starter.elapsed_time(ender)  # milliseconds for the whole loop
    return ms / iters, out  # ms/iter, last output

@pytest.mark.cuda
def test_speed_original_vs_vectorized_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    original, vectorized, V = _make_neighbors_and_models(V=64, beta=10.0, topk=10, seed=123)
    device = torch.device("cuda")
    original = original.to(device)
    vectorized = vectorized.to(device)

    B, T = 16, 200
    torch.manual_seed(0)

    nnet_output = torch.randn(B, T, V, device=device).log_softmax(dim=-1)
    Lmax = 50
    ys_pad = torch.zeros(B, Lmax, dtype=torch.long, device=device)
    ylens = torch.randint(low=10, high=Lmax, size=(B,), device=device)
    for b in range(B):
        L = int(ylens[b].item())
        ys_pad[b, :L] = torch.randint(low=1, high=V, size=(L,), device=device)
    hlens = torch.full((B,), T, dtype=torch.long, device=device)

    _assert_cuda_io(original, nnet_output, ys_pad, hlens, ylens)
    _assert_cuda_io(vectorized, nnet_output, ys_pad, hlens, ylens)

    # Warmup once so _topo is created/cached on CUDA (and to avoid first-call overhead)
    _ = original(nnet_output, ys_pad, hlens, ylens)
    _ = vectorized(nnet_output, ys_pad, hlens, ylens)
    torch.cuda.synchronize()

    _assert_topo_on_cuda(original)
    _assert_topo_on_cuda(vectorized)

    ms_orig, out_orig = _time_forward_cuda(original, nnet_output, ys_pad, hlens, ylens, iters=50, warmup=10)
    ms_vec, out_vec = _time_forward_cuda(vectorized, nnet_output, ys_pad, hlens, ylens, iters=50, warmup=10)

    assert out_orig.is_cuda and out_vec.is_cuda, "Loss tensors should be on CUDA"

    # Correctness (loose tol)
    mask = torch.isfinite(out_orig) & torch.isfinite(out_vec)
    if mask.any():
        torch.testing.assert_close(out_orig[mask], out_vec[mask], rtol=1e-3, atol=1e-4)

    print(f"original:   {ms_orig:.3f} ms/iter")
    print(f"vectorized: {ms_vec:.3f} ms/iter")
    print(f"speedup:    {ms_orig / ms_vec:.2f}x")