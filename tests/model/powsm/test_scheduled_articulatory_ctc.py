"""Tests for ScheduledArticulatoryCTC (penalty-scheduled articulatory CTC)."""

import math
import pytest
import torch

from src.model.powsm.scheduled_articulatory_ctc import ScheduledArticulatoryCTC
from src.model.powsm.vectorized_articulatory_ctc import VectorizedArticulatoryCTC
from src.model.xeusphoneme.builders import matrix_to_neighbor_lists


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dist_matrix(V, close_pairs=None, far_val=10.0):
    """Build a symmetric distance matrix with specified close pairs."""
    dist = torch.full((V, V), far_val)
    dist.fill_diagonal_(0.0)
    for i, j, d in close_pairs or []:
        if i < V and j < V:
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _make_neighbors(V=6, topk=3, close_pairs=None, blank_id=0):
    dist = _make_dist_matrix(V, close_pairs or [(1, 3, 0.2), (2, 5, 0.3), (4, 5, 0.15)])
    nids, ndists = matrix_to_neighbor_lists(dist, topk=topk, blank_id=blank_id)
    return {"global": (nids, ndists)}, V


def _make_model(neighbors, V, **kwargs):
    defaults = dict(
        beta=10.0, penalty_init=-8.0, penalty_final=-0.5, penalty_halflife=8000
    )
    defaults.update(kwargs)
    return ScheduledArticulatoryCTC(neighbors_by_lang=neighbors, **defaults)


def _assert_close(a, b, rtol=1e-4, atol=1e-5):
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    both_finite = torch.isfinite(a) & torch.isfinite(b)
    assert (torch.isfinite(a) == torch.isfinite(b)).all(), "Finiteness mismatch"
    if both_finite.any():
        torch.testing.assert_close(a[both_finite], b[both_finite], rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# 1. Penalty schedule math
# ---------------------------------------------------------------------------


class TestPenaltySchedule:
    def test_initial_value(self):
        neighbors, V = _make_neighbors()
        m = _make_model(
            neighbors, V, penalty_init=-10.0, penalty_final=-1.0, penalty_halflife=5000
        )
        m._global_step = 0
        assert abs(m.current_penalty() - (-10.0)) < 1e-6

    def test_halflife_midpoint(self):
        neighbors, V = _make_neighbors()
        m = _make_model(
            neighbors, V, penalty_init=-10.0, penalty_final=-2.0, penalty_halflife=5000
        )
        m._global_step = 5000
        expected = (-10.0 + -2.0) / 2  # midpoint
        assert abs(m.current_penalty() - expected) < 1e-4

    def test_converges_to_final(self):
        neighbors, V = _make_neighbors()
        m = _make_model(
            neighbors, V, penalty_init=-10.0, penalty_final=-0.5, penalty_halflife=1000
        )
        m._global_step = 100_000  # way past halflife
        assert abs(m.current_penalty() - (-0.5)) < 1e-4

    def test_monotonically_increasing(self):
        """Penalty should increase (become less negative) over time."""
        neighbors, V = _make_neighbors()
        m = _make_model(
            neighbors, V, penalty_init=-8.0, penalty_final=-0.5, penalty_halflife=4000
        )
        prev = -float("inf")
        for step in range(0, 40000, 500):
            m._global_step = step
            cur = m.current_penalty()
            assert cur >= prev - 1e-10, f"Non-monotonic at step {step}: {prev} -> {cur}"
            prev = cur

    def test_always_in_range(self):
        neighbors, V = _make_neighbors()
        m = _make_model(
            neighbors, V, penalty_init=-12.0, penalty_final=-1.0, penalty_halflife=3000
        )
        for step in [0, 100, 3000, 10000, 1_000_000]:
            m._global_step = step
            p = m.current_penalty()
            assert -12.0 - 1e-6 <= p <= -1.0 + 1e-6, f"Out of range at step {step}: {p}"


# ---------------------------------------------------------------------------
# 2. Step counter
# ---------------------------------------------------------------------------


class TestStepCounter:
    def test_step_advances_in_training_mode(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.train()
        B, T = 1, 10
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 2, 3]])
        hlens, ylens = torch.tensor([T]), torch.tensor([3])

        assert m._global_step == 0
        m(nnet, ys, hlens, ylens)
        assert m._global_step == 1
        m(nnet, ys, hlens, ylens)
        assert m._global_step == 2

    def test_step_frozen_in_eval_mode(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        B, T = 1, 10
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 2, 3]])
        hlens, ylens = torch.tensor([T]), torch.tensor([3])

        assert m._global_step == 0
        m(nnet, ys, hlens, ylens)
        assert m._global_step == 0


# ---------------------------------------------------------------------------
# 3. Basic forward pass
# ---------------------------------------------------------------------------


class TestForwardBasic:
    def test_single_utterance_finite(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        B, T = 1, 10
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        loss = m(nnet, torch.tensor([[1, 2, 3]]), torch.tensor([T]), torch.tensor([3]))
        assert loss.shape == (1,)
        assert torch.isfinite(loss).all()
        assert (loss >= 0).all()

    def test_batch_returns_per_utt_loss(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        B, T = 4, 12
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        ys = torch.tensor(
            [[1, 2, 3, 0, 0], [2, 3, 5, 4, 0], [1, 2, 0, 0, 0], [1, 2, 3, 4, 5]]
        )
        ylens = torch.tensor([3, 4, 2, 5])
        hlens = torch.tensor([12, 10, 8, 12])
        loss = m(nnet, ys, hlens, ylens)
        assert loss.shape == (B,)
        assert torch.isfinite(loss).all()

    def test_empty_transcript_handled(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        nnet = torch.randn(1, 5, V).log_softmax(dim=-1)
        loss = m(nnet, torch.tensor([[0, 0, 0]]), torch.tensor([5]), torch.tensor([3]))
        assert loss.shape == (1,)
        assert torch.isfinite(loss).all()

    def test_all_invalid_returns_zero(self):
        """When all samples fail min_hlens check, return 0."""
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        # Repeated tokens → min_hlen > hlens
        nnet = torch.randn(1, 3, V).log_softmax(dim=-1)
        loss = m(
            nnet, torch.tensor([[1, 1, 1, 1, 1]]), torch.tensor([3]), torch.tensor([5])
        )
        assert loss.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. CTC equivalence at extreme penalty
# ---------------------------------------------------------------------------


class TestCTCEquivalence:
    """At very negative penalty, neighbor arcs should carry ~0 probability,
    making the loss converge to standard CTC (no articulatory substitutions)."""

    def _standard_ctc_loss(self, nnet, ys, hlens, ylens, V, modified_topo=False):
        """Compute plain CTC loss via k2 for reference."""
        import k2

        B, T, _ = nnet.shape
        device = nnet.device
        topo = k2.ctc_topo(max_token=V - 1, modified=modified_topo, device=device)
        supervision = torch.stack(
            [torch.arange(B), torch.zeros(B), hlens.cpu()], dim=1
        ).int()
        dense = k2.DenseFsaVec(nnet, supervision)

        # Sort by descending hlens
        indices = torch.argsort(hlens, descending=True)

        graphs = []
        for idx in indices:
            b = idx.item()
            ylen = ylens[b].item()
            y = [int(ys[b, j]) for j in range(ylen) if int(ys[b, j]) != 0]
            if len(y) == 0:
                g = k2.Fsa.from_str("0 1 -1 0.0\n1").to(device)
            else:
                arcs_str = ""
                for i, tok in enumerate(y):
                    arcs_str += f"{i} {i + 1} {tok} 0.0\n"
                arcs_str += f"{len(y)} {len(y) + 1} -1 0.0\n{len(y) + 1}"
                g = k2.Fsa.from_str(arcs_str).to(device)
            g = k2.arc_sort(k2.add_epsilon_self_loops(g))
            g = k2.compose(topo, g, treat_epsilons_specially=False)
            g = k2.arc_sort(g)
            graphs.append(g)

        decoding = k2.create_fsa_vec(graphs)
        loss = k2.ctc_loss(
            decoding_graph=decoding,
            dense_fsa_vec=dense,
            output_beam=200.0,
            reduction="none",
            use_double_scores=True,
        )
        # Unsort
        unsort = torch.argsort(indices)
        return loss[unsort]

    def test_extreme_penalty_matches_ctc(self):
        neighbors, V = _make_neighbors(V=6, topk=2)
        m = _make_model(
            neighbors,
            V,
            beta=10.0,
            penalty_init=-50.0,
            penalty_final=-50.0,
            penalty_halflife=1000,
        )
        m.eval()

        B, T = 2, 12
        torch.manual_seed(42)
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 2, 3, 0], [2, 5, 4, 0]])
        ylens = torch.tensor([3, 3])
        hlens = torch.tensor([12, 10])

        loss_sched = m(nnet, ys, hlens, ylens)
        loss_ctc = self._standard_ctc_loss(nnet, ys, hlens, ylens, V)

        # With penalty=-50 the self-arc gets log_softmax(0 vs -50-beta*d) ≈ 0.0
        # so scores ≈ 0 at each position → equivalent to unweighted CTC
        _assert_close(loss_sched, loss_ctc, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 5. Equivalence with old VectorizedArticulatoryCTC at fixed penalty
# ---------------------------------------------------------------------------


class TestMatchesOldImplementation:
    """At a fixed step (penalty frozen), scores should match the old implementation
    when the old one uses the same beta, normalize=True, no label_smoothing,
    and the effective penalty is accounted for."""

    def _make_both(self, V=6, topk=2, beta=10.0, penalty=-0.5):
        close_pairs = [(2, 4, 0.1)]
        dist = _make_dist_matrix(V, close_pairs)
        nids, ndists = matrix_to_neighbor_lists(dist, topk=topk, blank_id=0)
        neighbors = {"global": (nids, ndists)}

        old = VectorizedArticulatoryCTC(
            neighbors_by_lang=neighbors,
            beta=beta,
            topk=topk,
            normalize=True,
        )
        new = ScheduledArticulatoryCTC(
            neighbors_by_lang=neighbors,
            beta=beta,
            penalty_init=penalty,
            penalty_final=penalty,
            penalty_halflife=1000,
        )
        new.eval()  # freeze step counter
        return old, new, V

    def test_zero_penalty_no_label_smoothing(self):
        """At penalty=0, the new model should match the old (normalize=True, ls=0)."""
        old, new, V = self._make_both(penalty=0.0)
        B, T = 2, 10
        torch.manual_seed(77)
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 2, 3, 0], [2, 4, 0, 0]])
        ylens = torch.tensor([3, 2])
        hlens = torch.tensor([10, 8])

        loss_old = old(nnet, ys, hlens, ylens)
        loss_new = new(nnet, ys, hlens, ylens)
        _assert_close(loss_old, loss_new, rtol=1e-4, atol=1e-5)

    def test_batch_various_lengths(self):
        old, new, V = self._make_both(penalty=0.0)
        B, T = 4, 14
        torch.manual_seed(88)
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        ys = torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [2, 2, 4, 3, 0],
                [1, 2, 0, 0, 0],
                [1, 2, 3, 4, 5],
            ]
        )
        ylens = torch.tensor([3, 4, 2, 5])
        hlens = torch.tensor([14, 12, 8, 14])

        loss_old = old(nnet, ys, hlens, ylens)
        loss_new = new(nnet, ys, hlens, ylens)
        _assert_close(loss_old, loss_new, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Penalty actually affects loss
# ---------------------------------------------------------------------------


class TestPenaltyAffectsLoss:
    def test_loss_changes_with_penalty(self):
        """Different penalty values should produce different losses.

        Note: the direction is NOT guaranteed. At penalty=0, neighbors get
        more probability mass in the graph (via log_softmax), which *reduces*
        the self-arc score. Whether total loss goes up or down depends on
        whether the nnet assigns probability to those neighbor tokens.
        We only assert that penalty has a measurable effect.
        """
        neighbors, V = _make_neighbors(
            V=6, topk=2, close_pairs=[(1, 3, 0.05), (2, 4, 0.08)]
        )
        torch.manual_seed(99)
        B, T = 1, 12
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 2, 3]])
        ylens = torch.tensor([3])
        hlens = torch.tensor([12])

        losses = {}
        for pen in [-20.0, -5.0, -1.0, 0.0]:
            m = _make_model(
                neighbors, V, penalty_init=pen, penalty_final=pen, penalty_halflife=1000
            )
            m.eval()
            loss = m(nnet, ys, hlens, ylens)
            losses[pen] = loss.item()

        # Penalty should have *some* effect: losses shouldn't all be identical
        vals = list(losses.values())
        assert not all(
            abs(v - vals[0]) < 1e-6 for v in vals
        ), f"Penalty had no effect on loss: {losses}"

        # Extreme penalties (-20 vs 0) should produce meaningfully different losses
        assert (
            abs(losses[-20.0] - losses[0.0]) > 1e-2
        ), f"Expected meaningful difference between pen=-20 and pen=0, got {losses}"

    def test_loss_changes_over_training_steps(self):
        """As step increases, penalty relaxes, arc score distribution changes,
        so loss should change. Direction depends on nnet output, so we only
        assert that the loss is not constant across all steps."""
        neighbors, V = _make_neighbors(V=6, topk=2, close_pairs=[(1, 3, 0.05)])
        torch.manual_seed(101)
        B, T = 1, 10
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 2, 3]])
        ylens = torch.tensor([3])
        hlens = torch.tensor([10])

        m = _make_model(
            neighbors, V, penalty_init=-15.0, penalty_final=-0.5, penalty_halflife=100
        )
        m.eval()

        losses = []
        for step in [0, 100, 500, 5000]:
            m._global_step = step
            loss = m(nnet, ys, hlens, ylens)
            losses.append(loss.item())

        # Loss should not be constant — the schedule should have an effect
        assert not all(
            abs(l - losses[0]) < 1e-6 for l in losses
        ), f"Loss did not change across steps: {losses}"


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_calls_identical(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        torch.manual_seed(55)
        B, T = 2, 10
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 2, 3, 0], [2, 5, 0, 0]])
        ylens = torch.tensor([3, 2])
        hlens = torch.tensor([10, 8])

        loss1 = m(nnet, ys, hlens, ylens)
        loss2 = m(nnet, ys, hlens, ylens)
        torch.testing.assert_close(loss1, loss2, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_token_transcript(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        nnet = torch.randn(1, 8, V).log_softmax(dim=-1)
        loss = m(nnet, torch.tensor([[3]]), torch.tensor([8]), torch.tensor([1]))
        assert loss.shape == (1,)
        assert torch.isfinite(loss).all()

    def test_repeated_tokens(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        # Repeated tokens need extra frames; ensure min_hlens filtering works
        nnet = torch.randn(1, 20, V).log_softmax(dim=-1)
        loss = m(
            nnet, torch.tensor([[1, 1, 1, 2, 2]]), torch.tensor([20]), torch.tensor([5])
        )
        assert loss.shape == (1,)
        assert torch.isfinite(loss).all()

    def test_mixed_valid_invalid_batch(self):
        """Some samples valid, some too short → invalid ones filtered, valid ones have loss."""
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        B = 3
        nnet = torch.randn(B, 10, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 1, 1, 1, 0], [2, 3, 0, 0, 0], [1, 2, 3, 0, 0]])
        ylens = torch.tensor([4, 2, 3])
        hlens = torch.tensor(
            [4, 10, 10]
        )  # first sample: min_hlen for [1,1,1,1] = 7 > 4

        loss = m(nnet, ys, hlens, ylens)
        # Should still return something (valid samples processed)
        assert torch.isfinite(loss).any()

    def test_long_transcript(self):
        neighbors, V = _make_neighbors(
            V=8, topk=3, close_pairs=[(1, 3, 0.2), (2, 5, 0.3), (4, 7, 0.15)]
        )
        m = _make_model(neighbors, V)
        m.eval()
        y = [1, 2, 3, 4, 1, 5, 2, 3, 7, 4, 5, 1, 2, 3, 4, 7, 5, 2, 1, 3]
        nnet = torch.randn(1, 50, V).log_softmax(dim=-1)
        loss = m(nnet, torch.tensor([y]), torch.tensor([50]), torch.tensor([len(y)]))
        assert loss.shape == (1,)
        assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# 9. Modified topo
# ---------------------------------------------------------------------------


class TestModifiedTopo:
    def test_modified_topo_runs(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V, modified_topo=True)
        m.eval()
        nnet = torch.randn(2, 10, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 3, 0], [2, 5, 0]])
        loss = m(nnet, ys, torch.tensor([10, 9]), torch.tensor([2, 2]))
        assert loss.shape == (2,)
        assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# 10. Language routing
# ---------------------------------------------------------------------------


class TestLanguageRouting:
    def test_unknown_lang_falls_back_to_global(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        B, T = 3, 10
        nnet = torch.randn(B, T, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 2, 0], [2, 5, 0], [1, 3, 4]])
        ylens = torch.tensor([2, 2, 3])
        hlens = torch.tensor([10, 9, 10])

        loss_global = m(
            nnet, ys, hlens, ylens, lang_per_utt=["global", "global", "global"]
        )
        loss_unknown = m(
            nnet, ys, hlens, ylens, lang_per_utt=["xx_UNK", "global", "yy_UNK"]
        )
        _assert_close(loss_global, loss_unknown)

    def test_none_lang_uses_global(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        nnet = torch.randn(1, 10, V).log_softmax(dim=-1)
        ys = torch.tensor([[1, 2, 3]])
        hlens, ylens = torch.tensor([10]), torch.tensor([3])

        loss_none = m(nnet, ys, hlens, ylens, lang_per_utt=None)
        loss_global = m(nnet, ys, hlens, ylens, lang_per_utt=["global"])
        _assert_close(loss_none, loss_global)


# ---------------------------------------------------------------------------
# 11. Extreme beta
# ---------------------------------------------------------------------------


class TestExtremeBeta:
    def test_very_large_beta(self):
        """Very large beta → neighbors so penalized they're ~invisible even at penalty=0."""
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V, beta=500.0, penalty_init=0.0, penalty_final=0.0)
        m.eval()
        nnet = torch.randn(1, 10, V).log_softmax(dim=-1)
        loss = m(nnet, torch.tensor([[1, 2, 3]]), torch.tensor([10]), torch.tensor([3]))
        assert torch.isfinite(loss).all()

    def test_very_small_beta(self):
        """Very small beta → neighbors nearly as good as self (before penalty)."""
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V, beta=0.01, penalty_init=0.0, penalty_final=0.0)
        m.eval()
        nnet = torch.randn(1, 10, V).log_softmax(dim=-1)
        loss = m(nnet, torch.tensor([[1, 2, 3]]), torch.tensor([10]), torch.tensor([3]))
        assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# 12. Gradient flow
# ---------------------------------------------------------------------------


class TestGradient:
    def test_gradient_flows_through_loss(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        raw = torch.randn(1, 10, V, requires_grad=True)
        nnet = raw.log_softmax(dim=-1)
        loss = m(nnet, torch.tensor([[1, 2, 3]]), torch.tensor([10]), torch.tensor([3]))
        loss.sum().backward()
        assert raw.grad is not None
        assert torch.isfinite(raw.grad).all()


# ---------------------------------------------------------------------------
# 13. Init validation
# ---------------------------------------------------------------------------


class TestInitValidation:
    def test_rejects_penalty_init_greater_than_final(self):
        neighbors, V = _make_neighbors()
        with pytest.raises(AssertionError):
            _make_model(neighbors, V, penalty_init=-1.0, penalty_final=-5.0)

    def test_rejects_positive_penalty(self):
        neighbors, V = _make_neighbors()
        with pytest.raises(AssertionError):
            _make_model(neighbors, V, penalty_init=-2.0, penalty_final=1.0)

    def test_rejects_zero_halflife(self):
        neighbors, V = _make_neighbors()
        with pytest.raises(AssertionError):
            _make_model(neighbors, V, penalty_halflife=0)

    def test_rejects_missing_global_key(self):
        dist = _make_dist_matrix(6)
        nids, ndists = matrix_to_neighbor_lists(dist, topk=2, blank_id=0)
        with pytest.raises(AssertionError):
            ScheduledArticulatoryCTC(
                neighbors_by_lang={"en": (nids, ndists)}, beta=10.0
            )


# ---------------------------------------------------------------------------
# 14. CUDA (if available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    def test_forward_on_cuda(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        device = torch.device("cuda")
        nnet = torch.randn(2, 10, V, device=device).log_softmax(dim=-1)
        ys = torch.tensor([[1, 2, 3, 0], [2, 5, 0, 0]], device=device)
        hlens = torch.tensor([10, 8], device=device)
        ylens = torch.tensor([3, 2], device=device)
        loss = m(nnet, ys, hlens, ylens)
        assert loss.is_cuda
        assert torch.isfinite(loss).all()

    def test_gradient_on_cuda(self):
        neighbors, V = _make_neighbors()
        m = _make_model(neighbors, V)
        m.eval()
        device = torch.device("cuda")
        raw = torch.randn(1, 10, V, device=device, requires_grad=True)
        nnet = raw.log_softmax(dim=-1)
        loss = m(
            nnet,
            torch.tensor([[1, 2]], device=device),
            torch.tensor([10], device=device),
            torch.tensor([2], device=device),
        )
        loss.sum().backward()
        assert raw.grad is not None
        assert raw.grad.is_cuda


# ---------------------------------------------------------------------------
# 15. Speed benchmark (not a correctness test — informational)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSpeedBenchmark:
    def test_benchmark(self):
        V, topk = 64, 10
        dist = _make_dist_matrix(
            V, [(i, (i + 1) % V, 0.1 + 0.01 * i) for i in range(1, V)]
        )
        nids, ndists = matrix_to_neighbor_lists(dist, topk=topk, blank_id=0)
        neighbors = {"global": (nids, ndists)}
        m = _make_model(
            neighbors,
            V,
            beta=10.0,
            penalty_init=-8.0,
            penalty_final=-0.5,
            penalty_halflife=8000,
        )
        m.eval()
        device = torch.device("cuda")

        B, T, Lmax = 16, 200, 50
        torch.manual_seed(0)
        nnet = torch.randn(B, T, V, device=device).log_softmax(dim=-1)
        ys = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        ylens = torch.randint(10, Lmax, (B,), device=device)
        for b in range(B):
            L = int(ylens[b].item())
            ys[b, :L] = torch.randint(1, V, (L,), device=device)
        hlens = torch.full((B,), T, dtype=torch.long, device=device)

        # Warmup
        for _ in range(5):
            m(nnet, ys, hlens, ylens)
        torch.cuda.synchronize()

        iters = 30
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            m(nnet, ys, hlens, ylens)
        end.record()
        torch.cuda.synchronize()

        ms_per_iter = start.elapsed_time(end) / iters
        print(
            f"\nScheduledArticulatoryCTC: {ms_per_iter:.2f} ms/iter "
            f"(B={B}, T={T}, V={V}, Lmax={Lmax})"
        )
