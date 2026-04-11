"""Tests for AutoSegmentationCriterion (ASG).

Correctness is established by a brute-force reference that enumerates
all V^T label paths, computes each path's score from the ASG formula,
and partitions into valid (matching target) and all paths.  This is
tractable for tiny V and T and directly verifies the DP against the
mathematical definition.
"""

import itertools
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.recipe.segment_recognize.losses import asg as asg_module
from src.recipe.segment_recognize.losses.asg import (
    AutoSegmentationCriterion,
    AutoSegmentationCriterionK2,
)


# -- Brute-force reference ------------------------------------


def _path_score(path, emissions, trans):
    """Score of a single label path under the ASG model.

    ``emissions[0, π₀] + Σ_{t=1}^{T-1} (trans[π_{t-1}, π_t]
    + emissions[t, π_t])``
    """
    s = emissions[0, path[0]]
    for t in range(1, len(path)):
        s = s + trans[path[t - 1], path[t]] + emissions[t, path[t]]
    return s


def _is_valid_path(path, target):
    """True iff *path* is a monotone alignment of *target*.

    At each step the path must stay on the current target
    position or advance to the next.  Must end at the last
    target position.
    """
    if path[0] != target[0]:
        return False
    pos = 0
    for t in range(1, len(path)):
        if path[t] == target[pos]:
            continue
        elif pos + 1 < len(target) and path[t] == target[pos + 1]:
            pos += 1
        else:
            return False
    return pos == len(target) - 1


def _brute_force_asg(emissions, target, trans):
    """Brute-force ASG loss for a single utterance.

    Enumerates all V^T paths.  Only feasible for tiny V and T.
    """
    T, V = emissions.shape
    all_s, num_s = [], []
    for path in itertools.product(range(V), repeat=T):
        s = _path_score(path, emissions, trans)
        all_s.append(s)
        if _is_valid_path(path, target):
            num_s.append(s)
    den = torch.logsumexp(torch.stack(all_s), dim=0)
    num = torch.logsumexp(torch.stack(num_s), dim=0)
    return -num + den


def _preprocess_target(target, repeat_idx=0):
    """Mirror the model's repeat-token preprocessing."""
    out, prev = [], None
    for t in target:
        out.append(repeat_idx if t == prev else t)
        prev = t
    return out


# -- Helpers ---------------------------------------------------


def _make_model(**kw):
    defaults = dict(
        num_labels=3,
        repeat_idx=0,
        use_transitions=True,
        use_double_scores=False,
    )
    defaults.update(kw)
    return AutoSegmentationCriterion(**defaults)


# -- Brute-force parity tests ---------------------------------


def test_brute_force_zero_trans():
    """DP matches brute-force enumeration with zero transitions."""
    V, T = 3, 4
    torch.manual_seed(0)
    emissions = torch.randn(1, T, V)
    target_raw = [1, 2]
    targets = torch.tensor([target_raw])
    hlens = torch.tensor([T])
    ylens = torch.tensor([len(target_raw)])

    model = _make_model(
        num_labels=V, use_transitions=False,
    )
    loss = model(emissions, targets, hlens, ylens)

    preprocessed = _preprocess_target(target_raw)
    ref = _brute_force_asg(
        emissions[0], preprocessed, model.transitions,
    )
    torch.testing.assert_close(loss[0], ref, atol=1e-5, rtol=1e-5)


def test_brute_force_with_trans():
    """DP matches brute-force with random transitions."""
    V, T = 3, 4
    torch.manual_seed(1)
    emissions = torch.randn(1, T, V)
    target_raw = [1, 2]

    model = _make_model(num_labels=V)
    with torch.no_grad():
        model.transitions.normal_(std=0.5)

    loss = model(
        emissions,
        torch.tensor([target_raw]),
        torch.tensor([T]),
        torch.tensor([len(target_raw)]),
    )

    preprocessed = _preprocess_target(target_raw)
    ref = _brute_force_asg(
        emissions[0], preprocessed, model.transitions,
    )
    torch.testing.assert_close(loss[0], ref, atol=1e-5, rtol=1e-5)


def test_brute_force_repeat_target():
    """Brute-force parity with consecutive-duplicate target.

    Target [1, 1, 2] preprocesses to [1, repeat(0), 2].  Valid
    paths must spell out [1, 0, 2] as a monotone alignment.
    """
    V, T = 3, 5
    torch.manual_seed(2)
    emissions = torch.randn(1, T, V)
    target_raw = [1, 1, 2]

    model = _make_model(num_labels=V)
    with torch.no_grad():
        model.transitions.normal_(std=0.3)

    loss = model(
        emissions,
        torch.tensor([target_raw]),
        torch.tensor([T]),
        torch.tensor([len(target_raw)]),
    )

    preprocessed = _preprocess_target(target_raw)
    ref = _brute_force_asg(
        emissions[0], preprocessed, model.transitions,
    )
    torch.testing.assert_close(loss[0], ref, atol=1e-5, rtol=1e-5)


# -- Mathematical identity tests ------------------------------


def test_zero_trans_is_framewise_ce():
    """With zero transitions and single-label target, ASG = Σ CE.

    Numerator = Σ_t emissions[t, a]  (only valid path is all-a).
    Denominator = Σ_t logsumexp(emissions[t, :])  (factorises).
    So loss = Σ_t (logsumexp(emissions[t,:]) - emissions[t,a])
            = Σ_t CE(emissions[t], a).
    """
    V, T = 5, 8
    torch.manual_seed(3)
    emissions = torch.randn(1, T, V)
    a = 2

    model = _make_model(
        num_labels=V, use_transitions=False,
    )
    loss = model(
        emissions,
        torch.tensor([[a]]),
        torch.tensor([T]),
        torch.tensor([1]),
    )

    ce = F.cross_entropy(
        emissions[0],
        torch.full((T,), a, dtype=torch.long),
        reduction="sum",
    )
    torch.testing.assert_close(loss[0], ce, atol=1e-5, rtol=1e-5)


# -- Mathematical invariant tests -----------------------------


def test_loss_nonnegative():
    """ASG loss ≥ 0 (denominator ⊇ numerator paths)."""
    V, T = 4, 10
    torch.manual_seed(4)
    model = _make_model(num_labels=V)
    with torch.no_grad():
        model.transitions.normal_()
    emissions = torch.randn(2, T, V)
    targets = torch.tensor([[1, 2, 3], [2, 1, 0]])
    hlens = torch.tensor([T, T])
    ylens = torch.tensor([3, 3])

    loss = model(emissions, targets, hlens, ylens)
    assert (loss >= -1e-6).all(), f"Negative loss: {loss}"


def test_perfect_emissions_low_loss():
    """Strong emissions toward target drive loss near zero."""
    V, T = 4, 6
    target_raw = [1, 2, 3]
    # Construct emissions that strongly favor the target path:
    # first 2 frames → label 1, next 2 → label 2, last 2 → label 3
    emissions = torch.full((1, T, V), -50.0)
    for t in range(T):
        label = target_raw[min(t // 2, len(target_raw) - 1)]
        emissions[0, t, label] = 50.0

    model = _make_model(
        num_labels=V, use_transitions=False,
    )
    loss = model(
        emissions,
        torch.tensor([target_raw]),
        torch.tensor([T]),
        torch.tensor([len(target_raw)]),
    )
    assert loss[0].item() < 1e-3


def test_gradient_step_lowers_loss():
    """One gradient step on transitions lowers the loss.

    Computes loss, takes a gradient step on transitions, then
    verifies the new loss is strictly lower.
    """
    V, T = 4, 8
    torch.manual_seed(5)
    emissions = torch.randn(1, T, V)
    args = (
        emissions,
        torch.tensor([[1, 2, 3]]),
        torch.tensor([T]),
        torch.tensor([3]),
    )

    model = _make_model(num_labels=V)
    loss_before = model(*args)[0].item()

    loss_before_t = model(*args)
    loss_before_t.sum().backward()
    with torch.no_grad():
        model.transitions -= 0.1 * model.transitions.grad
    model.transitions.grad = None

    loss_after = model(*args)[0].item()
    assert loss_after < loss_before


# -- Gradient tests --------------------------------------------


def test_gradient_finite():
    """Backward produces finite gradients on emissions and transitions."""
    V, T = 4, 6
    torch.manual_seed(6)
    model = _make_model(num_labels=V)
    emissions = torch.randn(1, T, V, requires_grad=True)
    loss = model(
        emissions,
        torch.tensor([[1, 2]]),
        torch.tensor([T]),
        torch.tensor([2]),
    )
    loss.sum().backward()

    assert emissions.grad is not None
    assert emissions.grad.isfinite().all()
    assert model.transitions.grad is not None
    assert model.transitions.grad.isfinite().all()


# -- Implementation correctness tests -------------------------


def test_batch_order_invariance():
    """Per-utterance losses don't depend on batch ordering."""
    V, T = 4, 8
    torch.manual_seed(7)
    model = _make_model(num_labels=V)
    with torch.no_grad():
        model.transitions.normal_(std=0.3)

    emissions = torch.randn(3, T, V)
    targets = torch.tensor([[1, 2, 0], [2, 3, 0], [1, 3, 0]])
    hlens = torch.tensor([T, 6, 7])
    ylens = torch.tensor([2, 2, 2])

    loss_fwd = model(emissions, targets, hlens, ylens)

    perm = [2, 0, 1]
    loss_perm = model(
        emissions[perm],
        targets[perm],
        hlens[perm],
        ylens[perm],
    )

    torch.testing.assert_close(
        loss_fwd, loss_perm[[1, 2, 0]], atol=1e-5, rtol=1e-5,
    )


def test_batched_mixed_lengths():
    """Batched DP matches per-utterance DP with varied hlens/ylens."""
    V, T = 4, 10
    torch.manual_seed(12)
    model = _make_model(num_labels=V)
    with torch.no_grad():
        model.transitions.normal_(std=0.4)

    emissions = torch.randn(4, T, V)
    targets = torch.tensor([
        [1, 2, 3, 0],
        [2, 1, 0, 0],
        [3, 2, 1, 1],
        [1, 0, 0, 0],
    ])
    hlens = torch.tensor([T, 7, 9, 5])
    ylens = torch.tensor([3, 2, 4, 1])

    # Batched result from the model
    loss_batched = model(emissions, targets, hlens, ylens)

    # Per-utterance reference using private methods
    dtype = (
        torch.float64
        if model.use_double_scores
        else emissions.dtype
    )
    em = emissions.to(dtype)
    trans = model.transitions.to(dtype)
    ys, _ = model._preprocess_targets(targets, ylens)
    ref = []
    for b in range(4):
        e = em[b, : hlens[b].item()]
        num = model._numerator(e, ys[b], trans)
        den = model._denominator(e, trans)
        ref.append(-num + den)
    loss_ref = torch.stack(ref)

    torch.testing.assert_close(
        loss_batched, loss_ref, atol=1e-5, rtol=1e-5,
    )


def test_k2_fallback_dp_parity():
    """K2 fallback (DP path) produces identical results to DP."""
    V, T = 4, 6
    torch.manual_seed(8)
    dp = _make_model(num_labels=V)
    with torch.no_grad():
        dp.transitions.normal_(std=0.5)

    # Force fallback to DP even if k2 is installed.
    with patch.object(asg_module, "_K2_AVAILABLE", False):
        k2m = AutoSegmentationCriterionK2(
            num_labels=V, use_transitions=True,
            use_double_scores=False,
        )
        with torch.no_grad():
            k2m.transitions.copy_(dp.transitions)

        emissions = torch.randn(2, T, V)
        targets = torch.tensor([[1, 2, 0], [3, 1, 0]])
        hlens = torch.tensor([T, 5])
        ylens = torch.tensor([2, 2])

        loss_dp = dp(emissions, targets, hlens, ylens)
        loss_k2 = k2m(emissions, targets, hlens, ylens)

    torch.testing.assert_close(
        loss_dp, loss_k2, atol=1e-5, rtol=1e-5,
    )


def test_gradient_flows_to_emissions():
    """Gradients flow to emission tensor (non-zero everywhere)."""
    V, T = 4, 6
    torch.manual_seed(10)
    model = _make_model(num_labels=V)
    emissions = torch.randn(1, T, V, requires_grad=True)
    loss = model(
        emissions,
        torch.tensor([[1, 2]]),
        torch.tensor([T]),
        torch.tensor([2]),
    )
    loss.sum().backward()
    assert emissions.grad is not None
    # Every frame should receive gradient since denominator
    # covers all labels at all frames.
    assert (emissions.grad[:, :T, :].abs() > 0).all()


def test_gradient_step_on_emissions_lowers_loss():
    """One gradient step on emissions lowers the loss."""
    V, T = 4, 6
    torch.manual_seed(11)
    model = _make_model(
        num_labels=V, use_transitions=False,
    )
    emissions = torch.randn(1, T, V)
    args = lambda e: (
        e,
        torch.tensor([[1, 2]]),
        torch.tensor([T]),
        torch.tensor([2]),
    )

    e = emissions.clone().requires_grad_(True)
    loss_before = model(*args(e))
    loss_before.sum().backward()
    with torch.no_grad():
        e_new = e - 0.5 * e.grad

    loss_after = model(*args(e_new))
    assert loss_after[0].item() < loss_before[0].item()


def test_all_invalid_returns_zero():
    """All utterances shorter than target → returns [0.0]."""
    model = _make_model(num_labels=4)
    loss = model(
        torch.randn(2, 3, 4),
        torch.tensor([[1, 2, 3, 1], [2, 3, 1, 2]]),
        torch.tensor([2, 1]),  # shorter than targets
        torch.tensor([4, 4]),
    )
    assert loss.shape == (1,)
    assert loss.item() == 0.0


# -- k2 implementation tests ----------------------------------

_k2 = None
try:
    import k2 as _k2  # noqa: F811
except ImportError:
    pass

_skip_no_k2 = (
    _k2 is None,
    "k2 not installed",
)


def _make_k2_model(**kw):
    defaults = dict(
        num_labels=3,
        repeat_idx=0,
        use_transitions=True,
        use_double_scores=False,
    )
    defaults.update(kw)
    return AutoSegmentationCriterionK2(**defaults)


@torch.no_grad()
def _share_transitions(src, dst):
    dst.transitions.copy_(src.transitions)


def test_k2_dp_parity_real():
    """k2 path matches DP when k2 IS available."""
    if _skip_no_k2[0]:
        __import__("pytest").skip(_skip_no_k2[1])

    V, T = 3, 4
    torch.manual_seed(20)

    dp = _make_model(num_labels=V)
    with torch.no_grad():
        dp.transitions.normal_(std=0.5)

    k2m = _make_k2_model(num_labels=V)
    _share_transitions(dp, k2m)

    emissions = torch.randn(2, T, V)
    targets = torch.tensor([[1, 2, 0], [2, 1, 0]])
    hlens = torch.tensor([T, 3])
    ylens = torch.tensor([2, 2])

    loss_dp = dp(emissions, targets, hlens, ylens)
    loss_k2 = k2m(emissions, targets, hlens, ylens)
    torch.testing.assert_close(
        loss_dp, loss_k2, atol=1e-3, rtol=1e-3,
    )


def test_k2_gradient_flows_to_transitions():
    """Gradients flow to transitions through the k2 path."""
    if _skip_no_k2[0]:
        __import__("pytest").skip(_skip_no_k2[1])

    V, T = 3, 5
    torch.manual_seed(21)
    model = _make_k2_model(num_labels=V)
    with torch.no_grad():
        model.transitions.normal_(std=0.3)

    emissions = torch.randn(1, T, V, requires_grad=True)
    loss = model(
        emissions,
        torch.tensor([[1, 2]]),
        torch.tensor([T]),
        torch.tensor([2]),
    )
    loss.sum().backward()

    assert model.transitions.grad is not None
    assert model.transitions.grad.isfinite().all()
    assert (model.transitions.grad.abs() > 0).any(), (
        "transition gradients are all zero"
    )
    assert emissions.grad is not None
    assert emissions.grad.isfinite().all()


def test_k2_brute_force():
    """k2 ASG matches brute-force reference."""
    if _skip_no_k2[0]:
        __import__("pytest").skip(_skip_no_k2[1])

    V, T = 3, 4
    torch.manual_seed(22)
    emissions = torch.randn(1, T, V)
    target_raw = [1, 2]

    model = _make_k2_model(num_labels=V)
    with torch.no_grad():
        model.transitions.normal_(std=0.5)

    loss = model(
        emissions,
        torch.tensor([target_raw]),
        torch.tensor([T]),
        torch.tensor([len(target_raw)]),
    )

    preprocessed = _preprocess_target(target_raw)
    ref = _brute_force_asg(
        emissions[0], preprocessed, model.transitions,
    )
    torch.testing.assert_close(
        loss[0], ref, atol=1e-3, rtol=1e-3,
    )
