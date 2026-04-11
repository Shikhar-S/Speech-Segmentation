"""Auto Segmentation Criterion (ASG) from Wav2Letter.

Reference: https://arxiv.org/pdf/1609.03193

ASG differs from CTC in three ways:

1. No blank labels -- uses repeat tokens for consecutive identical
   labels instead.
2. Un-normalized emission scores (logits, not log-probabilities).
3. Global normalization via a partition-function denominator.

Loss::

    ASG(θ, T) = - logadd_{π ∈ G_asg} Σ_t (f_πt(x) + g_{πt-1,πt})
               + logadd_{π ∈ G_full} Σ_t (f_πt(x) + g_{πt-1,πt})

- First term (numerator): promotes paths matching the transcription.
- Second term (denominator): normalizes over all possible paths.
- ``g_{i,j}``: learnable transition score from label *i* to *j*.

Two implementations:

- :class:`AutoSegmentationCriterion` -- pure-PyTorch DP (always
  available).
- :class:`AutoSegmentationCriterionK2` -- FSA-based via
  `k2 <https://github.com/k2-fsa/k2>`_.  Falls back to DP when
  k2 is not installed.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import k2  # type: ignore

    _K2_AVAILABLE = True
except Exception:
    k2 = None  # type: ignore
    _K2_AVAILABLE = False

try:
    from src.recipe.segment_recognize.losses.asg_triton import (
        asg_loss as _asg_loss_triton,
    )

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# -------------------------------------------------------------------
# Pure-PyTorch DP implementation
# -------------------------------------------------------------------


class AutoSegmentationCriterion(nn.Module):
    """ASG with global normalization and learnable transitions.

    Args:
        num_labels: Vocabulary size (including repeat token).
        repeat_idx: Index of the repeat token.
        use_transitions: Learn transition scores between labels.
        use_double_scores: Use float64 for numerical precision.
    """

    def __init__(
        self,
        num_labels: int,
        repeat_idx: int = 0,
        use_transitions: bool = True,
        use_double_scores: bool = True,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.repeat_idx = repeat_idx
        self.use_transitions = use_transitions
        self.use_double_scores = use_double_scores

        if use_transitions:
            self.transitions = nn.Parameter(
                torch.zeros(num_labels, num_labels),
            )
        else:
            self.register_buffer(
                "transitions",
                torch.zeros(num_labels, num_labels),
            )

    def forward(
        self,
        nnet_output: torch.Tensor,
        ys_pad: torch.Tensor,
        hlens: torch.Tensor,
        ylens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ASG loss.

        Args:
            nnet_output: ``(B, T, V)`` emission scores.
            ys_pad: ``(B, L)`` padded target sequences.
            hlens: ``(B,)`` emission lengths.
            ylens: ``(B,)`` target lengths.

        Returns:
            ``(B',)`` loss per valid utterance in original order.
        """
        indices = torch.argsort(hlens, descending=True)
        ys, min_hlens = self._preprocess_targets(
            ys_pad[indices], ylens[indices],
        )
        valid = (
            (min_hlens <= hlens[indices])
            .nonzero(as_tuple=True)[0]
        )
        if len(valid) < 1:
            logging.warning("All examples invalid for ASG.")
            return torch.tensor(
                [0.0], device=nnet_output.device,
            )

        indices = indices[valid]
        ys = [ys[i.item()] for i in valid]
        loss = self._forward_core(
            nnet_output[indices], ys, hlens[indices],
        )
        return loss[torch.argsort(indices)]

    def _forward_core(
        self,
        emissions: torch.Tensor,
        ys: List[List[int]],
        hlens: torch.Tensor,
    ) -> torch.Tensor:
        """Batched DP forward: numerator and denominator.

        Vectorises across both the batch dimension and label
        positions so the only remaining Python loop is over time
        steps (inherent sequential DP dependency).

        Args:
            emissions: ``(B, T, V)`` sorted/filtered emissions.
            ys: Preprocessed targets (with repeat tokens).
            hlens: ``(B,)`` emission lengths.

        Returns:
            ``(B,)`` loss per utterance.
        """
        dtype = (
            torch.float64
            if self.use_double_scores
            else emissions.dtype
        )
        emissions = emissions.to(dtype)
        trans = self.transitions.to(dtype)

        B = emissions.size(0)
        device = emissions.device
        ylens_list = [len(y) for y in ys]
        L_max = max(ylens_list)
        targets = torch.zeros(
            B, L_max, dtype=torch.long, device=device,
        )
        for b, y in enumerate(ys):
            targets[b, : len(y)] = torch.tensor(
                y, dtype=torch.long,
            )
        ylens = torch.tensor(
            ylens_list, dtype=torch.long, device=device,
        )

        num = self._batched_numerator(
            emissions, targets, ylens, hlens, trans,
        )
        den = self._batched_denominator(
            emissions, hlens, trans,
        )
        return -num + den

    def _batched_numerator(
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor,
        ylens: torch.Tensor,
        hlens: torch.Tensor,
        trans: torch.Tensor,
    ) -> torch.Tensor:
        """Constrained forward score, batched over utterances.

        ``alpha[b, i]`` = log-sum-exp score of all paths ending
        at target position ``i`` at the current time step, for
        utterance ``b``.

        Args:
            emissions: ``(B, T, V)`` emission scores.
            targets: ``(B, L_max)`` padded target indices.
            ylens: ``(B,)`` target lengths.
            hlens: ``(B,)`` emission lengths.
            trans: ``(V, V)`` transition scores.

        Returns:
            ``(B,)`` log-sum-exp numerator scores.
        """
        B, T, _ = emissions.shape
        L = targets.shape[1]
        NEG = torch.tensor(
            -1e10, device=emissions.device,
            dtype=emissions.dtype,
        )

        # Pre-gather emission scores at target positions.
        idx = targets.unsqueeze(1).expand(-1, T, -1)
        emit = emissions.gather(2, idx)          # (B, T, L)

        # Pre-compute transition scores.
        stay_tr = trans[targets, targets]         # (B, L)
        prev_t = torch.cat(
            [targets[:, :1], targets[:, :-1]], dim=1,
        )
        adv_tr = trans[prev_t, targets]           # (B, L)

        alpha = NEG.expand(B, L).clone()
        alpha[:, 0] = emit[:, 0, 0]

        for t in range(1, T):
            stay = alpha + stay_tr
            adv = NEG.expand(B, L).clone()
            adv[:, 1:] = alpha[:, :-1] + adv_tr[:, 1:]
            alpha_next = (
                torch.logaddexp(stay, adv)
                + emit[:, t, :]
            )
            mask = (t < hlens).unsqueeze(1)
            alpha = torch.where(mask, alpha_next, alpha)

        return alpha[
            torch.arange(B, device=alpha.device),
            ylens - 1,
        ]

    def _batched_denominator(
        self,
        emissions: torch.Tensor,
        hlens: torch.Tensor,
        trans: torch.Tensor,
    ) -> torch.Tensor:
        """Partition function, batched over utterances.

        ``beta[b, j]`` = log-sum-exp score of all paths ending
        in label ``j`` at the current time step, for utterance
        ``b``.

        Args:
            emissions: ``(B, T, V)`` emission scores.
            hlens: ``(B,)`` emission lengths.
            trans: ``(V, V)`` transition scores.

        Returns:
            ``(B,)`` log-partition scores.
        """
        B, T, _ = emissions.shape
        beta = emissions[:, 0, :]                # (B, V)

        for t in range(1, T):
            beta_next = (
                torch.logsumexp(
                    beta.unsqueeze(2) + trans, dim=1,
                )
                + emissions[:, t, :]
            )
            mask = (t < hlens).unsqueeze(1)
            beta = torch.where(mask, beta_next, beta)

        return torch.logsumexp(beta, dim=1)

    def _numerator(
        self,
        emissions: torch.Tensor,
        target: List[int],
        trans: torch.Tensor,
    ) -> torch.Tensor:
        """Constrained forward score over paths matching *target*.

        Uses the forward algorithm on the ASG graph ``G_asg``.
        ``alpha[t, i]`` = log-sum-exp score of all paths of length
        ``t+1`` ending at target position ``i``.

        Args:
            emissions: ``(T, V)`` single-utterance emissions.
            target: Target labels with repeat tokens inserted.
            trans: ``(V, V)`` transition scores.

        Returns:
            Scalar log-sum-exp score.
        """
        T = emissions.size(0)
        L = len(target)
        NEG_INF = torch.tensor(
            -1e10, device=emissions.device, dtype=emissions.dtype,
        )
        alpha = NEG_INF.expand(T, L).clone()
        alpha[0, 0] = emissions[0, target[0]]

        for t in range(1, T):
            for i in range(L):
                stay = (
                    alpha[t - 1, i]
                    + trans[target[i], target[i]]
                )
                if i > 0:
                    advance = (
                        alpha[t - 1, i - 1]
                        + trans[target[i - 1], target[i]]
                    )
                    alpha[t, i] = (
                        torch.logaddexp(stay, advance)
                        + emissions[t, target[i]]
                    )
                else:
                    alpha[t, i] = (
                        stay + emissions[t, target[i]]
                    )

        return alpha[T - 1, L - 1]

    def _denominator(
        self,
        emissions: torch.Tensor,
        trans: torch.Tensor,
    ) -> torch.Tensor:
        """Partition function over all label paths (``G_full``).

        ``beta[t, j]`` = log-sum-exp score of all paths ending in
        label ``j`` at time ``t``.

        Args:
            emissions: ``(T, V)`` single-utterance emissions.
            trans: ``(V, V)`` transition scores.

        Returns:
            Scalar log-sum-exp score.
        """
        T, V = emissions.size()
        beta = torch.full(
            (T, V),
            float("-inf"),
            device=emissions.device,
            dtype=emissions.dtype,
        )
        beta[0] = emissions[0]

        for t in range(1, T):
            beta[t] = (
                torch.logsumexp(
                    beta[t - 1, :, None] + trans, dim=0,
                )
                + emissions[t]
            )

        return torch.logsumexp(beta[T - 1], dim=0)

    def _preprocess_targets(
        self,
        ys_pad: torch.Tensor,
        ylens: torch.Tensor,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Replace consecutive duplicate labels with repeat tokens.

        ``[a, a, b]`` becomes ``[a, repeat, b]``.

        Returns:
            ``(ys, min_hlens)`` -- preprocessed targets and the
            minimum emission length each requires.
        """
        device = ys_pad.device
        ys: List[List[int]] = []
        min_hlens: List[int] = []
        for y_pad, ylen in zip(
            ys_pad.cpu().tolist(), ylens.cpu().tolist(),
        ):
            y: List[int] = []
            prev = None
            for i in range(ylen):
                tok = y_pad[i]
                y.append(
                    self.repeat_idx if tok == prev else tok,
                )
                prev = tok
            ys.append(y)
            min_hlens.append(len(y))
        return ys, torch.tensor(
            min_hlens, dtype=torch.long, device=device,
        )


# -------------------------------------------------------------------
# Triton-fused implementation
# -------------------------------------------------------------------


class AutoSegmentationCriterionTriton(AutoSegmentationCriterion):
    """Triton-fused ASG. Falls back to DP when triton is absent."""

    def _forward_core(
        self,
        emissions: torch.Tensor,
        ys: List[List[int]],
        hlens: torch.Tensor,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            return super()._forward_core(
                emissions, ys, hlens,
            )

        B = emissions.size(0)
        device = emissions.device
        ylens_list = [len(y) for y in ys]
        L_max = max(ylens_list)
        targets = torch.zeros(
            B, L_max, dtype=torch.long, device=device,
        )
        for b, y in enumerate(ys):
            targets[b, : len(y)] = torch.tensor(
                y, dtype=torch.long,
            )
        ylens_t = torch.tensor(
            ylens_list, dtype=torch.long, device=device,
        )

        return _asg_loss_triton(
            emissions, targets, hlens, ylens_t,
            self.transitions,
        )


# -------------------------------------------------------------------
# k2-based implementation
# -------------------------------------------------------------------


class AutoSegmentationCriterionK2(AutoSegmentationCriterion):
    """k2-based ASG. Falls back to DP when k2 is not installed."""

    def _forward_core(
        self,
        emissions: torch.Tensor,
        ys: List[List[int]],
        hlens: torch.Tensor,
    ) -> torch.Tensor:
        if not _K2_AVAILABLE:
            return super()._forward_core(emissions, ys, hlens)

        B, T, V = emissions.shape
        device = emissions.device
        dtype = (
            torch.float64
            if self.use_double_scores
            else emissions.dtype
        )
        emissions = emissions.to(dtype)

        # Pad emissions with -inf at index 0 (k2 epsilon).
        em_k2 = emissions.new_full(
            (B, T, V + 1), float("-inf"),
        )
        em_k2[:, :, 1:] = emissions

        # Differentiable transition matrix in k2 label space.
        tr_k2 = F.pad(
            self.transitions.to(dtype),
            (1, 0, 1, 0),
            value=-1e10,
        )

        segs = torch.zeros(B, 3, dtype=torch.int32)
        segs[:, 0] = torch.arange(B, dtype=torch.int32)
        segs[:, 2] = hlens.cpu().to(torch.int32)
        dense = k2.DenseFsaVec(em_k2, segs)

        # Denominator: any-path graph, same for all utts.
        den_fsa = _build_den_graph(V).to(device)
        _set_differentiable_scores(den_fsa, tr_k2, V)
        den_graphs = k2.create_fsa_vec(
            [den_fsa.clone() for _ in range(B)],
        )
        den_lattice = k2.intersect_dense(
            den_graphs, dense, output_beam=10.0,
        )
        den_scores = den_lattice.get_tot_scores(
            log_semiring=True,
            use_double_scores=self.use_double_scores,
        )

        # Numerator: one constrained graph per utterance.
        num_fsas = []
        for b in range(B):
            tgt_k2 = [t + 1 for t in ys[b]]
            nfsa = _build_num_graph(tgt_k2).to(device)
            tgt_t = torch.tensor(
                [0] + tgt_k2,
                device=device,
                dtype=torch.long,
            )
            _set_differentiable_scores(
                nfsa, tr_k2, V, tgt_t,
            )
            num_fsas.append(nfsa)
        num_graphs = k2.create_fsa_vec(num_fsas)
        num_lattice = k2.intersect_dense(
            num_graphs, dense, output_beam=10.0,
        )
        num_scores = num_lattice.get_tot_scores(
            log_semiring=True,
            use_double_scores=self.use_double_scores,
        )

        return -num_scores + den_scores


# -------------------------------------------------------------------
# k2 graph helpers
# -------------------------------------------------------------------


def _set_differentiable_scores(
    fsa: "k2.Fsa",
    tr_k2: torch.Tensor,
    num_labels: int,
    target_tensor: Optional[torch.Tensor] = None,
) -> None:
    """Replace FSA arc scores with differentiable transitions.

    For **denominator** graphs (``target_tensor is None``),
    states 1..V directly represent k2-space labels, so the
    source state index *is* the ``from`` label.

    For **numerator** graphs, ``target_tensor`` maps each state
    index to the k2-space label at that target position
    (``target_tensor[0] = 0`` as a dummy for the start state).

    Args:
        fsa: FSA on the target device (topology-only, all
            scores 0).
        tr_k2: ``(V+1, V+1)`` padded transition matrix,
            connected to the autograd graph.
        num_labels: Original vocabulary size *V*.
        target_tensor: Optional ``(L+1,)`` label map for
            numerator graphs.
    """
    V = num_labels
    arcs = fsa.arcs.values()
    src = arcs[:, 0].long()
    lbl = arcs[:, 2].long()

    if target_tensor is not None:
        L = target_tensor.shape[0] - 1
        internal = (src > 0) & (src <= L) & (lbl > 0)
        from_lbl = target_tensor[src.clamp(0, L)]
    else:
        internal = (
            (src > 0) & (src <= V) & (lbl > 0)
        )
        from_lbl = src.clamp(0, V)

    scores = (
        tr_k2[from_lbl, lbl.clamp(0, V)].float()
        * internal.float()
    )
    fsa.scores = scores


def _build_num_graph(
    target_k2: List[int],
) -> "k2.Fsa":
    """Numerator topology for *target_k2* (all scores 0).

    States 0 (start) -> 1..L (target positions) -> L+1 (final).
    """
    L = len(target_k2)
    final = L + 1
    lines: List[str] = []
    lines.append(f"0 1 {target_k2[0]} 0.0")
    for i in range(L):
        si = i + 1
        yi = target_k2[i]
        lines.append(f"{si} {si} {yi} 0.0")
        if i + 1 < L:
            yj = target_k2[i + 1]
            lines.append(f"{si} {i + 2} {yj} 0.0")
    lines.append(f"{L} {final} -1 0.0")
    lines.append(f"{final}")
    return k2.Fsa.from_str(
        "\n".join(lines), acceptor=True,
    )


def _build_den_graph(num_labels: int) -> "k2.Fsa":
    """Denominator topology (all scores 0).

    State 0 (start) -> states 1..V -> V+1 (final).
    """
    V = num_labels
    final = V + 1
    lines: List[str] = []
    for j in range(1, V + 1):
        lines.append(f"0 {j} {j} 0.0")
    for i in range(1, V + 1):
        for j in range(1, V + 1):
            lines.append(f"{i} {j} {j} 0.0")
        lines.append(f"{i} {final} -1 0.0")
    lines.append(f"{final}")
    return k2.Fsa.from_str(
        "\n".join(lines), acceptor=True,
    )


if __name__ == "__main__":
    print("Testing AutoSegmentationCriterion...")

    V, B, T = 5, 2, 10
    model = AutoSegmentationCriterion(
        num_labels=V, use_double_scores=True,
    )
    nnet_output = torch.randn(B, T, V)
    ys_pad = torch.tensor([[1, 2, 3, 0, 0], [2, 2, 4, 3, 0]])
    hlens = torch.tensor([10, 8])
    ylens = torch.tensor([3, 4])

    loss = model(nnet_output, ys_pad, hlens, ylens)
    loss.sum().backward()
    print(f"Loss: {loss}")
    print(f"Transition grad norm: "
          f"{model.transitions.grad.norm():.4f}")
    print("Passed!")
