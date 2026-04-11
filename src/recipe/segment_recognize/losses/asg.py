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
from typing import List, Tuple

import torch
import torch.nn as nn

try:
    import k2  # type: ignore

    _K2_AVAILABLE = True
except Exception:
    k2 = None  # type: ignore
    _K2_AVAILABLE = False


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
        """DP forward: per-utterance numerator and denominator.

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

        losses = []
        for b in range(emissions.size(0)):
            e = emissions[b, : hlens[b].item()]
            num = self._numerator(e, ys[b], trans)
            den = self._denominator(e, trans)
            losses.append(-num + den)

        return torch.stack(losses)

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

        # Pad emissions with -inf at index 0 (k2 epsilon slot).
        em_k2 = emissions.new_full((B, T, V + 1), float("-inf"))
        em_k2[:, :, 1:] = emissions

        # Build transitions in k2 label space (shifted by 1).
        tr_k2 = em_k2.new_full((V + 1, V + 1), -1e10)
        tr_k2[1:, 1:] = self.transitions.to(dtype)

        segs = torch.stack([
            torch.tensor(
                [i, 0, int(hlens[i])],
                device=device, dtype=torch.int32,
            )
            for i in range(B)
        ])
        dense = k2.DenseFsaVec(em_k2, segs)

        # Denominator: any-path graph, same for all utterances.
        den_fsa = _build_den_graph(V, tr_k2)
        den_graphs = k2.create_fsa_vec(
            [den_fsa.clone() for _ in range(B)],
        ).to(device)
        den_scores = k2.get_tot_scores(
            k2.intersect_dense(
                den_graphs, dense, output_beam=10.0,
            ),
            log_semiring=True,
            use_double_scores=self.use_double_scores,
        )

        # Numerator: one constrained graph per utterance.
        num_graphs = k2.create_fsa_vec([
            _build_num_graph(
                [t + 1 for t in ys[b]], tr_k2,
            )
            for b in range(B)
        ]).to(device)
        num_scores = k2.get_tot_scores(
            k2.intersect_dense(
                num_graphs, dense, output_beam=10.0,
            ),
            log_semiring=True,
            use_double_scores=self.use_double_scores,
        )

        return -num_scores + den_scores


# -------------------------------------------------------------------
# k2 graph helpers
# -------------------------------------------------------------------


def _build_num_graph(
    target_k2: List[int],
    transitions_k2: torch.Tensor,
) -> "k2.Fsa":
    """Numerator FSA: allows paths spelling out *target_k2*.

    States 0 (start) → 1..L (target positions). At each position,
    a self-loop (stay) or advance arc to the next position.
    """
    L = len(target_k2)
    lines: List[str] = []
    lines.append(f"0 1 {target_k2[0]} 0.0")
    for i in range(L):
        si = i + 1
        yi = target_k2[i]
        lines.append(
            f"{si} {si} {yi} "
            f"{transitions_k2[yi, yi].item()}"
        )
        if i + 1 < L:
            yj = target_k2[i + 1]
            lines.append(
                f"{si} {i + 2} {yj} "
                f"{transitions_k2[yi, yj].item()}"
            )
    lines.append(f"{L} 0.0")
    return k2.Fsa.from_str("\n".join(lines), acceptor=True)


def _build_den_graph(
    num_labels: int,
    transitions_k2: torch.Tensor,
) -> "k2.Fsa":
    """Denominator FSA: allows any label sequence.

    State 0 (start) → states 1..V (one per label, all final).
    Full transitions between all label states.
    """
    V = num_labels
    lines: List[str] = []
    for j in range(1, V + 1):
        lines.append(f"0 {j} {j} 0.0")
    for i in range(1, V + 1):
        for j in range(1, V + 1):
            lines.append(
                f"{i} {j} {j} "
                f"{transitions_k2[i, j].item()}"
            )
    for j in range(1, V + 1):
        lines.append(f"{j} 0.0")
    return k2.Fsa.from_str("\n".join(lines), acceptor=True)


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
