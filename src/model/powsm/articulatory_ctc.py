"""CTC loss that uses articulatory distance-based soft targets.

Usage:
    python -m src.model.powsm.articulatory_ctc
"""

import logging
from typing import Dict, List, Optional, Tuple

import k2
import torch


class ArticulatoryCTC(torch.nn.Module):
    """
    Soft CTC via a weighted transcript lattice:
      - At each target position i with phone p = y[i],
        allow emitting candidates c in TopKNeighbors(p) with score = log_prior(p->c).
      - Compose that transcript lattice with CTC topology, then compute CTC loss with k2.

    Contract matches your BayesRiskCTC:
      forward(nnet_output, ys_pad, hlens, ylens) -> loss_utt (valid-only, original order)
    """

    def __init__(
        self,
        neighbors_by_lang: Dict[str, Tuple[torch.Tensor, torch.Tensor]],  # lang -> (neighbor_ids (V,k-1), neighbor_dists (V,k-1))
        beta: float = 50.0,  # larger => harsher penalty for far substitutions
        topk: int = 5,  # number of allowed substitutions per target token (must match k-1 in tensors + 1)
        normalize: bool = True,  # if True, per-position substitution weights are log-softmaxed
        modified_topo: bool = False,  # if True, skip blanks in CTC topology, efficient but approx
        output_beam: float = 200,
        use_double_scores: bool = True,  # float scores in double precision
        label_smoothing: float = 0.0,  # if >0: force (1-label_smoothing) mass on self token
    ):
        super().__init__()
        assert "global" in neighbors_by_lang, "neighbors_by_lang must contain key 'global'"
        self._neighbors_by_lang = neighbors_by_lang
        nids, _ = next(iter(neighbors_by_lang.values()))
        self._V = nids.size(0)
        # Second dim can vary by lang (e.g. 1 for identity fallback); we use actual tensor.size(1) at lookup
        self.beta = float(beta)
        self.topk = int(topk)
        self.normalize = bool(normalize)
        self.modified_topo = bool(modified_topo)
        self.output_beam = float(output_beam)
        self.use_double_scores = bool(use_double_scores)

        self.label_smoothing = float(label_smoothing)
        assert 0.0 <= self.label_smoothing < 1.0, "label_smoothing must be in [0, 1)."

        self._topo = None  # cached per-device topo

    def forward(self, nnet_output, ys_pad, hlens, ylens, lang_per_utt: Optional[List[str]] = None):
        # Reorder and filter invalid examples
        indices = torch.argsort(hlens, descending=True)
        ys, min_hlens = self.find_minimum_hlens(ys_pad[indices], ylens[indices])
        valid_sample_indices = (min_hlens <= hlens[indices]).nonzero(as_tuple=True)[0]
        if len(valid_sample_indices) < 1:
            logging.warning(
                "All examples are invalid for ArticulatoryCTC. Skip this batch"
            )
            return torch.Tensor([0.0]).to(nnet_output.device)
        indices = indices[valid_sample_indices]
        nnet_output, hlens, ylens = nnet_output[indices], hlens[indices], ylens[indices]
        ys = [ys[i.item()] for i in valid_sample_indices]  # list[list[int]]
        if lang_per_utt is not None:
            # indices at this point are original batch indices of the valid samples
            lang_per_utt = [lang_per_utt[indices[i].item()] for i in range(len(indices))]
        loss_utt = self.forward_core(nnet_output, ys, hlens, ylens, lang_per_utt=lang_per_utt)
        # Recover original order for the remaining valid examples
        indices2 = torch.argsort(indices)
        loss_utt = loss_utt[indices2]
        return loss_utt

    def forward_core(
        self,
        nnet_output,
        ys: List[List[int]],
        hlens,
        ylens,
        lang_per_utt: Optional[List[str]] = None,
    ):
        """
        nnet_output: (B, T, V) log-probs (log_softmaxed)
        ys: list of target sequences (already stripped of padding), tokens in [0..V-1] but normally non-blank
        """
        device = nnet_output.device
        B, T, V = nnet_output.size()

        # (1) DenseFsaVec
        supervision = torch.stack(
            [torch.arange(B), torch.zeros(B), hlens.cpu()], dim=1
        ).int()
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)

        # (2) Build decoding graphs: compose(CTC_topo, weighted_transcript)
        topo = self._get_topo(V=V, device=device)  # FSA on correct device

        graphs = []
        for b, y in enumerate(ys):
            lang = (lang_per_utt[b] if (lang_per_utt and b < len(lang_per_utt)) else None) or "global"
            nids, ndists = self._neighbors_by_lang.get(lang, self._neighbors_by_lang["global"])
            nids = nids.to(device)
            ndists = ndists.to(device)
            # IMPORTANT: CTC topo handles blank (0); transcript should NOT include blank.
            y = [int(t) for t in y if int(t) != 0]
            if len(y) == 0:
                # Empty transcript: build trivial acceptor with final arc
                transcript = k2.Fsa.from_str("0 1 -1 0.0\n1").to(device)
            else:
                transcript = self._build_weighted_transcript_fsa(
                    y, V=V, device=device, neighbor_ids=nids, neighbor_dists=ndists
                )

            transcript = k2.arc_sort(k2.add_epsilon_self_loops(transcript))
            g = k2.compose(topo, transcript, treat_epsilons_specially=False)
            g = k2.arc_sort(g)
            graphs.append(g)

        decoding_graphs = k2.create_fsa_vec(graphs)

        # (3) Loss per utterance (no reduction)
        # k2.ctc_loss returns a vector if reduction="none"
        loss_utt = k2.ctc_loss(
            decoding_graph=decoding_graphs,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=self.output_beam,
            reduction="none",
            use_double_scores=self.use_double_scores,
        )
        return loss_utt

    def _get_topo(self, V: int, device: torch.device) -> k2.Fsa:
        # Cache topo per device (k2 FSAs are device-specific)
        if self._topo is None or str(self._topo.device) != str(device):
            # max_token is the max *non-blank* label. If blank=0, tokens are 1..V-1
            self._topo = k2.ctc_topo(
                max_token=V - 1, modified=self.modified_topo, device=device
            )
        return self._topo

    def _rescore_candidates(
        self,
        p: int,
        cand: List[int],
        dist_list: List[float],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns per-candidate arc scores aligned with `cand`.
        dist_list[i] is the distance for cand[i]; self has distance 0.

        Base logits: s_c = -beta * dist_list[i]
        If normalize: scores = log_softmax(logits)
        If label_smoothing > 0:
          force P(self)=1-label_smoothing, and distribute label_smoothing over others
          proportionally to the base distribution over others.
        """
        assert len(dist_list) == len(cand)
        logits = torch.tensor(
            [-self.beta * float(d) for d in dist_list],
            device=device,
            dtype=torch.float32,
        )

        # no smoothing: keep existing behavior exactly
        if self.label_smoothing <= 0.0:
            return torch.log_softmax(logits, dim=0) if self.normalize else logits

        # smoothing requires probabilities; use log-probs internally
        logp = torch.log_softmax(logits, dim=0)

        cand_t = torch.tensor(cand, device=device, dtype=torch.long)
        self_pos = (cand_t == int(p)).nonzero(as_tuple=False)
        self_i = int(self_pos[0].item())

        # if there are no "other" candidates, put all mass on self
        if len(cand) == 1:
            forced = torch.empty_like(logp)
            forced[0] = 0.0  # log(1)
            return forced if self.normalize else forced  # log-scores are fine

        eps = self.label_smoothing
        p_self = 1.0 - eps

        mask = torch.ones(len(cand), dtype=torch.bool, device=device)
        mask[self_i] = False

        # renormalize others relative to their base distribution
        logp_other = torch.log_softmax(
            logp[mask], dim=0
        )  # stable; equivalent to log-softmax(logits_other)

        forced = torch.empty_like(logp)
        forced[self_i] = torch.log(
            torch.tensor(p_self, device=device, dtype=torch.float32)
        )
        forced[mask] = logp_other + torch.log(
            torch.tensor(eps, device=device, dtype=torch.float32)
        )

        # Return forced log-probs (recommended). If normalize=False, we still return log-probs
        # because the guarantee ">=0.8 on self" is defined in probability space.
        if not self.normalize:
            logging.warning(
                "label_smoothing>0 requested but normalize=False; returning log-prob scores anyway."
            )
        return forced

    def _build_weighted_transcript_fsa(
        self,
        y: List[int],
        V: int,
        device: torch.device,
        neighbor_ids: torch.Tensor,
        neighbor_dists: torch.Tensor,
    ) -> k2.Fsa:
        """
        Build an acceptor with states 0..L, arcs i->i+1 with label=c and score=log_prior(p->c).
        neighbor_ids, neighbor_dists: (V, k_others); self is not stored, always first in cand with dist 0.
        k_others can vary by language (e.g. 1 for identity/"global" fallback).
        """
        assert neighbor_ids.size(0) == V and neighbor_dists.shape == neighbor_ids.shape
        k_others = neighbor_ids.size(1)
        lines = []
        L = len(y)
        for i, p in enumerate(y):
            p = int(p)
            assert 0 <= p < V, f"phone index {p} out of range [0, {V})"
            cand = [p]
            dist_list = [0.0]
            for j in range(k_others):
                c = neighbor_ids[p, j].item()
                if c >= 0:
                    cand.append(c)
                    dist_list.append(neighbor_dists[p, j].item())
            scores = self._rescore_candidates(p=p, cand=cand, dist_list=dist_list, device=device)
            cand_score_pairs = sorted(zip(cand, scores.tolist()), key=lambda x: x[0])
            for c, s in cand_score_pairs:
                lines.append(f"{i} {i+1} {int(c)} {float(s)}")
        lines.append(f"{L} {L+1} -1 0.0")
        lines.append(f"{L+1}")
        return k2.Fsa.from_str("\n".join(lines)).to(device)

    def find_minimum_hlens(self, ys_pad, ylens):
        device = ys_pad.device
        ys_pad, ylens = ys_pad.cpu().tolist(), ylens.cpu().tolist()
        ys, min_hlens = [], []

        for y_pad, ylen in zip(ys_pad, ylens):
            y, min_hlen = [], 0
            prev = None

            for i in range(ylen):
                tok = y_pad[i]
                y.append(tok)
                min_hlen += 1
                if tok == prev:
                    min_hlen += 1
                prev = tok

            ys.append(y)
            min_hlens.append(min_hlen)

        min_hlens = torch.Tensor(min_hlens).long().to(device)
        return ys, min_hlens


# if __name__ == "__main__":
#     # python -m src.model.powsm.articulatory_ctc
#     V = 5
#     dist = torch.randn(V, V)
#     dist = (dist + dist.t()).abs()  # make it symmetric and non-negative

#     model = ArticulatoryCTC(dist=dist, beta=1.0, topk=3)

#     B, T = 2, 10
#     nnet_output = torch.randn(B, T, V).log_softmax(dim=-1)

#     ys_pad = torch.tensor([[1, 2, 3, 0, 0], [2, 2, 4, 3, 0]])
#     hlens = torch.tensor([10, 8])
#     ylens = torch.tensor([3, 4])

#     loss_utt = model(nnet_output, ys_pad, hlens, ylens)
#     print(loss_utt)


def create_network_output(sequence_labels, T, V, temperature=1.0):
    """
    Create network output that strongly predicts the given sequence.

    Args:
        sequence_labels: list of label indices (including blank=0)
        T: total number of timesteps
        V: vocabulary size
        temperature: softmax temperature (lower = more confident)
    """
    B = 1
    nnet_output = torch.zeros(B, T, V)

    # Distribute sequence across timesteps
    for t, label in enumerate(sequence_labels[:T]):
        nnet_output[0, t, label] = 10.0 / temperature

    # Fill remaining timesteps with blank
    for t in range(len(sequence_labels), T):
        nnet_output[0, t, 0] = 10.0 / temperature

    return torch.log_softmax(nnet_output, dim=-1)


def test_sequence(
    model, sequence_labels, sequence_name, target_ys_pad, hlens, ylens, V
):
    """Test a specific sequence against the target."""
    T = max(len(sequence_labels), 5)  # Ensure enough timesteps
    nnet_output = create_network_output(sequence_labels, T, V)

    # Adjust hlens to match T
    hlens_adjusted = torch.tensor([T])

    try:
        loss = model(nnet_output, target_ys_pad, hlens_adjusted, ylens)
        loss_val = loss.item()
        status = "✓ ACCEPTED" if loss_val < 5 else "? UNCLEAR"
        if loss_val == float("inf"):
            status = "✗ REJECTED"
    except Exception as e:
        loss_val = float("inf")
        status = "✗ REJECTED (error)"

    return sequence_name, loss_val, status


if __name__ == "__main__":
    # python -m src.model.powsm.articulatory_ctc
    print("=" * 70)
    print("TESTING ArticulatoryCTC: Which sequences match target 'CAT'?")
    print("=" * 70)

    # Setup
    # Vocabulary: {blank=0, C=1, A=2, T=3, E=4}
    V = 5
    label_names = ["blank", "C", "A", "T", "E"]

    # Create distance matrix where A and E are close
    dist = torch.zeros(V, V)
    for i in range(V):
        dist[i, i] = 0.0  # Self-distance is 0

    # A (2) and E (4) are close
    dist[2, 4] = 0.1
    dist[4, 2] = 0.1

    # All other non-self distances are large
    for i in range(V):
        for j in range(V):
            if i != j and dist[i, j] == 0:
                dist[i, j] = 10.0

    print("\nDistance Matrix:")
    print("        " + "  ".join(f"{name:5s}" for name in label_names))
    for i, label in enumerate(label_names):
        row_str = " ".join(f"{dist[i, j].item():5.1f}" for j in range(V))
        print(f"{label:5s}   {row_str}")

    from src.model.xeusphoneme.builders import matrix_to_neighbor_lists

    nids, ndists = matrix_to_neighbor_lists(dist, topk=2, blank_id=0)
    model = ArticulatoryCTC(
        neighbors_by_lang={"global": (nids, ndists)},
        beta=10.0,
        topk=2,
        normalize=True,
    )

    # Target: CAT = [C=1, A=2, T=3]
    target_ys_pad = torch.tensor([[1, 2, 3]])
    ylens = torch.tensor([3])
    hlens = torch.tensor([7])  # Will be adjusted per test

    print("\n" + "=" * 70)
    print("TARGET: CAT (indices [1, 2, 3])")
    print("=" * 70)

    # Test cases: (sequence_labels, name)
    # Note: 0 = blank
    test_cases = [
        # Exact matches
        ([1, 2, 3], "CAT (exact)"),
        ([0, 1, 0, 2, 0, 3, 0], "blank-C-blank-A-blank-T-blank"),
        ([1, 1, 2, 2, 3, 3], "CCAATT (with repeats)"),
        # E substituting for A
        ([1, 4, 3], "CET (E substitutes A)"),
        ([0, 1, 0, 4, 0, 3, 0], "blank-C-blank-E-blank-T-blank"),
        ([1, 4, 4, 3], "CEET (E repeated)"),
        # Both A and E - should be REJECTED
        ([1, 2, 4, 3], "CAET (both A and E)"),
        ([1, 0, 2, 0, 4, 0, 3], "C-blank-A-blank-E-blank-T"),
        ([1, 2, 2, 4, 3], "CAAET"),
        # Missing symbols
        ([1, 3], "CT (missing middle)"),
        ([1, 2], "CA (missing end)"),
        ([2, 3], "AT (missing beginning)"),
        # Wrong symbols at positions
        ([1, 3, 3], "CTT (wrong at pos 1)"),
        ([4, 2, 3], "EAT (E at pos 0, not allowed)"),
        ([1, 1, 3], "CCT (C repeated at pos 1)"),
        # Extra symbols
        ([1, 2, 3, 1], "CATC (extra at end)"),
        ([1, 2, 1, 3], "CACT (extra in middle)"),
    ]

    print("\nTest Results:")
    print("-" * 70)
    print(f"{'Sequence':<35} {'Loss':>12} {'Status':>15}")
    print("-" * 70)

    results = []
    for seq_labels, seq_name in test_cases:
        name, loss, status = test_sequence(
            model, seq_labels, seq_name, target_ys_pad, hlens, ylens, V
        )
        results.append((name, loss, status))

        # Format loss display
        if loss == float("inf"):
            loss_str = "inf"
        elif loss > 1000:
            loss_str = f"{loss:.2e}"
        else:
            loss_str = f"{loss:.4f}"

        print(f"{name:<35} {loss_str:>12} {status:>15}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    accepted = [r for r in results if "ACCEPTED" in r[2]]
    rejected = [r for r in results if "REJECTED" in r[2]]

    print(f"\n✓ ACCEPTED ({len(accepted)}):")
    for name, loss, status in accepted:
        print(f"  - {name}")

    print(f"\n✗ REJECTED ({len(rejected)}):")
    for name, loss, status in rejected:
        print(f"  - {name}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    print("1. CAT (exact match): Should be accepted")
    print("2. CET (E for A): Should be accepted (A and E are neighbors)")
    print("3. CAET (both A and E): Should be REJECTED (4 positions vs 3 in target)")
    print("4. CT (missing symbol): Depends on CTC topology and blanks")
    print("=" * 70)
