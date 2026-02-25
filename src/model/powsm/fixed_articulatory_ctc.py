"""
Articulatory CTC with STC/OTC-style penalty schedule.

Re-parameterized to use a single decaying penalty on neighbor arcs,
following the training tricks from:
  - Star Temporal Classification (Pratap et al., 2022)
  - Omni-temporal Classification (Gao et al., 2023)

Key idea: at each transcript position, the FSA allows the true label
OR articulatory neighbors. Neighbor arcs carry a penalty λ(t) that
starts large-negative (trust the transcript) and decays toward a
small-negative final value (allow articulatory flexibility).

Scoring at position i with true label y[i]:
  - self arc:     logit = 0
  - neighbor arc: logit = λ(t) - β * dist(y[i], neighbor)
  - arc scores  = log_softmax(logits)

Penalty schedule (following STC):
  λ(t) = λ_final + (λ_init - λ_final) * exp(-t * ln2 / halflife)

Model hyperparameters (4 total):
  beta             - articulatory distance scaling (phone-space property)
  penalty_init     - initial log-penalty on neighbor arcs
  penalty_final    - asymptotic log-penalty (keeps some transcript bias)
  penalty_halflife - steps to reach midpoint of decay

IMPORTANT — Expected-loss formulation:
  The correct objective is  L = -sum_c D(c|y) log P_θ(c|x)  (log INSIDE).
  This is computed by running a separate CTC forward-backward for each
  candidate sequence c, then weighting the resulting losses by D(c|y).
  Candidate sequences are the original transcript plus samples drawn
  from the per-position denoiser D(·|y_i).
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import torch


def _float_to_int32_score(score_float: torch.Tensor) -> torch.Tensor:
    """Reinterpret float32 score bits as int32 for k2 arc tensor."""
    arr = score_float.detach().cpu().numpy()
    arr = np.ascontiguousarray(arr.astype(np.float32))
    return torch.from_numpy(arr.view(np.int32).copy()).to(
        device=score_float.device, dtype=torch.int32
    )


class ScheduledArticulatoryCTC(torch.nn.Module):
    """
    CTC variant that allows articulatory neighbors at each transcript
    position, with an STC/OTC-style decaying penalty to stabilize
    early training.

    Computes the expected CTC loss:
        L = - sum_c  D(c | y) * log P_θ(c | x)
    by enumerating a small set of candidate sequences sampled from D,
    computing a standard CTC loss for each, and combining with D-weights.
    """

    def __init__(
        self,
        neighbors_by_lang: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        # --- Model hyperparameters (4 knobs) ---
        beta: float = 15.0,
        penalty_init: float = -12.0,
        penalty_final: float = 0,
        penalty_halflife: int = 2500,
        # --- Candidate sampling ---
        num_candidate_seqs: int = 4,
        # --- k2 implementation details (not tuning knobs) ---
        modified_topo: bool = False,
        output_beam: float = 200.0,
        use_double_scores: bool = True,
    ):
        """
        Args:
            neighbors_by_lang: Dict mapping language code -> (neighbor_ids, neighbor_dists)
                tensors of shape (V, K). Must contain key 'global'.
            beta: Scaling factor for articulatory distances. Higher = sharper
                preference for the true label over distant neighbors.
            penalty_init: Initial log-penalty on all neighbor (non-self) arcs.
                Large negative → neighbors suppressed → standard CTC.
            penalty_final: Asymptotic log-penalty.
            penalty_halflife: Steps to reach midpoint between init and final.
            num_candidate_seqs: Total number of candidate sequences per
                utterance (including the original transcript). Additional
                sequences are sampled from the per-position denoiser.
            modified_topo: Whether to use k2's modified CTC topology.
            output_beam: Beam size for k2.ctc_loss pruning.
            use_double_scores: Use float64 for k2 forward-backward.
        """
        super().__init__()
        assert (
            "global" in neighbors_by_lang
        ), "neighbors_by_lang must contain key 'global'"

        self._neighbors_by_lang = neighbors_by_lang
        nids, _ = next(iter(neighbors_by_lang.values()))
        self._V = nids.size(0)

        # Model hyperparameters
        self.beta = float(beta)
        self.penalty_init = float(penalty_init)
        self.penalty_final = float(penalty_final)
        self.penalty_halflife = int(penalty_halflife)
        assert (
            self.penalty_init <= self.penalty_final <= 0.0
        ), "Need penalty_init <= penalty_final <= 0"
        assert self.penalty_halflife > 0

        # Candidate sampling
        self.num_candidate_seqs = int(num_candidate_seqs)
        assert self.num_candidate_seqs >= 1

        # k2 details
        self.modified_topo = bool(modified_topo)
        self.output_beam = float(output_beam)
        self.use_double_scores = bool(use_double_scores)

        # Internal state
        self._topo = None
        self._global_step = 0

    # ------------------------------------------------------------------
    # Penalty schedule
    # ------------------------------------------------------------------

    def current_penalty(self) -> float:
        """
        Compute λ(t) following the STC exponential decay schedule:
            λ(t) = λ_final + (λ_init - λ_final) * exp(-t * ln2 / halflife)

        At step 0:         λ ≈ λ_init  (neighbors suppressed)
        At step halflife:  λ = midpoint of (λ_init, λ_final)
        As step → ∞:       λ → λ_final (neighbors at full articulatory weight)
        """
        tau = self.penalty_halflife / math.log(2)
        return self.penalty_final + (self.penalty_init - self.penalty_final) * math.exp(
            -self._global_step / tau
        )

    # ------------------------------------------------------------------
    # Minimum hlens (unchanged logic)
    # ------------------------------------------------------------------

    @staticmethod
    def find_minimum_hlens(ys_pad, ylens):
        """Compute minimum encoder output lengths for valid CTC alignment."""
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
        return ys, torch.tensor(min_hlens, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        nnet_output: torch.Tensor,
        ys_pad: torch.Tensor,
        hlens: torch.Tensor,
        ylens: torch.Tensor,
        lang_per_utt: Optional[List[str]] = None,
    ) -> torch.Tensor:
        device = nnet_output.device

        # Sort by descending length (k2 requirement)
        indices = torch.argsort(hlens, descending=True)
        ys, min_hlens = self.find_minimum_hlens(ys_pad[indices], ylens[indices])

        # Filter invalid samples
        valid_sample_indices = (min_hlens <= hlens[indices]).nonzero(as_tuple=True)[0]
        if len(valid_sample_indices) < 1:
            logging.warning("All examples invalid for ArticulatoryCTC. Skipping batch.")
            return torch.tensor([0.0], device=device)

        indices = indices[valid_sample_indices]
        nnet_output = nnet_output[indices]
        hlens = hlens[indices]
        ylens = ylens[indices]
        ys = [ys[i.item()] for i in valid_sample_indices]
        if lang_per_utt is not None:
            lang_per_utt = [
                lang_per_utt[indices[i].item()] for i in range(len(indices))
            ]

        loss_utt = self._forward_core(
            nnet_output, ys, hlens, ylens, lang_per_utt=lang_per_utt
        )

        # Unsort
        indices2 = torch.argsort(indices)
        loss_utt = loss_utt[indices2]

        # Advance step counter (once per batch, not per sample)
        if self.training:
            self._global_step += 1

        return loss_utt

    def _forward_core(
        self,
        nnet_output: torch.Tensor,
        ys: List[List[int]],
        hlens: torch.Tensor,
        ylens: torch.Tensor,
        lang_per_utt: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Expected-loss forward:
          1. For each utterance, generate candidate sequences from D(·|y).
          2. Build an unweighted transcript FSA for each candidate.
          3. Run CTC loss for all candidates in one batched k2 call
             (expanding acoustic features to match).
          4. Combine:  loss_b = sum_j  D(c_j | y_b) * ctc_loss(c_j, x_b)
             with D-weights renormalized over the candidate set.
        """
        device = nnet_output.device
        B, T, V = nnet_output.size()
        topo = self._get_topo(V=V, device=device)
        lam = self.current_penalty()

        # ---- Collect candidates across all utterances ----
        cand_graphs = []
        cand_batch_idx = []  # maps each candidate → utterance index in [0, B)
        cand_log_D = []  # log D(c | y) for each candidate

        for b, y in enumerate(ys):
            lang = (
                lang_per_utt[b] if (lang_per_utt and b < len(lang_per_utt)) else None
            ) or "global"
            nids, ndists = self._neighbors_by_lang.get(
                lang, self._neighbors_by_lang["global"]
            )
            nids = nids.to(device)
            ndists = ndists.to(device)

            y_clean = [int(t) for t in y if int(t) != 0]
            if len(y_clean) == 0:
                fsa = k2.Fsa.from_str("0 1 -1 0.0\n1").to(device)
                fsa = k2.arc_sort(k2.add_epsilon_self_loops(fsa))
                g = k2.compose(topo, fsa, treat_epsilons_specially=False)
                cand_graphs.append(k2.arc_sort(g))
                cand_batch_idx.append(b)
                cand_log_D.append(0.0)
                continue

            seqs, log_ws = self._generate_candidates(
                y_clean, V, device, nids, ndists, lam
            )

            for seq, lw in zip(seqs, log_ws):
                fsa = self._build_unweighted_fsa(seq, V, device)
                fsa = k2.arc_sort(k2.add_epsilon_self_loops(fsa))
                g = k2.compose(topo, fsa, treat_epsilons_specially=False)
                cand_graphs.append(k2.arc_sort(g))
                cand_batch_idx.append(b)
                cand_log_D.append(lw)

        # ---- Expand acoustic features to match candidates ----
        # Utterances are already sorted by descending hlens.
        # Candidates are emitted in utterance order, so expanded hlens
        # are still non-increasing (as k2 requires).
        idx = torch.tensor(cand_batch_idx, device=device, dtype=torch.long)
        expanded_nnet = nnet_output[idx]  # (C, T, V)
        expanded_hlens = hlens[idx]  # (C,)

        C = len(cand_graphs)
        supervision = torch.stack(
            [torch.arange(C), torch.zeros(C), expanded_hlens.cpu()], dim=1
        ).int()
        dense_fsa_vec = k2.DenseFsaVec(expanded_nnet, supervision)
        decoding_graphs = k2.create_fsa_vec(cand_graphs)

        # ---- Per-candidate CTC loss: -log P_θ(c | x)  [log INSIDE] ----
        cand_losses = k2.ctc_loss(
            decoding_graph=decoding_graphs,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=self.output_beam,
            reduction="none",
            use_double_scores=self.use_double_scores,
        )

        # ---- Combine: loss_b = sum_j  D_norm(c_j | y_b) * loss_j ----
        log_D = torch.tensor(cand_log_D, device=device, dtype=torch.float32)
        D_weights = log_D.exp()

        # Renormalize D-weights per utterance so they sum to 1
        weight_sums = torch.zeros(B, device=device, dtype=torch.float32)
        weight_sums.scatter_add_(0, idx, D_weights)
        D_weights = D_weights / weight_sums[idx]

        weighted = D_weights * cand_losses

        loss_utt = torch.zeros(B, device=device, dtype=cand_losses.dtype)
        loss_utt.scatter_add_(0, idx, weighted)

        return loss_utt

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def _get_topo(self, V: int, device: torch.device) -> k2.Fsa:
        if self._topo is None or str(self._topo.device) != str(device):
            self._topo = k2.ctc_topo(
                max_token=V - 1, modified=self.modified_topo, device=device
            )
        return self._topo

    # ------------------------------------------------------------------
    # Candidate generation from per-position denoiser D(c_i | y_i)
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        y: List[int],
        V: int,
        device: torch.device,
        neighbor_ids: torch.Tensor,
        neighbor_dists: torch.Tensor,
        penalty: float,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Build per-position denoiser D(c_i | y_i) and sample candidate
        sequences from the factored distribution D(c|y) = prod_i D(c_i|y_i).

        Returns:
            seqs: list of candidate sequences (original always first)
            log_ws: corresponding log D(c | y) values
        """
        L = len(y)
        y_tensor = torch.tensor(y, device=device, dtype=torch.long)

        # Per-position candidates: self + precomputed neighbors
        cand_self = y_tensor.unsqueeze(1)  # (L, 1)
        cand_neighbors = neighbor_ids[y_tensor]  # (L, K)
        cand_all = torch.cat([cand_self, cand_neighbors], dim=1)  # (L, 1+K)

        valid_neighbors = cand_neighbors >= 0
        valid_self = torch.ones(L, 1, device=device, dtype=torch.bool)
        valid = torch.cat([valid_self, valid_neighbors], dim=1)

        # Denoiser logits:  self → 0,  neighbor → penalty - β*dist
        dist_self = torch.zeros(L, 1, device=device, dtype=torch.float32)
        dist_neighbors = neighbor_dists[y_tensor]
        dist_all = torch.cat([dist_self, dist_neighbors], dim=1)

        logits = -self.beta * dist_all
        logits[:, 1:] += penalty
        logits = logits.masked_fill(~valid, -float("inf"))

        log_D_pos = torch.log_softmax(logits, dim=-1)  # (L, 1+K)

        # Always include the original transcript
        orig_log_w = log_D_pos[:, 0].sum().item()
        seen = {tuple(y): orig_log_w}

        # Sample additional sequences (deduplicate)
        if self.num_candidate_seqs > 1:
            probs = log_D_pos.exp().clamp(min=0.0)
            arange_L = torch.arange(L, device=device)
            for _ in range(self.num_candidate_seqs - 1):
                cols = torch.multinomial(probs, num_samples=1).squeeze(1)  # (L,)
                seq_ids = cand_all[arange_L, cols]
                log_w = log_D_pos[arange_L, cols].sum().item()
                key = tuple(seq_ids.cpu().tolist())
                if key not in seen:
                    seen[key] = log_w

        seqs = [list(k) for k in seen.keys()]
        log_ws = list(seen.values())
        return seqs, log_ws

    # ------------------------------------------------------------------
    # Unweighted transcript FSA (scores = 0 on all arcs)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_unweighted_fsa(
        y: List[int],
        V: int,
        device: torch.device,
    ) -> k2.Fsa:
        """
        Build a standard linear transcript FSA with zero arc scores.
        State i --label y[i]--> state i+1, for i in 0..L-1.
        State L --label -1--> state L+1 (final).
        """
        L = len(y)
        src = torch.arange(L, device=device, dtype=torch.int32)
        dest = src + 1
        labels = torch.tensor(y, device=device, dtype=torch.int32)
        zero_scores = _float_to_int32_score(
            torch.zeros(L, device=device, dtype=torch.float32)
        )

        arcs = torch.stack([src, dest, labels, zero_scores], dim=1)

        # Final arc: L → L+1, label = -1, score = 0
        final_score = _float_to_int32_score(
            torch.tensor([0.0], device=device, dtype=torch.float32)
        )
        final_arc = torch.tensor([[L, L + 1, -1]], device=device, dtype=torch.int32)
        final_arc = torch.cat([final_arc, final_score.view(1, 1)], dim=1)

        return k2.Fsa(torch.cat([arcs, final_arc], dim=0))
