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
    """

    def __init__(
        self,
        neighbors_by_lang: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        # --- Model hyperparameters (4 knobs) ---
        beta: float = 50.0,
        penalty_init: float = -8.0,
        penalty_final: float = -0.5,
        penalty_halflife: int = 8000,
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
                This is a property of your phone/articulatory space.
            penalty_init: Initial log-penalty on all neighbor (non-self) arcs.
                Large negative (e.g. -8) means neighbors are essentially
                impossible early in training → behaves like standard CTC.
            penalty_final: Asymptotic log-penalty. Slightly negative (e.g. -0.5)
                maintains some bias toward the given transcript even after
                convergence. Set to 0.0 for full articulatory flexibility.
            penalty_halflife: Number of training steps for the penalty to
                reach the midpoint between penalty_init and penalty_final.
                Typical values: 5000-15000 (cf. STC's t½ = 8000-10000).
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
        device = nnet_output.device
        B, T, V = nnet_output.size()

        supervision = torch.stack(
            [torch.arange(B), torch.zeros(B), hlens.cpu()], dim=1
        ).int()
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
        topo = self._get_topo(V=V, device=device)

        lam = self.current_penalty()

        graphs = []
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
                transcript = k2.Fsa.from_str("0 1 -1 0.0\n1").to(device)
            else:
                transcript = self._build_transcript_fsa(
                    y_clean,
                    V=V,
                    device=device,
                    neighbor_ids=nids,
                    neighbor_dists=ndists,
                    penalty=lam,
                )

            transcript = k2.arc_sort(k2.add_epsilon_self_loops(transcript))
            g = k2.compose(topo, transcript, treat_epsilons_specially=False)
            g = k2.arc_sort(g)
            graphs.append(g)

        decoding_graphs = k2.create_fsa_vec(graphs)
        loss_utt = k2.ctc_loss(
            decoding_graph=decoding_graphs,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=self.output_beam,
            reduction="none",
            use_double_scores=self.use_double_scores,
        )
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
    # Weighted transcript FSA (single mode, STC-style scoring)
    # ------------------------------------------------------------------

    def _build_transcript_fsa(
        self,
        y: List[int],
        V: int,
        device: torch.device,
        neighbor_ids: torch.Tensor,
        neighbor_dists: torch.Tensor,
        penalty: float,
    ) -> k2.Fsa:
        """
        Build a weighted transcript FSA for sequence y.

        At each position i (state i → state i+1), there are arcs for:
          - The true label y[i]           with logit = 0
          - Each valid neighbor of y[i]   with logit = penalty - beta * dist

        Arc scores = log_softmax(logits) so they form a proper log-
        probability distribution at each position.

        The penalty parameter controls how much probability mass can flow
        through neighbor arcs. Early in training (penalty very negative),
        almost all mass is on the true label → standard CTC behavior.
        """
        L = len(y)
        y_tensor = torch.tensor(y, device=device, dtype=torch.long)
        assert (y_tensor >= 0).all() and (
            y_tensor < V
        ).all(), "phone index out of range"

        # Gather candidates: self + all neighbors from precomputed table
        cand_self = y_tensor.unsqueeze(1)  # (L, 1)
        cand_neighbors = neighbor_ids[y_tensor]  # (L, K)
        cand_all = torch.cat([cand_self, cand_neighbors], dim=1)  # (L, 1+K)

        valid_neighbors = cand_neighbors >= 0  # (L, K)
        valid_self = torch.ones(L, 1, device=device, dtype=torch.bool)
        valid = torch.cat([valid_self, valid_neighbors], dim=1)  # (L, 1+K)

        # Compute logits
        #   self:     0
        #   neighbor: penalty - beta * dist
        dist_self = torch.zeros(L, 1, device=device, dtype=torch.float32)
        dist_neighbors = neighbor_dists[y_tensor]  # (L, K)
        dist_all = torch.cat([dist_self, dist_neighbors], dim=1)

        logits = -self.beta * dist_all
        logits[:, 1:] += penalty  # penalty only on neighbor arcs
        logits = logits.masked_fill(~valid, -float("inf"))

        # log_softmax → proper log-probability arc scores
        scores = torch.log_softmax(logits, dim=-1)

        # Build arc tensor: (src, dest, label, score_as_int32)
        K_total = cand_all.size(1)
        row_idx = (
            torch.arange(L, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(L, K_total)
        )
        valid_flat = valid.reshape(-1)
        row_flat = row_idx.reshape(-1)[valid_flat]

        col_idx = torch.arange(K_total, device=device).unsqueeze(0).expand(L, K_total)
        col_flat = col_idx.reshape(-1)[valid_flat]

        src = row_flat
        dest = row_flat + 1
        labels = cand_all[row_flat, col_flat]
        score_vals = scores[row_flat, col_flat]

        arcs = torch.stack(
            [
                src.to(torch.int32),
                dest.to(torch.int32),
                labels.to(torch.int32),
                _float_to_int32_score(score_vals),
            ],
            dim=1,
        )

        # Sort by (src_state, label) for k2
        sort_key = src.to(torch.int64) * (V + 1) + labels.to(torch.int64)
        arcs = arcs[torch.argsort(sort_key)]

        # Append final arc: state L → state L+1 with label -1
        final_score_int = _float_to_int32_score(
            torch.tensor([0.0], device=device, dtype=torch.float32)
        )
        final_arc = torch.tensor([[L, L + 1, -1]], device=device, dtype=torch.int32)
        final_arc = torch.cat([final_arc, final_score_int.view(1, 1)], dim=1)

        return k2.Fsa(torch.cat([arcs, final_arc], dim=0))
