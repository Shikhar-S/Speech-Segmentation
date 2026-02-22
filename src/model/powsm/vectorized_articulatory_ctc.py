"""Vectorized Articulatory CTC: weighted FSA built from tensors (no per-position loops)."""

import logging
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import torch


def _float_to_int32_score(score_float: torch.Tensor) -> torch.Tensor:
    """Reinterpret float32 score bits as int32 for k2 arc tensor (same bit pattern)."""
    arr = score_float.detach().cpu().numpy()
    arr = np.ascontiguousarray(arr.astype(np.float32))
    return torch.from_numpy(arr.view(np.int32).copy()).to(
        device=score_float.device, dtype=torch.int32
    )


class VectorizedArticulatoryCTC(torch.nn.Module):
    """
    Same as ArticulatoryCTC but builds weighted transcript FSAs with batched tensor ops
    and k2.Fsa(arcs=tensor) instead of per-position loops and from_str.
    """

    def __init__(
        self,
        neighbors_by_lang: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        beta: float = 50.0,
        topk: int = 5,
        normalize: bool = True,
        modified_topo: bool = False,
        output_beam: float = 200,
        use_double_scores: bool = True,
        label_smoothing: float = 0.0,
        use_linear_chains: bool = False,
        num_linear_chain_paths: int = 10,
    ):
        super().__init__()
        assert (
            "global" in neighbors_by_lang
        ), "neighbors_by_lang must contain key 'global'"
        self._neighbors_by_lang = neighbors_by_lang
        nids, _ = next(iter(neighbors_by_lang.values()))
        self._V = nids.size(0)
        self.beta = float(beta)
        self.topk = int(topk)
        self.normalize = bool(normalize)
        self.modified_topo = bool(modified_topo)
        self.output_beam = float(output_beam)
        self.use_double_scores = bool(use_double_scores)
        self.label_smoothing = float(label_smoothing)
        assert 0.0 <= self.label_smoothing < 1.0, "label_smoothing must be in [0, 1)."
        self.use_linear_chains = bool(use_linear_chains)
        self.num_linear_chain_paths = int(num_linear_chain_paths)
        self._topo = None

    @staticmethod
    def find_minimum_hlens(ys_pad, ylens):
        """Same as ArticulatoryCTC.find_minimum_hlens (shared logic)."""
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

    def forward(
        self,
        nnet_output,
        ys_pad,
        hlens,
        ylens,
        lang_per_utt: Optional[List[str]] = None,
    ):
        device = nnet_output.device
        indices = torch.argsort(hlens, descending=True)
        ys, min_hlens = self.find_minimum_hlens(ys_pad[indices], ylens[indices])
        valid_sample_indices = (min_hlens <= hlens[indices]).nonzero(as_tuple=True)[0]
        if len(valid_sample_indices) < 1:
            logging.warning(
                "All examples are invalid for VectorizedArticulatoryCTC. Skip this batch"
            )
            return torch.tensor([0.0], device=device)
        indices = indices[valid_sample_indices]
        nnet_output, hlens, ylens = nnet_output[indices], hlens[indices], ylens[indices]
        ys = [ys[i.item()] for i in valid_sample_indices]
        if lang_per_utt is not None:
            lang_per_utt = [
                lang_per_utt[indices[i].item()] for i in range(len(indices))
            ]
        loss_utt = self.forward_core(
            nnet_output, ys, hlens, ylens, lang_per_utt=lang_per_utt
        )
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
        device = nnet_output.device
        B, T, V = nnet_output.size()
        supervision = torch.stack(
            [torch.arange(B), torch.zeros(B), hlens.cpu()], dim=1
        ).int()
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
        topo = self._get_topo(V=V, device=device)

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
            y = [int(t) for t in y if int(t) != 0]
            if len(y) == 0:
                transcript = k2.Fsa.from_str("0 1 -1 0.0\n1").to(device)
            elif self.use_linear_chains:
                transcript = self._build_linear_chains_fsa(
                    y,
                    V=V,
                    device=device,
                    neighbor_ids=nids,
                    neighbor_dists=ndists,
                    num_paths=self.num_linear_chain_paths,
                )
            else:
                transcript = self._build_weighted_transcript_fsa(
                    y, V=V, device=device, neighbor_ids=nids, neighbor_dists=ndists
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

    def _get_topo(self, V: int, device: torch.device) -> k2.Fsa:
        if self._topo is None or str(self._topo.device) != str(device):
            self._topo = k2.ctc_topo(
                max_token=V - 1, modified=self.modified_topo, device=device
            )
        return self._topo

    def _rescore_candidates_batched(
        self,
        logits: torch.Tensor,
        valid: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        logits: (L, K), valid: (L, K). Invalid positions are already -inf in logits.
        Apply log_softmax over last dim; if label_smoothing > 0, force self (col 0) to 1-eps.
        """
        L, K = logits.shape
        if self.label_smoothing <= 0.0:
            return torch.log_softmax(logits, dim=-1) if self.normalize else logits

        logp = torch.log_softmax(logits, dim=-1)
        eps = self.label_smoothing
        p_self = 1.0 - eps
        self_mask = torch.zeros_like(valid, dtype=torch.bool, device=device)
        self_mask[:, 0] = True
        other_mask = valid & (~self_mask)
        forced = torch.empty_like(logp, device=device, dtype=torch.float32)
        forced[self_mask] = torch.log(
            torch.tensor(p_self, device=device, dtype=torch.float32)
        )
        logp_other = logp.clone()
        logp_other[~other_mask] = -float("inf")
        logp_other = torch.log_softmax(logp_other, dim=-1)
        forced[other_mask] = logp_other[other_mask] + torch.log(
            torch.tensor(eps, device=device, dtype=torch.float32)
        )
        forced[~valid] = -float("inf")
        return forced

    def _build_weighted_transcript_fsa(
        self,
        y: List[int],
        V: int,
        device: torch.device,
        neighbor_ids: torch.Tensor,
        neighbor_dists: torch.Tensor,
    ) -> k2.Fsa:
        assert neighbor_ids.size(0) == V and neighbor_dists.shape == neighbor_ids.shape
        k_others = neighbor_ids.size(1)
        L = len(y)
        y_tensor = torch.tensor(y, device=device, dtype=torch.long)
        assert (y_tensor >= 0).all() and (
            y_tensor < V
        ).all(), "phone index out of range"

        cand_self = y_tensor.unsqueeze(1)
        cand_neighbors = neighbor_ids[y_tensor]
        cand_all = torch.cat([cand_self, cand_neighbors], dim=1)
        dist_self = torch.zeros(L, 1, device=device, dtype=torch.float32)
        dist_neighbors = neighbor_dists[y_tensor]
        dist_all = torch.cat([dist_self, dist_neighbors], dim=1)
        valid = cand_all >= 0

        logits = -self.beta * dist_all.float()
        logits = logits.masked_fill(~valid, -float("inf"))
        scores = self._rescore_candidates_batched(logits, valid, device)

        row_idx = (
            torch.arange(L, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(-1, cand_all.size(1))
        )
        col_idx = (
            torch.arange(cand_all.size(1), device=device).unsqueeze(0).expand(L, -1)
        )
        valid_flat = valid.view(-1)
        row_flat = row_idx.reshape(-1)[valid_flat]
        col_flat = col_idx.reshape(-1)[valid_flat]
        src = row_flat
        dest = row_flat + 1
        labels = cand_all[row_flat, col_flat]
        score_vals = scores[row_flat, col_flat]

        arcs_no_final = torch.stack(
            [
                src.to(torch.int32),
                dest.to(torch.int32),
                labels.to(torch.int32),
                _float_to_int32_score(score_vals),
            ],
            dim=1,
        )
        # sort_idx = torch.lexsort((labels.cpu(), src.cpu())).to(device)
        key = src.to(torch.int64) * (V + 1) + labels.to(
            torch.int64
        )  # labels are in [0, V-1]
        sort_idx = torch.argsort(key)
        arcs_no_final = arcs_no_final[sort_idx]
        zero_score_int = _float_to_int32_score(
            torch.tensor([0.0], device=device, dtype=torch.float32)
        )
        final_row = torch.tensor(
            [[L, L + 1, -1]],
            device=device,
            dtype=torch.int32,
        )
        # final_row = torch.cat([final_row, zero_score_int.unsqueeze(0).unsqueeze(1)], dim=1)
        final_row = torch.cat([final_row, zero_score_int.view(1, 1)], dim=1)
        full_arcs = torch.cat([arcs_no_final, final_row], dim=0)
        return k2.Fsa(full_arcs)

    def _build_linear_chains_fsa(
        self,
        y: List[int],
        V: int,
        device: torch.device,
        neighbor_ids: torch.Tensor,
        neighbor_dists: torch.Tensor,
        num_paths: int = 10,
    ) -> k2.Fsa:
        """
        Build a union-of-linear-chains FSA: one chain for the original transcript
        plus up to `num_paths` alternate chains where each unique phone is
        consistently replaced by a sampled neighbor.

        Sampling distribution for each unique phone is over its valid neighbors
        only (self is excluded), weighted by softmax(-beta * distance).

        All arcs have score 0.0 (purely structural FSA).

        State layout:
          - State 0: shared start state
          - Path i (0-indexed) uses internal states [i*L + 1, ..., (i+1)*L]
            where L = len(y)
          - State N*L + 1: super-final state (N = total number of paths)

        Arcs per path i with sequence path_i of length L:
          - (0, i*L+1, path_i[0], 0.0)          fan-out from start
          - (i*L+j, i*L+j+1, path_i[j], 0.0)    for j = 1..L-1
          - ((i+1)*L, N*L+1, -1, 0.0)            to super-final
        """
        assert neighbor_ids.size(0) == V and neighbor_dists.shape == neighbor_ids.shape
        L = len(y)
        y_tensor = torch.tensor(y, device=device, dtype=torch.long)
        assert (y_tensor >= 0).all() and (
            y_tensor < V
        ).all(), "phone index out of range"

        # --- Step 1: Build per-unique-phone sampling distributions (neighbors only) ---
        unique_phones = torch.unique(y_tensor)
        U = unique_phones.size(0)

        # Gather neighbor info for each unique phone
        u_neighbor_ids = neighbor_ids[unique_phones]  # (U, K)
        u_neighbor_dists = neighbor_dists[unique_phones]  # (U, K)
        u_valid = u_neighbor_ids >= 0  # (U, K)

        # Compute sampling weights: softmax(-beta * dist) over valid neighbors
        u_logits = -self.beta * u_neighbor_dists.float()
        u_logits = u_logits.masked_fill(~u_valid, -float("inf"))
        u_probs = torch.softmax(u_logits, dim=-1)  # (U, K)
        # Zero out invalid positions (softmax of -inf may produce nan)
        u_probs = u_probs.masked_fill(~u_valid, 0.0)

        # Detect phones with no valid neighbors (all probs zero).
        # For these, we set a dummy probability in col 0 so multinomial doesn't crash,
        # and after sampling we override their sampled ids to the original phone (self).
        has_no_neighbors = ~u_valid.any(dim=1)  # (U,) bool
        if has_no_neighbors.any():
            u_probs[has_no_neighbors, 0] = 1.0  # dummy so multinomial works

        # --- Step 2: Sample `num_paths` alternate paths ---
        # For each unique phone, sample a neighbor index for each of num_paths trials
        # u_probs: (U, K) -> sample (U, num_paths) indices into the K neighbors
        sampled_neighbor_col = torch.multinomial(
            u_probs.clamp(min=0.0), num_samples=num_paths, replacement=True
        )  # (U, num_paths)

        # Resolve to actual phone ids
        sampled_phone_ids = u_neighbor_ids.gather(
            1, sampled_neighbor_col
        )  # (U, num_paths)

        # For phones with no valid neighbors, override to the original phone (keep self)
        if has_no_neighbors.any():
            no_nb_mask = has_no_neighbors.unsqueeze(1).expand_as(
                sampled_phone_ids
            )  # (U, num_paths)
            self_ids = unique_phones.unsqueeze(1).expand_as(
                sampled_phone_ids
            )  # (U, num_paths)
            sampled_phone_ids[no_nb_mask] = self_ids[no_nb_mask]

        # Build a mapping from unique phone -> position in unique_phones
        phone_to_uidx = torch.empty(V, device=device, dtype=torch.long)
        phone_to_uidx[unique_phones] = torch.arange(U, device=device)

        # Map each position in y to its unique phone index
        y_uidx = phone_to_uidx[y_tensor]  # (L,)

        # Build alternate path sequences: (num_paths, L)
        # For each path p and position j, the label is sampled_phone_ids[y_uidx[j], p]
        alt_paths = sampled_phone_ids[y_uidx]  # (L, num_paths)
        alt_paths = alt_paths.t()  # (num_paths, L)

        # --- Step 3: Deduplicate and remove paths identical to original ---
        original = y_tensor.unsqueeze(0)  # (1, L)
        all_candidates = alt_paths  # (num_paths, L)

        # Convert to tuples for hashing/dedup
        seen = set()
        original_tuple = tuple(y)
        seen.add(original_tuple)

        unique_alt_list = []
        for i in range(all_candidates.size(0)):
            path_tuple = tuple(all_candidates[i].cpu().tolist())
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique_alt_list.append(all_candidates[i])

        # --- Step 4: Build union-of-chains FSA ---
        # All paths = [original] + unique alternates
        if len(unique_alt_list) > 0:
            alt_tensor = torch.stack(unique_alt_list, dim=0)  # (num_alt, L)
            all_paths = torch.cat([original, alt_tensor], dim=0)  # (N, L)
        else:
            all_paths = original  # (1, L)

        N = all_paths.size(0)  # total number of paths
        total_states = (
            N * L + 2
        )  # state 0 = start, states 1..N*L = chain internals, N*L+1 = super-final
        super_final = N * L + 1

        zero_score = torch.tensor(0.0, device=device, dtype=torch.float32)
        zero_score_int = _float_to_int32_score(zero_score.unsqueeze(0))  # (1,)

        # Build arc tensors for all paths at once
        # For path i, positions j=0..L-1:
        #   src = i*L + j      (but j=0 src is 0, the shared start)
        #   dest = i*L + j + 1
        #   label = all_paths[i, j]

        path_idx = torch.arange(N, device=device, dtype=torch.int32)  # (N,)
        pos_idx = torch.arange(L, device=device, dtype=torch.int32)  # (L,)

        # Expand to (N, L)
        pi = path_idx.unsqueeze(1).expand(N, L)  # (N, L)
        pj = pos_idx.unsqueeze(0).expand(N, L)  # (N, L)

        # Internal state numbers: i*L + j + 1 (1-indexed since state 0 is start)
        internal_states = pi * L + pj + 1  # (N, L) values in [1, N*L]

        # Source states: for j=0 it's state 0 (shared start), for j>0 it's internal_states[:, j-1]
        src_states = torch.zeros(N, L, device=device, dtype=torch.int32)
        src_states[:, 0] = 0
        src_states[:, 1:] = internal_states[:, :-1]

        # Destination states
        dest_states = internal_states  # (N, L)

        # Labels
        labels = all_paths.to(torch.int32)  # (N, L)

        # Scores: all zero
        scores_int = zero_score_int.expand(N * L)  # (N*L,)

        # Flatten chain arcs
        src_flat = src_states.reshape(-1)  # (N*L,)
        dest_flat = dest_states.reshape(-1)  # (N*L,)
        labels_flat = labels.reshape(-1)  # (N*L,)

        chain_arcs = torch.stack(
            [src_flat, dest_flat, labels_flat, scores_int],
            dim=1,
        )  # (N*L, 4)

        # Final arcs: from last state of each chain to super-final
        # Last state of path i = i*L + L = (i+1)*L
        last_states = (path_idx + 1).to(torch.int32) * L  # (N,)
        final_dest = torch.full((N,), super_final, device=device, dtype=torch.int32)
        final_labels = torch.full((N,), -1, device=device, dtype=torch.int32)
        final_scores = zero_score_int.expand(N)

        final_arcs = torch.stack(
            [last_states, final_dest, final_labels, final_scores],
            dim=1,
        )  # (N, 4)

        # Concatenate all arcs
        all_arcs = torch.cat([chain_arcs, final_arcs], dim=0)  # (N*L + N, 4)

        # Sort by (src, label) for k2
        src_col = all_arcs[:, 0].to(torch.int64)
        # Use label+2 so that -1 sorts correctly (becomes 1, while real labels start at 2)
        label_col = all_arcs[:, 2].to(torch.int64) + 2
        sort_key = src_col * (V + 3) + label_col
        sort_idx = torch.argsort(sort_key)
        all_arcs = all_arcs[sort_idx]

        return k2.Fsa(all_arcs)
