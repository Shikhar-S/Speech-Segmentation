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
    ):
        super().__init__()
        assert "global" in neighbors_by_lang, "neighbors_by_lang must contain key 'global'"
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

    def forward(self, nnet_output, ys_pad, hlens, ylens, lang_per_utt: Optional[List[str]] = None):
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
            lang_per_utt = [lang_per_utt[indices[i].item()] for i in range(len(indices))]
        loss_utt = self.forward_core(nnet_output, ys, hlens, ylens, lang_per_utt=lang_per_utt)
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
            lang = (lang_per_utt[b] if (lang_per_utt and b < len(lang_per_utt)) else None) or "global"
            nids, ndists = self._neighbors_by_lang.get(lang, self._neighbors_by_lang["global"])
            nids = nids.to(device)
            ndists = ndists.to(device)
            y = [int(t) for t in y if int(t) != 0]
            if len(y) == 0:
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
        assert (y_tensor >= 0).all() and (y_tensor < V).all(), "phone index out of range"

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

        row_idx = torch.arange(L, device=device, dtype=torch.int32).unsqueeze(1).expand(-1, cand_all.size(1))
        col_idx = torch.arange(cand_all.size(1), device=device).unsqueeze(0).expand(L, -1)
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
        sort_idx = torch.lexsort((labels.cpu(), src.cpu())).to(device)
        arcs_no_final = arcs_no_final[sort_idx]
        zero_score_int = _float_to_int32_score(
            torch.tensor([0.0], device=device, dtype=torch.float32)
        )
        final_row = torch.tensor(
            [[L, L + 1, -1]],
            device=device,
            dtype=torch.int32,
        )
        final_row = torch.cat([final_row, zero_score_int.unsqueeze(0).unsqueeze(1)], dim=1)
        full_arcs = torch.cat([arcs_no_final, final_row], dim=0)
        return k2.Fsa(full_arcs)
