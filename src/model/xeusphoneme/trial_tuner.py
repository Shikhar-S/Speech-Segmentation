from typing import Dict, List

import numpy as np
import torch


def build_base_index(vocab: Dict[str, int]) -> Dict[str, List[str]]:
    """Map base -> symbols. Base is first codepoint, except affricates like 'd͡ʒ' where base is first 3 codepoints."""

    def base_key(s: str) -> str:
        if not s:
            return ""
        return s[:3] if len(s) >= 3 and s[1] == "͡" else s[0]

    idx: Dict[str, List[str]] = {}
    for s in vocab.keys():
        k = base_key(s)
        if k:
            idx.setdefault(k, []).append(s)
    return idx


def same_base_variants(sym: str, base_idx: Dict[str, List[str]]) -> List[str]:
    """
    If sym is an affricate/connected (e.g. 'd͡ʒ', 't͡ɕʰ'), treat the whole connected unit
    (up through the second segment) as the base for diacritic variants.
    Otherwise use the first codepoint as base.
    """
    if not sym:
        return []
    if len(sym) >= 3 and sym[1] == "͡":
        base = sym[:3]  # e.g. "t͡ɕ"
        return [s for s in base_idx.get(base, []) if s.startswith(base)]
    return base_idx.get(sym[0], [])


def distance_neighbors(
    dist_matrix: np.ndarray,
    token_id: int,
    *,
    topk: int,
    beta: float,
    thresh: float,
    blank_id: int = 0,
) -> List[int]:
    drow = torch.as_tensor(dist_matrix[token_id], dtype=torch.float32).clone()
    drow[blank_id] = float("inf")
    k = min(topk, drow.numel() - 1)
    cand = torch.topk(drow, k=k, largest=False).indices
    if not (cand == token_id).any().item():
        cand = torch.cat([cand[:-1], torch.tensor([token_id], dtype=cand.dtype)])
    keep = drow[cand] < thresh
    _ = torch.log_softmax(-beta * drow[cand], dim=0)[keep]  # parity / future use
    return cand[keep].tolist()


def distance_neighbors_with_probs(
    dist_matrix: np.ndarray,
    token_id: int,
    *,
    topk: int,
    beta: float,
    thresh: float,
    blank_id: int = 0,
) -> tuple[list[int], list[float]]:
    drow = torch.as_tensor(dist_matrix[token_id], dtype=torch.float32).clone()
    drow[blank_id] = float("inf")
    k = min(topk, drow.numel() - 1)
    cand = torch.topk(drow, k=k, largest=False).indices
    if not (cand == token_id).any().item():
        cand = torch.cat([cand[:-1], torch.tensor([token_id], dtype=cand.dtype)])
    logp = torch.log_softmax(-beta * drow[cand], dim=0)
    keep = drow[cand] < thresh
    cand = cand[keep].tolist()
    p = torch.exp(logp[keep]).tolist()
    return cand, p


