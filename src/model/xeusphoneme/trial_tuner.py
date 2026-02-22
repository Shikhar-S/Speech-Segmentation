# python -m src.model.xeusphoneme.builders_iou
import json
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.model.xeusphoneme.builders import (
    build_panphon_distance_matrix,
    build_base_distance_matrix,
)
from src.model.xeusphoneme.resources.phonetic_substitutions import (
    ENGLISH_PHONEME_SUBSTITUTIONS,
    get_substitutions,
)


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


def mean_metrics_all_langs_union(
    *,
    vocab: Dict[str, int],
    dist_matrix: np.ndarray,
    topk: int,
    beta: float,
    thresh: float,
    blank_id: int = 0,
    symbols: Optional[List[str]] = None,
) -> tuple[float, float]:
    """Returns (mean_hard_iou, mean_soft_iou) with GT(sym)=union over langs."""
    langs = list(ENGLISH_PHONEME_SUBSTITUTIONS.keys())
    syms = symbols if symbols is not None else list(vocab.keys())
    hard_vals, soft_vals = [], []

    for sym in syms:
        if sym not in vocab:
            continue

        gt_syms = set().union(*(get_substitutions(lang, sym) for lang in langs))
        gt = [vocab[s] for s in gt_syms if s in vocab]
        if not gt:
            continue

        nb = distance_neighbors(
            dist_matrix,
            vocab[sym],
            topk=topk,
            beta=beta,
            thresh=thresh,
            blank_id=blank_id,
        )
        A, B = set(nb), set(gt)
        hard_vals.append((len(A & B) / max(1, len(A | B))) if (A or B) else 1.0)

        cand, p = distance_neighbors_with_probs(
            dist_matrix,
            vocab[sym],
            topk=topk,
            beta=beta,
            thresh=thresh,
            blank_id=blank_id,
        )
        p_map = dict(zip(cand, p))
        q = 1.0 / len(gt)
        U = set(p_map.keys()) | set(gt)
        inter = sum(min(p_map.get(i, 0.0), q if i in B else 0.0) for i in U)
        uni = sum(max(p_map.get(i, 0.0), q if i in B else 0.0) for i in U)
        soft_vals.append(inter / max(1e-12, uni))

    hard = float(np.mean(hard_vals)) if hard_vals else 0.0
    soft = float(np.mean(soft_vals)) if soft_vals else 0.0
    return hard, soft


if __name__ == "__main__":
    vocab_path = "src/model/xeusphoneme/resources/ipa_vocab.json"
    V = json.load(open(vocab_path))
    revV = {v: k for k, v in V.items()}
    base_idx = build_base_index(V)

    dist = build_panphon_distance_matrix(vocab=list(V.keys()))
    # dist = build_base_distance_matrix(vocab=list(V.keys()))

    TOPKS = [50]
    BETAS = [50.0]
    THRESHS = [0.5]
    BLANK_ID = 0

    grid = [(k, b, t) for k in TOPKS for b in BETAS for t in THRESHS]

    best_h, best_h_cfg = -1.0, None
    best_s, best_s_cfg = -1.0, None
    rows = []

    for topk, beta, thresh in tqdm(grid, desc="tuning (topk,beta,thresh)"):
        hard_iou, soft_iou = mean_metrics_all_langs_union(
            vocab=V,
            dist_matrix=dist,
            topk=topk,
            beta=beta,
            thresh=thresh,
            blank_id=BLANK_ID,
        )
        rows.append((soft_iou, hard_iou, topk, beta, thresh))
        if hard_iou > best_h:
            best_h, best_h_cfg = hard_iou, (topk, beta, thresh)
        if soft_iou > best_s:
            best_s, best_s_cfg = soft_iou, (topk, beta, thresh)

    rows.sort(reverse=True)
    print("Best HARD:", f"{best_h:.4f}", "cfg=", best_h_cfg)
    print("Best SOFT:", f"{best_s:.4f}", "cfg=", best_s_cfg)
    print("Top 10 by SOFT IoU (soft, hard, topk, beta, thresh):")
    for soft_iou, hard_iou, topk, beta, thresh in rows[:10]:
        print(
            f"  {soft_iou:.4f}  {hard_iou:.4f}  topk={topk:<2d}  beta={beta:<6g}  thresh={thresh:<6g}"
        )

    # ---- inspect specific symbols: neighbors + probs + GT union + base variants
    langs = list(ENGLISH_PHONEME_SUBSTITUTIONS.keys())
    # for sym in ["bʰ", "dʰ", "tʰ"]:
    for sym in V.keys():
        if sym not in V:
            print(f"{sym}: not in vocab")
            continue

        nb = distance_neighbors(
            dist,
            V[sym],
            topk=TOPKS[0],
            beta=BETAS[0],
            thresh=THRESHS[0],
            blank_id=BLANK_ID,
        )
        cand, p = distance_neighbors_with_probs(
            dist,
            V[sym],
            topk=TOPKS[0],
            beta=BETAS[0],
            thresh=THRESHS[0],
            blank_id=BLANK_ID,
        )
        soft_pairs = sorted(zip(cand, p), key=lambda x: x[1], reverse=True)

        gt_syms = set().union(*(get_substitutions(lang, sym) for lang in langs))
        gt = sorted([V[s] for s in gt_syms if s in V])

        print("\n" + "==" * 30)
        print(f"SYM: {sym}")
        print("Base variants:", sorted(same_base_variants(sym, base_idx)))
        print("HARD neighbors:", [revV[i] for i in nb])
        print(
            "SOFT neighbors (sym, p):", [(revV[i], float(pi)) for i, pi in soft_pairs]
        )
        print("GT union:", [revV[i] for i in gt])
