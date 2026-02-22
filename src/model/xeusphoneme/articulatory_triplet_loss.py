"""Articulatory Triplet Auxiliary Loss for CTC-based Phone Recognition.

Cross-modal triplet loss that grounds CTC phone predictions in articulatory-acoustic
reality, providing a debiasing signal against noisy G2P labels.

The loss operates between:
  1. Acoustic embeddings: encoder features projected into articulatory space
  2. Articulatory embeddings: FIXED PanPhon feature vectors for predicted phones

Key design decisions:
  - Articulatory embeddings are completely fixed (no learned parameters) to prevent
    G2P bias from collapsing articulatory distinctions via a learned projection.
  - Stop-gradient on encoder: triplet loss only corrects the CTC head, not the encoder.
  - Blank frames are excluded from the loss.
  - Anchor and negative subsampling to bound compute regardless of batch/seq length.

Compute budget (defaults):
  With B=16, T=1000 → ~8000 valid frames after blank filtering.
  We subsample 256 anchors and 2048 negative candidates.
  Distance matrix: (256, 2048) × F=24 → ~2MB. Negligible.
"""

from typing import Dict, Set, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

SPECIAL_TOKENS: Set[str] = {"<blank>", "<sos>", "<eos>", "<unk>", "<pad>"}


def build_panphon_feature_matrix(
    token_list: list[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract fixed PanPhon articulatory feature vectors for each phone.

    Args:
        token_list: Vocabulary list where token_list[i] is phone string for id i.

    Returns:
        features: (V, F) float tensor of L2-normalized, weighted articulatory features.
                  Special tokens get zero vectors.
        valid_mask: (V,) bool tensor, True for phones with valid articulatory features.
    """
    import panphon

    ft = panphon.FeatureTable()

    V = len(token_list)
    sample_fts = ft.word_fts("a")
    F_dim = len(sample_fts[0].numeric()) if sample_fts else 24

    features = torch.zeros(V, F_dim, dtype=torch.float32)
    valid_mask = torch.zeros(V, dtype=torch.bool)

    try:
        from panphon.distance import Distance

        dst = Distance()
        if hasattr(dst, "fm") and hasattr(dst.fm, "weights"):
            weights = torch.tensor(
                [dst.fm.weights.get(f, 1.0) for f in ft.names], dtype=torch.float32
            )
        else:
            weights = torch.ones(F_dim, dtype=torch.float32)
    except Exception:
        weights = torch.ones(F_dim, dtype=torch.float32)

    matched = 0
    failed = []
    for i, phone in enumerate(token_list):
        if phone in SPECIAL_TOKENS:
            continue
        try:
            segs = ft.word_fts(phone)
            if not segs:
                failed.append(phone)
                continue
            fv = torch.tensor(segs[0].numeric(), dtype=torch.float32)
            fv = fv * weights
            norm = fv.norm()
            if norm > 0:
                fv = fv / norm
            features[i] = fv
            valid_mask[i] = True
            matched += 1
        except Exception:
            failed.append(phone)

    log.info(
        f"PanPhon features: {matched}/{V} phones matched, "
        f"{len(failed)} failed: {failed[:20]}{'...' if len(failed) > 20 else ''}"
    )
    return features, valid_mask


class ArticulatoryTripletLoss(nn.Module):
    """Cross-modal triplet loss with subsampling for large batches.

    With B=16 and T=1000, there are ~8000 valid frames per forward pass.
    Computing all-pairs distances would be a (8000, 8000) matrix (~256MB).
    Instead we subsample:
      1. `max_anchors` frames as anchors (keep grad — these drive learning)
      2. `max_negatives` frames as negative candidates (detached for mining only)
      3. Distance matrix is (max_anchors, max_negatives) — ~2MB with defaults

    The negative pool is stratified across utterances so that every utterance
    contributes roughly equally, ensuring cross-utterance diversity.

    Args:
        token_list: Vocabulary list (phone strings).
        encoder_dim: Dimension of encoder output (D).
        mlp_hidden: Hidden dimension for acoustic projection MLP.
        margin: Triplet loss margin.
        lambda_crossmodal: Weight for cross-modal triplet loss.
        mining: Negative mining strategy: "semi_hard", "hard", or "random".
        max_anchors: Max anchor frames per step. Bounds loss computation.
        max_negatives: Max negative candidates per step. Bounds distance matrix.
    """

    def __init__(
        self,
        token_list: list[str],
        encoder_dim: int,
        mlp_hidden: int = 256,
        margin: float = 0.3,
        lambda_crossmodal: float = 0.1,
        mining: str = "semi_hard",
        max_anchors: int = 256,
        max_negatives: int = 2048,
    ):
        super().__init__()
        self.margin = margin
        self.lambda_crossmodal = lambda_crossmodal
        self.mining = mining
        self.max_anchors = max_anchors
        self.max_negatives = max_negatives

        # Build fixed articulatory embeddings
        art_features, valid_mask = build_panphon_feature_matrix(token_list)
        panphon_dim = art_features.shape[1]
        self.panphon_dim = panphon_dim

        # Registered as buffers — never receive gradients
        self.register_buffer("articulatory_embeddings", art_features)  # (V, F)
        self.register_buffer("valid_phone_mask", valid_mask)  # (V,)

        # Blank id
        blank_id = 0
        for i, tok in enumerate(token_list):
            if tok == "<blank>":
                blank_id = i
                break
        self.blank_id = blank_id

        # Acoustic projection MLP: D → F (panphon feature space)
        self.acoustic_mlp = nn.Sequential(
            nn.Linear(encoder_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, panphon_dim),
        )

        # ---- Diagnostics: running EMA of phone marginal for drift tracking ----
        V = len(token_list)
        self.register_buffer("_marginal_ema", torch.ones(V) / V)  # init uniform
        self.register_buffer("_ema_initialized", torch.tensor(False))
        self.ema_decay = 0.99  # how much history to keep
        self.min_artic_norm = 0.1  # frames with expected artic norm below this are excluded

        log.info(
            f"ArticulatoryTripletLoss: encoder_dim={encoder_dim}, "
            f"panphon_dim={panphon_dim}, margin={margin}, mining={mining}, "
            f"max_anchors={max_anchors}, max_negatives={max_negatives}"
        )

    # ------------------------------------------------------------------ #
    #  Core projections
    # ------------------------------------------------------------------ #

    def compute_expected_articulatory(
        self, ctc_posteriors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Posterior-weighted articulatory embeddings, L2-normalized.

        Args:
            ctc_posteriors: (B, T, V)
        Returns:
            normalized: (B, T, F) — L2-normalized expected articulatory embeddings.
            norms: (B, T) — pre-normalization L2 norms.
                Low norm means posterior mass is concentrated on phones with
                zero/missing articulatory features (blank, special tokens,
                PanPhon failures) or on phones that cancel articulatorily.
                Use this to filter degenerate frames.
        """
        expected = torch.matmul(ctc_posteriors, self.articulatory_embeddings)
        norms = expected.norm(p=2, dim=-1)  # (B, T)
        normalized = F.normalize(expected, p=2, dim=-1)
        return normalized, norms

    def project_acoustics(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Project encoder features into articulatory space, L2-normalized.

        Args:
            encoder_out: (B, T, D) — caller must .detach() for stop-gradient.
        Returns:
            (B, T, F)
        """
        projected = self.acoustic_mlp(encoder_out)
        return F.normalize(projected, p=2, dim=-1)

    # ------------------------------------------------------------------ #
    #  Frame masking and subsampling
    # ------------------------------------------------------------------ #

    def _build_frame_mask(
        self,
        ctc_posteriors: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> torch.Tensor:
        """True for frames that are valid for triplet loss. Shape (B, T).

        Excludes:
          - Padded frames (beyond encoder_out_lens)
          - Frames where blank is the argmax
          - Frames where argmax phone has no valid articulatory features
            (PanPhon parse failure → zero vector → degenerate distances)
        """
        B, T, _ = ctc_posteriors.shape
        device = ctc_posteriors.device
        arange = torch.arange(T, device=device).unsqueeze(0)
        not_padded = arange < encoder_out_lens.unsqueeze(1)
        argmax_ids = ctc_posteriors.argmax(dim=-1)  # (B, T)
        not_blank = argmax_ids != self.blank_id
        # Exclude frames whose argmax phone has no articulatory features
        has_artic = self.valid_phone_mask[argmax_ids]  # (B, T) bool
        return not_padded & not_blank & has_artic

    def _subsample(
        self,
        acoustic_flat: torch.Tensor,
        artic_flat: torch.Tensor,
        utt_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample anchors (uniform) and negatives (stratified by utterance).

        Returns:
            anc_acoustic: (A, F) — keeps grad
            anc_artic:    (A, F) — keeps grad via posteriors
            anc_utt:      (A,)
            neg_artic:    (C, F) — detached, for mining only
            neg_utt:      (C,)
        """
        N = acoustic_flat.shape[0]
        device = acoustic_flat.device

        # --- Anchors: uniform random ---
        A = min(self.max_anchors, N)
        anc_idx = torch.randperm(N, device=device)[:A]
        anc_acoustic = acoustic_flat[anc_idx]  # keeps grad
        anc_artic = artic_flat[anc_idx]  # keeps grad
        anc_utt = utt_flat[anc_idx]

        # --- Negatives: stratified across utterances ---
        unique_utts = utt_flat.unique()
        U = unique_utts.numel()
        per_utt = max(1, self.max_negatives // U)

        neg_parts = []
        for u in unique_utts:
            utt_idx = (utt_flat == u).nonzero(as_tuple=True)[0]
            k = min(per_utt, utt_idx.numel())
            chosen = utt_idx[torch.randperm(utt_idx.numel(), device=device)[:k]]
            neg_parts.append(chosen)
        neg_idx = torch.cat(neg_parts)

        # Cap if we overshot
        if neg_idx.numel() > self.max_negatives:
            neg_idx = neg_idx[
                torch.randperm(neg_idx.numel(), device=device)[: self.max_negatives]
            ]

        neg_artic = artic_flat[neg_idx].detach()
        neg_utt = utt_flat[neg_idx]

        return anc_acoustic, anc_artic, anc_utt, neg_artic, neg_utt

    # ------------------------------------------------------------------ #
    #  Mining + loss computation on the small (A, C) matrix
    # ------------------------------------------------------------------ #

    def _mine_and_compute(
        self,
        anc_acoustic: torch.Tensor,
        anc_artic: torch.Tensor,
        anc_utt: torch.Tensor,
        neg_artic: torch.Tensor,
        neg_utt: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int, float, float]:
        """Mine negatives and compute triplet loss.

        Distance matrix: (A, C) where A ≤ max_anchors, C ≤ max_negatives.
        With defaults: (256, 2048) × 24 floats ≈ 2MB.

        Returns:
            loss: scalar (has grad; zero tensor if no valid triplets).
            num_active: int.
            num_total: int.
            mean_pos_dist: float — average positive distance.
            mean_neg_dist: float — average chosen negative distance.
        """
        device = anc_acoustic.device
        A = anc_acoustic.shape[0]

        # Positive distances: d(h_anchor, ã_anchor) — (A,)
        d_pos = (anc_acoustic - anc_artic).pow(2).sum(dim=-1)

        # All anchor-to-negative distances: (A, C)
        # Manual broadcast to preserve grad through anc_acoustic
        d_neg_all = (
            (anc_acoustic.unsqueeze(1) - neg_artic.unsqueeze(0)).pow(2).sum(dim=-1)
        )

        # Cross-utterance mask: True = valid negative
        cross_utt = anc_utt.unsqueeze(1) != neg_utt.unsqueeze(0)  # (A, C)

        # Mask same-utterance → inf
        d_neg_masked = d_neg_all.masked_fill(~cross_utt, float("inf"))

        # Any valid negatives per anchor?
        has_valid = cross_utt.any(dim=1)  # (A,)
        if not has_valid.any():
            return (
                torch.zeros(1, device=device, requires_grad=True).squeeze(),
                0, 0, 0.0, 0.0,
            )

        # --- Select negative per anchor based on mining strategy ---
        if self.mining == "hard":
            d_neg_chosen, _ = d_neg_masked.min(dim=1)

        elif self.mining == "semi_hard":
            lower = d_pos.unsqueeze(1)
            upper = lower + self.margin
            sh_mask = (d_neg_masked > lower) & (d_neg_masked < upper) & cross_utt
            has_sh = sh_mask.any(dim=1)

            sh_dists = d_neg_masked.masked_fill(~sh_mask, float("inf"))
            sh_chosen, _ = sh_dists.min(dim=1)
            hard_chosen, _ = d_neg_masked.min(dim=1)

            d_neg_chosen = torch.where(has_sh, sh_chosen, hard_chosen)

        else:  # random
            rand_scores = torch.rand(A, d_neg_all.shape[1], device=device)
            rand_scores.masked_fill_(~cross_utt, -1.0)
            chosen_idx = rand_scores.argmax(dim=1)
            d_neg_chosen = d_neg_all.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)

        # Filter anchors with no finite negative
        valid = has_valid & torch.isfinite(d_neg_chosen)
        if not valid.any():
            return (
                torch.zeros(1, device=device, requires_grad=True).squeeze(),
                0, 0, 0.0, 0.0,
            )

        d_p = d_pos[valid]
        d_n = d_neg_chosen[valid]
        M = d_p.shape[0]

        triplet_losses = F.relu(d_p - d_n + self.margin)
        loss = triplet_losses.mean()
        num_active = int((triplet_losses > 0).sum().item())

        mean_pos_dist = d_p.mean().item()
        mean_neg_dist = d_n.mean().item()

        return loss, num_active, M, mean_pos_dist, mean_neg_dist

    # ------------------------------------------------------------------ #
    #  Diagnostics (all O(N), no extra distance matrices)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _compute_diagnostics(
        self,
        ctc_posteriors: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute diagnostic stats for logging. All cheap O(B*T) ops.

        Computed on ALL non-padded frames (not filtered by articulatory validity),
        giving a full picture of the model's posterior behavior.

        Stats returned:
            posterior_entropy:
                Mean entropy of π_t across valid (non-padded) frames.
                High → spread posteriors (less confident).
                Low → peaky posteriors (CTC-typical).
                Track over training: if triplet loss works, this may increase
                slightly as the model considers acoustically plausible alternatives.

            blank_rate:
                Fraction of non-padded frames where blank is argmax.
                Typical CTC: 0.5–0.8. Should stay roughly stable.

            top1_confidence:
                Mean probability mass on the argmax phone (excluding blank frames).
                1.0 = perfectly confident, 1/V = uniform.
                Complements entropy with an intuitive scale.

            phone_marginal_entropy:
                Entropy of the batch-averaged phone distribution
                (marginal across all non-blank valid frames).
                High → predictions spread across many phones.
                Low → predictions concentrated on few phones.

            phone_marginal_drift:
                KL(current_batch_marginal || running_EMA_marginal).
                Measures how much the phone distribution is shifting step-to-step.
                Spikes indicate the triplet loss is actively changing predictions.
                Near zero = stable.

            num_valid_frames:
                Total non-blank, non-padded frames in this batch.
        """
        B, T, V = ctc_posteriors.shape
        device = ctc_posteriors.device
        stats: Dict[str, float] = {}

        # ---- Padding mask (non-padded frames, including blank) ----
        arange = torch.arange(T, device=device).unsqueeze(0)
        not_padded = arange < encoder_out_lens.unsqueeze(1)  # (B, T)
        n_not_padded = not_padded.sum().item()

        if n_not_padded == 0:
            return {
                "posterior_entropy": 0.0,
                "blank_rate": 1.0,
                "top1_confidence": 0.0,
                "phone_marginal_entropy": 0.0,
                "phone_marginal_drift": 0.0,
                "num_valid_frames": 0,
            }

        posteriors_valid = ctc_posteriors[not_padded]  # (N_all, V)

        # ---- Posterior entropy: H(π_t) averaged over non-padded frames ----
        log_probs = torch.log(posteriors_valid.clamp(min=1e-10))
        entropies = -(posteriors_valid * log_probs).sum(dim=-1)  # (N_all,)
        stats["posterior_entropy"] = entropies.mean().item()

        # ---- Blank rate ----
        argmax_ids = posteriors_valid.argmax(dim=-1)  # (N_all,)
        is_blank = argmax_ids == self.blank_id
        stats["blank_rate"] = is_blank.float().mean().item()

        # ---- Top-1 confidence on non-blank frames ----
        non_blank_mask = ~is_blank
        n_non_blank = non_blank_mask.sum().item()
        stats["num_valid_frames"] = n_non_blank

        if n_non_blank > 0:
            top1_probs = posteriors_valid.max(dim=-1).values  # (N_all,)
            stats["top1_confidence"] = top1_probs[non_blank_mask].mean().item()
        else:
            stats["top1_confidence"] = 0.0

        # ---- Phone marginal distribution (over non-blank valid frames) ----
        if n_non_blank > 0:
            # Average posterior = marginal phone distribution for this batch
            marginal = posteriors_valid[non_blank_mask].mean(dim=0)  # (V,)
            marginal = marginal.clamp(min=1e-10)
            marginal = marginal / marginal.sum()  # renormalize for safety

            # Entropy of the marginal
            marginal_log = torch.log(marginal)
            stats["phone_marginal_entropy"] = -(marginal * marginal_log).sum().item()

            # ---- Phone marginal drift: KL(batch || running_EMA) ----
            ema = self._marginal_ema.clamp(min=1e-10)
            ema = ema / ema.sum()
            kl = (marginal * (marginal_log - torch.log(ema))).sum().item()
            stats["phone_marginal_drift"] = max(0.0, kl)  # clamp numerical noise

            # Update running EMA
            if not self._ema_initialized.item():
                self._marginal_ema.copy_(marginal)
                self._ema_initialized.fill_(True)
            else:
                self._marginal_ema.mul_(self.ema_decay).add_(
                    marginal, alpha=1.0 - self.ema_decay
                )
        else:
            stats["phone_marginal_entropy"] = 0.0
            stats["phone_marginal_drift"] = 0.0

        return stats

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ctc_posteriors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute articulatory triplet auxiliary loss and diagnostics.

        IMPORTANT: encoder_out should be .detach()'d by the caller (stop-gradient).
        ctc_posteriors should NOT be detached (gradients flow to CTC head).

        Args:
            encoder_out: (B, T, D) encoder output (detached).
            encoder_out_lens: (B,) valid lengths.
            ctc_posteriors: (B, T, V) softmax posteriors (not detached).

        Returns:
            Dict with loss keys (have grad) and diagnostic keys (floats, for logging):
                loss_aux_total: weighted loss (scalar, always has grad)
                loss_triplet_crossmodal: unweighted loss (detached)
                num_active_triplets: int
                num_total_triplets: int
                mean_positive_dist: float — avg d(acoustic, articulatory) for anchors
                mean_negative_dist: float — avg d(acoustic, neg_articulatory) chosen
                posterior_entropy: float
                blank_rate: float
                top1_confidence: float
                phone_marginal_entropy: float
                phone_marginal_drift: float
                num_valid_frames: int
        """
        B, T, D = encoder_out.shape
        device = encoder_out.device

        zero = torch.zeros(1, device=device, requires_grad=True).squeeze()
        zero_stats = {
            "loss_aux_total": zero,
            "loss_triplet_crossmodal": torch.tensor(0.0, device=device),
            "num_active_triplets": 0,
            "num_total_triplets": 0,
            "mean_positive_dist": 0.0,
            "mean_negative_dist": 0.0,
            "posterior_entropy": 0.0,
            "blank_rate": 0.0,
            "top1_confidence": 0.0,
            "phone_marginal_entropy": 0.0,
            "phone_marginal_drift": 0.0,
            "num_valid_frames": 0,
            "num_triplet_frames": 0,
            "mean_artic_norm": 0.0,
        }

        # 1. Project acoustics (on detached encoder output)
        acoustic_proj = self.project_acoustics(encoder_out)  # (B, T, F)

        # 2. Expected articulatory embeddings (grad flows through posteriors)
        artic_expected, artic_norms = self.compute_expected_articulatory(
            ctc_posteriors
        )  # (B, T, F), (B, T)

        # 3. Frame mask: valid + non-blank + valid articulatory argmax
        frame_mask = self._build_frame_mask(ctc_posteriors, encoder_out_lens)

        # 4. Additional norm filter: exclude frames where the expected articulatory
        #    vector has near-zero norm before normalization. This catches edge cases:
        #    - Posterior mass concentrated on phones with zero articulatory features
        #      (PanPhon failures that passed the argmax check because argmax was valid
        #       but the rest of the posterior mass was on zero-vector phones)
        #    - Posterior mass on phones whose articulatory vectors nearly cancel
        #    Threshold: 0.1 means we need at least ~10% effective posterior mass on
        #    phones with non-cancelling articulatory features.
        artic_norm_ok = artic_norms > self.min_artic_norm  # (B, T)
        frame_mask = frame_mask & artic_norm_ok

        # ---- Diagnostics (cheap, on full batch, no grad) ----
        # Computed on ALL non-padded frames, not filtered by articulatory validity.
        # This gives a full picture of the model's posterior behavior.
        diag_stats = self._compute_diagnostics(ctc_posteriors, encoder_out_lens)

        # 5. Utterance ids
        utt_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, T)

        # Flatten valid frames
        acoustic_flat = acoustic_proj[frame_mask]  # (N, F)
        artic_flat = artic_expected[frame_mask]  # (N, F)
        utt_flat = utt_ids[frame_mask]  # (N,)

        N = acoustic_flat.shape[0]

        # Add filtering stats (how many frames survived all filters)
        diag_stats["num_triplet_frames"] = N
        diag_stats["mean_artic_norm"] = (
            artic_norms[frame_mask].mean().item() if N > 0 else 0.0
        )

        if N < 2 or utt_flat.unique().numel() < 2:
            zero_stats.update(diag_stats)
            return zero_stats

        # 5. Subsample anchors + stratified negative pool
        anc_acoustic, anc_artic, anc_utt, neg_artic, neg_utt = self._subsample(
            acoustic_flat, artic_flat, utt_flat
        )

        if anc_acoustic.shape[0] == 0 or neg_artic.shape[0] == 0:
            zero_stats.update(diag_stats)
            return zero_stats

        # 6. Mine and compute on small (A, C) matrix
        loss_raw, num_active, num_total, d_pos_mean, d_neg_mean = (
            self._mine_and_compute(
                anc_acoustic, anc_artic, anc_utt, neg_artic, neg_utt
            )
        )

        loss_total = self.lambda_crossmodal * loss_raw

        result = {
            "loss_aux_total": loss_total,
            "loss_triplet_crossmodal": loss_raw.detach(),
            "num_active_triplets": num_active,
            "num_total_triplets": num_total,
            "mean_positive_dist": d_pos_mean,
            "mean_negative_dist": d_neg_mean,
        }
        result.update(diag_stats)
        return result