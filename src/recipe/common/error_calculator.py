"""Adapter over :func:`phone_metrics.phone_error_rates`.

Callable with ``(ys_hat, ys_pad, ys_pad_lens)``; returns a metric dict
that Lightning loggers key by ``cer_ctc``, ``per_ctc``, etc.

Keys emitted:
    cer  — character-level edit rate on the joined token string.
    per  — phone error rate (Levenshtein on panphon-segmented IPA).
    pfer — phonological feature error rate.
    ins  — insertion rate (phones, %).
    del  — deletion rate (phones, %).
    sub  — substitution rate (phones, %).

FER (frame error rate) is not emitted by this adapter.
"""

from typing import List

import numpy as np
import panphon
import torch
from phone_metrics import Utterance, canonical_ipa, phone_error_rates
from phone_metrics.timit import Seg
from rapidfuzz.distance import Levenshtein


class ErrorCalculator:
    """CTC-based phone recognition error calculator.

    Greedy-decodes the model's CTC outputs, collapses repeats and blanks,
    and reports CER plus (optionally) PER / PFER / INS / DEL / SUB. The
    detailed phone metrics are computed via :mod:`phone_metrics`.

    Args:
        token_list: Vocabulary used by the CTC head (index → token string).
        blank_id: CTC blank index.
        sym_space: Vocab entry for the inter-word space (filtered out).
        ignore_id: Pad-id used by the dataloader to mark missing positions.
        log_phone_metrics: If ``False``, only CER is computed.
    """

    def __init__(
        self,
        token_list: List[str],
        blank_id: int,
        sym_space: str = "<space>",
        ignore_id: int = -1,
        log_phone_metrics: bool = True,
    ):
        self.token_list = np.array(token_list)
        self.blank_id = blank_id
        self.ignore_id = ignore_id
        self.idx_space = (
            token_list.index(sym_space) if sym_space in token_list else None
        )
        self.log_phone_metrics = log_phone_metrics
        self._segmenter = (
            panphon.featuretable.FeatureTable() if log_phone_metrics else None
        )

    def _ctc_collapse_batch(self, x: torch.Tensor):
        if x.numel() == 0:
            return x, torch.zeros(0, device=x.device, dtype=torch.long)

        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 1:] = x[:, 1:] != x[:, :-1]

        lengths = mask.sum(1)
        max_len = int(lengths.max().item())
        out = torch.full(
            (x.size(0), max_len),
            self.ignore_id,
            device=x.device,
            dtype=x.dtype,
        )

        pos = mask.long().cumsum(1) - 1
        batch_offsets = (
            torch.arange(x.size(0), device=x.device).unsqueeze(1) * max_len
        )
        flat_dest_indices = (pos + batch_offsets)[mask]
        out.view(-1)[flat_dest_indices] = x[mask]

        return out, lengths

    def _ids_to_strs(
        self, ids_cpu: np.ndarray, lens_cpu: np.ndarray
    ) -> List[str]:
        """Decode each row of token ids to its joined string, dropping blank/space/pad."""
        results = []
        t_list = self.token_list
        i_id, b_id, s_id = self.ignore_id, self.blank_id, self.idx_space
        for i in range(len(ids_cpu)):
            row = ids_cpu[i, : lens_cpu[i]]
            filtered = row[(row != i_id) & (row != b_id) & (row != s_id)]
            results.append("".join(t_list[filtered]))
        return results

    def _segment_to_phones(self, text: str) -> List[str]:
        """panphon IPA segmentation with canonical normalization per phone."""
        segs = self._segmenter.ipa_segs(text)
        return [canonical_ipa(p) or p for p in segs if p]

    def _phone_metrics(
        self, hyp_strs: List[str], ref_strs: List[str]
    ) -> dict[str, float]:
        """Build Utterance objects, call phone_error_rates, compute INS/DEL/SUB."""
        hyp_phones = [self._segment_to_phones(h) for h in hyp_strs]
        ref_phones = [self._segment_to_phones(r) for r in ref_strs]
        # Drop pairs where the reference has no segmented phones — phone-metrics
        # would divide by zero.
        kept = [
            (hp, rp) for hp, rp in zip(hyp_phones, ref_phones) if rp
        ]
        if not kept:
            return {"per": 0.0, "pfer": 0.0, "ins": 0.0, "del": 0.0, "sub": 0.0}
        hyp_phones, ref_phones = map(list, zip(*kept))
        utterances = [
            Utterance(
                audio_path=str(i),
                language="eng",
                split="train",
                segments=[Seg(0.0, 0.0, p, p) for p in ref],
            )
            for i, ref in enumerate(ref_phones)
        ]
        result = phone_error_rates(
            utterances, hyp_phones, label="ipa", pfer=True
        )
        # INS/DEL/SUB via per-utterance editops on segmented phone lists.
        ins = dels = sub = 0
        for hp, rp in zip(hyp_phones, ref_phones):
            for op in Levenshtein.editops(hp, rp):
                if op.tag == "insert":
                    ins += 1
                elif op.tag == "delete":
                    dels += 1
                elif op.tag == "replace":
                    sub += 1
        ref_total = sum(len(r) for r in ref_phones)
        return {
            "per": result.per * 100,
            "pfer": result.pfer * 100,
            "ins": ins / ref_total * 100 if ref_total else 0.0,
            "del": dels / ref_total * 100 if ref_total else 0.0,
            "sub": sub / ref_total * 100 if ref_total else 0.0,
        }

    def __call__(
        self,
        ys_hat: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> dict[str, float]:
        collapsed_hat, hat_lens = self._ctc_collapse_batch(ys_hat)
        if ys_hat.is_cuda:
            torch.cuda.synchronize()

        hat_np = collapsed_hat.cpu().numpy()
        hat_lens_np = hat_lens.cpu().numpy()
        ref_np = ys_pad.cpu().numpy()
        ref_lens_np = ys_pad_lens.cpu().numpy()

        hyps = self._ids_to_strs(hat_np, hat_lens_np)
        refs = self._ids_to_strs(ref_np, ref_lens_np)
        pairs = [(h, r) for h, r in zip(hyps, refs) if len(r) > 0]

        metrics = {"cer": 0.0, "per": 0.0, "pfer": 0.0}
        if not pairs:
            return metrics

        v_hyp, v_ref = zip(*pairs)
        cer_dist = sum(
            Levenshtein.distance(h, r) for h, r in zip(v_hyp, v_ref)
        )
        total_chars = sum(len(r) for r in v_ref)
        metrics["cer"] = cer_dist / total_chars * 100 if total_chars else 0.0

        if self.log_phone_metrics:
            metrics.update(self._phone_metrics(list(v_hyp), list(v_ref)))
        return metrics
