import torch
import torch.nn as nn
import pandas as pd
import panphon
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
from src.model.heads.base_head import TaskType


def _maybe3d2latlong(arr):
    if not isinstance(arr, np.ndarray):
        return arr
    if len(arr) != 3:
        return arr

    x, y, z = arr

    # normalize to unit sphere
    norm = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm

    # radians
    latitude_rad = np.arcsin(z)
    longitude_rad = np.arctan2(y, x)

    # degrees
    # latitude = np.degrees(latitude_rad)
    # longitude = np.degrees(longitude_rad)

    return np.stack([latitude_rad, longitude_rad], axis=0)


class ModelInterpreter:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = model.device
        self.model.eval()
        self.model.zero_grad()
        self._ft = None  # Lazy load panphon

    @property
    def ft(self):
        if self._ft is None:
            self._ft = panphon.FeatureTable()
        return self._ft

    def _get_embedding_layer(self) -> nn.Module:
        """Auto-detects embedding layer."""
        candidates = [
            "net.embedding",
            "net.embeddings",
            "net.encoder.embeddings",
            "net.wte",
        ]
        for name, module in self.model.named_modules():
            if name in candidates or isinstance(module, nn.Embedding):
                return module
        raise ValueError("Could not auto-detect embedding layer.")

    def _get_broad_category(self, token: str) -> str:
        """Classifies an IPA token/character into broad categories."""
        special_tokens = {
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
            "<os>",
            "<sil>",
            "[PAD]",
            " ",
            "",
        }
        if token in special_tokens:
            return "Silence/Special"

        try:
            vectors = self.ft.word_fts(token)
            if not vectors:
                return "Other"
            vec = vectors[0]  # Analyze primary segment
        except:
            return "Other"

        # Categorization Logic
        # Panphon uses 1 for '+', -1 for '-', and 0 for '0'
        is_p = lambda f: vec[f] == 1
        is_n = lambda f: vec[f] == -1

        if is_p("syl"):
            return "Vowel"
        if is_p("nas"):
            return "Nasal"

        # 'lat' (lateral) is a sub-feature; ensure safety if your model creates sparse segments
        if is_p("lat") or (is_p("son") and is_p("cons") and is_n("nas")):
            return "Liquid"

        if is_p("son") and is_n("cons"):
            return "Glide"
        if is_n("son"):
            return "Fricative" if is_p("cont") else "Stop"

        return "Other"

    def integrated_gradients_text(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
        tokenizer: Any,
        steps: int = 50,
        batch_size_multiplier: int = 8,
    ) -> List[Dict[str, Any]]:
        input_ids, lengths, targets = (
            input_ids.to(self.device),
            lengths.to(self.device),
            targets.to(self.device),
        )
        emb_layer = self._get_embedding_layer()
        results = []

        with torch.backends.cudnn.flags(enabled=False):
            for i in range(len(input_ids)):
                curr_input = input_ids[i : i + 1]
                curr_len = lengths[i : i + 1]
                curr_target = targets[i : i + 1]

                # 1. Capture Embeddings
                captured_emb = None

                def hook(m, i, o):
                    nonlocal captured_emb
                    captured_emb = o.detach()

                h = emb_layer.register_forward_hook(hook)
                self.model(curr_input, curr_len)
                h.remove()

                baseline_emb = torch.zeros_like(captured_emb)

                # 2. Interpolate
                alphas = torch.linspace(0, 1, steps + 1, device=self.device).view(
                    -1, 1, 1
                )
                interpolated = baseline_emb + alphas * (captured_emb - baseline_emb)
                interpolated.requires_grad_(True)

                # 3. Compute Gradients (Chunked)
                grads = []

                def override(m, i, o):
                    return chunk_emb

                h_over = emb_layer.register_forward_hook(override)
                predictions = None

                try:
                    for start in range(0, steps + 1, batch_size_multiplier):
                        end = min(start + batch_size_multiplier, steps + 1)
                        chunk_emb = interpolated[start:end]
                        chunk_emb.retain_grad()

                        self.model.zero_grad()
                        b = {
                            "text": curr_input,
                            "lengths": curr_len,
                            "target": curr_target,
                        }
                        ret = self.model.model_step(b)
                        score = -ret["loss"]
                        predictions = ret["preds"].cpu().numpy()
                        score.backward()
                        grads.append(chunk_emb.grad.detach().cpu())
                finally:
                    h_over.remove()

                # 4. Aggregate
                avg_grads = torch.cat(grads, dim=0)[:-1].mean(dim=0).to(self.device)
                attributions = (
                    ((captured_emb - baseline_emb) * avg_grads).sum(dim=-1).squeeze(0)
                )
                attributions = attributions / (torch.norm(attributions) + 1e-9)

                # 5. Decode
                seq_l = curr_len.item()
                raw_ids = curr_input[0, :seq_l].cpu().tolist()
                # Handle char vs subword tokenizer differences for "raw_tokens" list
                raw_tokens = [tokenizer.decode([rid]) for rid in raw_ids]
                scores = attributions[:seq_l].cpu().numpy()
                results.append(
                    {
                        "tokens": raw_tokens,  # List of str
                        "attributions": scores,  # Numpy array
                        "input_ids": raw_ids,  # List of int
                        "true_class": curr_target.cpu().numpy(),  # List of int/float
                        "pred_class": _maybe3d2latlong(predictions[0]),
                    }
                )
        return results

    # def aggregate_by_phonetic_class(self, ig_results: List[Dict[str, Any]]) -> pd.DataFrame:
    #     """
    #     Aggregates IG results into a DataFrame for visualization.
    #     """
    #     stats = defaultdict(lambda: {'total_abs_score': 0.0, 'count': 0})

    #     for res in ig_results:
    #         tokens = res['tokens']
    #         scores = res['attributions']

    #         for token, score in zip(tokens, scores):
    #             # If token is multiple chars (e.g. 'tʃ'), panphon usually handles it or takes first char properties
    #             # which is sufficient for broad 'Vowel vs Consonant' analysis.
    #             cat = self._get_broad_category(token)
    #             stats[cat]['total_abs_score'] += abs(score)
    #             stats[cat]['count'] += 1

    #     data = [{
    #         'Category': cat,
    #         'Mean_Abs_Attribution': vals['total_abs_score'] / vals['count'],
    #         'Count': vals['count']
    #     } for cat, vals in stats.items() if vals['count'] > 0]

    #     return pd.DataFrame(data).sort_values('Mean_Abs_Attribution', ascending=False)

    def aggregate_by_phonetic_class(
        self, ig_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Returns long-form data (one row per token) to enable statistical error bars."""
        return pd.DataFrame(
            [
                {"Category": self._get_broad_category(token), "Attribution": abs(score)}
                for res in ig_results
                for token, score in zip(res["tokens"], res["attributions"])
            ]
        )
