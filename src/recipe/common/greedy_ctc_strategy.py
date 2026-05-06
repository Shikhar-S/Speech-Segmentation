import torch
from typing import List, Dict, Any, Union


def ctc_collapse_vectorized(
    ids: torch.Tensor, blank_id: int, ignore_id: int = -1
) -> List[List[int]]:
    """Optimized CTC collapse for batch tensors."""
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, 1:] = (
        ids[:, 1:] != ids[:, :-1]
    )  # true if this pred is not same as previous
    mask &= ids != blank_id  # true if this pred is not blank
    if ignore_id != -1:
        mask &= ids != ignore_id  # true if this pred is not ignore_id

    return [ids[i][mask[i]].tolist() for i in range(ids.size(0))]


class GreedyCTCInference:
    """A scalable inference engine for any CTC-based phone recognizer."""

    def __init__(self, token_list: List[str], blank_id: int):
        self.token_list = token_list
        self.blank_id = blank_id

    @torch.no_grad()
    def __call__(
        self,
        model: torch.nn.Module,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        features: torch.Tensor = None,
        logits: torch.Tensor = None,
        feature_lens: torch.Tensor = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Return schema:
            List of dicts, one per utterance, with keys
            - "processed_transcript": post-processed text (e.g. no special tokens)
            - "predicted_transcript": raw text from token mapping (e.g. with special tokens)
            - "ids": List[int] of predicted token ids (after CTC collapse)
            - "logits": Optional[torch.Tensor] of frame logits (if return_logits=True)
        """

        if logits is None:
            if features is None:
                # 1. Standardized Forward pass
                # Works as long as model has .encode() and .ctc
                encoder_out, _ = model.encode(speech, speech_lengths)
                if isinstance(encoder_out, tuple):
                    encoder_out = encoder_out[0]
            else:
                encoder_out = features
            logits = model.ctc.ctc_lo(encoder_out)

        # 2. Greedy search
        y_hat = torch.argmax(logits, dim=-1)

        # Mask padded frames to blank so collapse never produces tokens
        # past the true frame count (otherwise downstream forced_align gets
        # target_len > input_len at random/early stages).
        if feature_lens is not None:
            T = y_hat.shape[1]
            pad_mask = torch.arange(T, device=y_hat.device).unsqueeze(
                0
            ) >= feature_lens.to(y_hat.device).unsqueeze(1)
            y_hat = y_hat.masked_fill(pad_mask, self.blank_id)

        # 3. Collapse
        collapsed_ids = ctc_collapse_vectorized(y_hat, self.blank_id)

        # 4. Map to text
        results = []
        for b, ids in enumerate(collapsed_ids):
            tokens = [self.token_list[i] for i in ids]
            raw_text = "/".join(tokens)
            # Filter special tokens - for powsm and eos bos in some tokenizers
            # TODO(shikhar): this can cause issues. Tokenizer should maintain a special token list
            clean_tokens = [
                t for t in tokens if not (t.startswith("<") and t.endswith(">"))
            ]
            processed = "".join(
                clean_tokens
            ).strip()  # replace(self.sym_space, " ")
            result = {
                "processed_transcript": processed,
                "predicted_transcript": raw_text,
                "ids": ids,
            }
            if kwargs.get("return_logits", False):
                result["logits"] = logits[b : b + 1]  # 1,T,C
            results.append(result)
        return results
