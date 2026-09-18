"""Two pass strategy for segmentation with CTC/ASG models.
First run decode, then run forced alignment with predictions.

Usage:
    python -m src.recipe.common.decode_align_strategy
"""

class DecodeAlignStrategy:
    def __init__(self, decode_strategy, align_strategy):
        self.decode_strategy = decode_strategy
        self.align_strategy = align_strategy

    def __call__(self, net, speech, speech_lengths, features=None, logits=None, feature_lens=None, **kwargs):
        """Run decode strategy to get raw predictions, then align with forced alignment.
        NOTE(shikhar):
        If features are provided, greedy ctc decode strategy skips encoding,
        and runs decode directly on features.
        If logits are provided, it skips both encoding and decoding, and uses the provided logits.
        When logits are provided, net can be None since it won't be used.

        ``feature_lens`` is required: the per-utterance true frame count
        used by forced alignment. Without it, alignment would run over
        padded frames.

        Return schema:
            List of dicts, one per utterance, with keys
            - "processed_transcript": post-processed text (e.g. no special tokens)
            - "predicted_transcript": raw text from token mapping (e.g. with special tokens)
            - "ids": List[int] of predicted token ids (after CTC collapse)
            - "logits": Optional[torch.Tensor] of frame logits (if return_logits=True)
            - "aligned_labels": List[int] of aligned labels
            - "alignment_scores": List[float] of alignment scores
            First 4 keys come from decode strategy, last 2 keys come from align strategy.
        """
        assert feature_lens is not None, (
            "DecodeAlignStrategy requires feature_lens (per-utterance true frame counts)"
        )
        decode_results = self.decode_strategy(net, speech, speech_lengths, features, logits=logits, feature_lens=feature_lens, return_logits=True, **kwargs)
        logprobs = []
        targets = []
        for res in decode_results:
            l = res.get("logits")
            assert l is not None, "Decode strategy must return logits for alignment"
            logprobs.append(l.log_softmax(dim=-1).squeeze(0))  # (T, C)
            targets.append(res['ids'])
        input_lengths = [int(x) for x in feature_lens]

        aligned_labels, aligned_scores = self.align_strategy(logprobs, input_lengths, targets)

        decode_align_results = []
        for decode_res, labels, scores in zip(decode_results, aligned_labels, aligned_scores):
            decode_align_results.append({
                **decode_res,
                "aligned_labels": labels,
                "alignment_scores": scores,
            })
        return decode_align_results
