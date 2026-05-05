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
        decode_results = self.decode_strategy(net, speech, speech_lengths, features, logits=logits, return_logits=True, **kwargs)
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


if __name__=='__main__':
    import torch
    from src.recipe.common.forced_alignment_strategy import ForcedAlignmentInference
    from src.recipe.common.greedy_ctc_strategy import GreedyCTCInference
    
    #####
    from src.model.xeusphoneme.builders import build_xeus_pr_from_hf
    ckpt_path = '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/speech_segmentation/seg_pxeus_frac0_053/checkpoints/last.ckpt'
    net = build_xeus_pr_from_hf(
        work_dir='/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/cache/xeus',
        hf_repo='espnet/xeus',
        load_ckpt=False,
        vocab_file='src/model/xeusphoneme/resources/ipa_vocab.json',
        interctc_weight=0.3,
        interctc_layer_idx=[4, 8, 12],
        interctc_use_conditioning=True,
        ctc_weight=1.0,
    )
    #####
    decode_strategy = GreedyCTCInference(token_list=net.token_list, blank_id=0)
    align_strategy = ForcedAlignmentInference(blank_idx=0)
    strategy = DecodeAlignStrategy(decode_strategy, align_strategy)
    
    speech = torch.randn(1, 16000)  # 1 second of fake audio at 16kHz
    speech_lengths = torch.tensor([16000], dtype=torch.long)
    encoder_out, _ = net.encode(speech, speech_lengths)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]
    feature_lens = torch.tensor([encoder_out.shape[1]], dtype=torch.long)
    results = strategy(
        net=net, speech=speech, speech_lengths=speech_lengths,
        feature_lens=feature_lens,
    )
    print(results)