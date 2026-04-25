"""Forced alignment with Viterbi decoding strategy for segmentation.

Input expects both the predicted logprobs and the target phone sequence to align with.

Usage:
    python -m src.recipe.common.forced_alignment_strategy
"""
import torch
from torchaudio.functional import forced_align


class ForcedAlignmentInference:
    def __init__(self, blank_idx=0):
        self.blank_idx = blank_idx
    
    def __call__(self, logprobs, input_lengths, target, **kwargs):
        """
        Args:
            logprobs: list of tensors of shape (T_i, C) containing log probabilities for each frame
            input_lengths: list of ints, lengths of the input sequences, shape (B,)
            target: list of lists of int, target phone sequences for each utterance
        Returns:
            aligned_labels: list of lists of int, aligned phone labels for each frame
            alignment_scores: list of lists of float, alignment scores for each frame
        """
        labels=[]
        scores=[]
        bs=len(logprobs)
        device=logprobs[0].device
        for bidx in range(bs):
            tgt=torch.tensor(target[bidx], dtype=torch.long, device=device).unsqueeze(0)
            tgtlen=torch.tensor(len(target[bidx]), dtype=torch.long, device=device).unsqueeze(0)
            logp=logprobs[bidx][:input_lengths[bidx],:].unsqueeze(0)  # (1, T, C)
            ilen=torch.tensor(input_lengths[bidx], dtype=torch.long, device=device).unsqueeze(0)
            label, score = forced_align(logp, tgt, ilen, tgtlen, blank=self.blank_idx)
            labels.append(label[0])
            scores.append(score[0])
        return labels, scores
        
if __name__=='__main__':
    B, S, C = 2, 10, 3
    logprobs = torch.randn(B, S, C).log_softmax(dim=-1)
    logprobs = [logprobs[i] for i in range(B)] # convert to list of tensors
    input_lengths = [6, 8]
    target = [[1, 2, 1], [2, 1]]
    aligner = ForcedAlignmentInference(blank_idx=0)
    labels, scores = aligner(logprobs, input_lengths, target)
    print("Aligned labels:", labels)
    print("Alignment scores:", scores)