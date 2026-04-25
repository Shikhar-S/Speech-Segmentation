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
    
    def __call__(self, logprobs, input_lengths, target, target_lengths, **kwargs):
        label, score = forced_align(logprobs, target, input_lengths, target_lengths, blank=self.blank_idx)
        return label, score
        

if __name__=='__main__':
    # TODO(shikhar): check this works!
    B, S, C = 2, 5, 3
    T = 4
    logprobs = torch.randn(B, S, C).log_softmax(dim=-1)
    input_lengths = torch.tensor([5, 4])
    target = torch.tensor([[1, 2, 0, 0], [1, 0, 0, 0]])
    target_lengths = torch.tensor([2, 1])
    
    strategy = ForcedAlignmentInference(blank_idx=0)
    labels, scores = strategy(logprobs, input_lengths, target, target_lengths)
    print("Aligned labels:", labels)
    print("Alignment scores:", scores)