"""Two pass strategy for segmentation with CTC/ASG models.
First run decode, then run forced alignment with predictions.

Usage:
    python -m src.recipe.common.decode_align_strategy
"""

class DecodeAlignStrategy:
    def __init__(self, decode_strategy, align_strategy):
        self.decode_fn = decode_strategy
        self.align_fn = align_strategy

    def __call__(self, *args, **kwargs):
        decode_results = self.decode_fn(*args, **kwargs)
        align_results = self.align_fn(*args, **kwargs, **decode_results)
        return align_results


if __name__=='__main__':
    # TODO(shikhar): make this work!
    pass