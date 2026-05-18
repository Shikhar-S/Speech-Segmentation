"""Vendored Kreuk UnsupSeg inference utilities.

Copied near-verbatim from github.com/felixkreuk/UnsupSeg.
"""

import numpy as np
import torch
from scipy.signal import find_peaks


def replicate_first_k_frames(x, k, dim):
    return torch.cat(
        [
            x.index_select(
                dim=dim,
                index=torch.LongTensor([0] * k).to(x.device),
            ),
            x,
        ],
        dim=dim,
    )


def max_min_norm(x):
    x -= x.min(-1, keepdim=True)[0]
    x /= x.max(-1, keepdim=True)[0]
    return x


def detect_peaks(x, lengths, prominence=0.1, width=None, distance=None):
    """Detect peaks in next-frame-classifier dissimilarity scores."""
    out = []
    for xi, li in zip(x, lengths):
        if isinstance(xi, torch.Tensor):
            xi = xi.cpu().detach().numpy()
        xi = xi[:li]
        xmin, xmax = xi.min(), xi.max()
        xi = (xi - xmin) / (xmax - xmin)
        peaks, _ = find_peaks(
            xi, prominence=prominence, width=width, distance=distance
        )
        if len(peaks) == 0:
            peaks = np.array([len(xi) - 1])
        out.append(peaks)
    return out
