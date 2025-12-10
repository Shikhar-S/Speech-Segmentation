import torch


def get_kv_pooling_mask(lengths):
    max_len = lengths.max()
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) >= lengths.unsqueeze(1)
    return mask  # (B, T)
