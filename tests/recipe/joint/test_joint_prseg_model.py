"""Integration tests for JointPRSegModel and JointPRSegmentDataset."""

import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace

from src.recipe.joint.model_module import JointPRSegModel
from src.data.joint_prseg_dataset import (
    JointPRSegmentDataset,
    joint_prseg_collate_fn,
    _normalize_seg_item,
    _normalize_pr_item,
)


# -------------------------------------------------------------------
# DUMMY NET (supports both PR and Seg interfaces)
# -------------------------------------------------------------------


class DummyNet(nn.Module):
    ENCODER_DIM = 16
    VOCAB = 8
    POINTS = 320
    IGNORE_ID = -1

    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self.ignore_id = self.IGNORE_ID
        self._enc = nn.Linear(self.ENCODER_DIM, self.ENCODER_DIM)
        self.ctc = SimpleNamespace(
            ctc_lo=nn.Linear(self.ENCODER_DIM, self.VOCAB),
        )

    def encode(self, speech, lengths):
        B = speech.shape[0]
        S = max(1, int(lengths.max()) // self.POINTS)
        features = self._enc(torch.randn(B, S, self.ENCODER_DIM))
        logit_lens = (lengths // self.POINTS).clamp(min=1).long()
        return features, logit_lens

    def encoder_output_size(self):
        return self.ENCODER_DIM

    def points_by_frames(self):
        return self.POINTS

    def get_blank_id(self):
        return 0

    def _calc_ctc_loss(self, encoder_out, encoder_out_lens, text,
                       text_lengths, **kwargs):
        logits = self.ctc.ctc_lo(encoder_out)
        log_probs = nn.functional.log_softmax(logits, dim=-1).permute(
            1, 0, 2,
        )
        text_clean = torch.where(text == -1, 0, text)
        loss = nn.functional.ctc_loss(
            log_probs, text_clean,
            encoder_out_lens, text_lengths,
            blank=0, reduction="mean",
        )
        return loss, {"loss_ctc": loss.detach()}


# -------------------------------------------------------------------
# DUMMY DATASETS
# -------------------------------------------------------------------

POINTS = DummyNet.POINTS
VOCAB = DummyNet.VOCAB


class DummySegDataset:
    """Mimics SegmentationDataset.__getitem__ output."""

    def __init__(self, n=10):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        T_speech = 6400
        T_phones = 4
        return {
            "speech": torch.randn(T_speech),
            "speech_length": T_speech,
            "target": torch.randint(1, VOCAB, (T_phones,)),
            "target_length": T_phones,
            "phone_pointstamps": [
                (i * POINTS, (i + 1) * POINTS - 1)
                for i in range(T_phones)
            ],
            "utt_id": f"seg_{idx}",
            "phones": ["a", "b", "c", "d"],
        }


class DummyPRDataset:
    """Mimics KaldiDataset.__getitem__ output."""

    def __init__(self, n=20):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        T_speech = 4800
        T_phones = 3
        return {
            "speech": torch.randn(T_speech),
            "speech_length": T_speech,
            "text_tokens": list(range(1, T_phones + 1)),
            "text": "a/b/c",
            "lang_sym": "<eng>",
            "key": f"pr_{idx}",
            "utt_id": f"pr_{idx}",
        }


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _make_model(**kwargs):
    defaults = dict(net=DummyNet(), optimizer=torch.optim.Adam, bce_weight=1.0)
    defaults.update(kwargs)
    return JointPRSegModel(**defaults)


def _make_seg_sub_batch(B=2, T_speech=6400, T_phones=4):
    return {
        "speech": torch.randn(B, T_speech),
        "speech_length": torch.full((B,), T_speech, dtype=torch.long),
        "text": torch.randint(1, VOCAB, (B, T_phones)),
        "text_length": torch.full((B,), T_phones, dtype=torch.long),
        "target_start": torch.stack(
            [torch.arange(T_phones, dtype=torch.float32) * POINTS] * B
        ),
        "target_end": torch.stack(
            [torch.arange(T_phones, dtype=torch.float32) * POINTS
             + POINTS - 1] * B
        ),
        "utt_id": [f"seg_{i}" for i in range(B)],
    }


def _make_pr_sub_batch(B=2, T_speech=4800, T_phones=3):
    return {
        "speech": torch.randn(B, T_speech),
        "speech_length": torch.full((B,), T_speech, dtype=torch.long),
        "text": torch.randint(1, VOCAB, (B, T_phones)),
        "text_length": torch.full((B,), T_phones, dtype=torch.long),
        "lang_sym": ["<eng>"] * B,
        "utt_id": [f"pr_{i}" for i in range(B)],
    }


# -------------------------------------------------------------------
# DATASET TESTS
# -------------------------------------------------------------------


def test_joint_dataset_length_weight_1():
    ds = JointPRSegmentDataset([
        (DummySegDataset(10), "segmentation", 1.0),
        (DummyPRDataset(20), "recognition", 1.0),
    ])
    assert len(ds) == 30


def test_joint_dataset_length_weight_2():
    ds = JointPRSegmentDataset([
        (DummySegDataset(10), "segmentation", 2.0),
        (DummyPRDataset(20), "recognition", 0.5),
    ])
    assert len(ds) == 20 + 10  # 10*2 + 20*0.5


def test_joint_dataset_getitem_seg():
    ds = JointPRSegmentDataset([
        (DummySegDataset(5), "segmentation", 1.0),
    ])
    item = ds[0]
    assert item["type"] == "segmentation"
    assert "speech" in item
    assert "text" in item
    assert "phone_pointstamps" in item
    assert isinstance(item["text"], torch.Tensor)


def test_joint_dataset_getitem_pr():
    ds = JointPRSegmentDataset([
        (DummyPRDataset(5), "recognition", 1.0),
    ])
    item = ds[0]
    assert item["type"] == "recognition"
    assert "speech" in item
    assert "text" in item
    assert "lang_sym" in item
    assert isinstance(item["text"], torch.Tensor)


def test_joint_dataset_weight_wrapping():
    """Weight > 1.0 wraps indices via modulo."""
    ds = JointPRSegmentDataset([
        (DummySegDataset(3), "segmentation", 2.0),
    ])
    assert len(ds) == 6
    # Items 0,1,2 map to ds[0,1,2]; items 3,4,5 map to ds[0,1,2]
    assert ds[0]["utt_id"] == ds[3]["utt_id"]


# -------------------------------------------------------------------
# COLLATE TESTS
# -------------------------------------------------------------------


def test_collate_mixed_batch():
    ds = JointPRSegmentDataset([
        (DummySegDataset(2), "segmentation", 1.0),
        (DummyPRDataset(2), "recognition", 1.0),
    ])
    items = [ds[i] for i in range(4)]
    batch = joint_prseg_collate_fn(items)
    assert batch["segmentation"] is not None
    assert batch["recognition"] is not None
    assert batch["segmentation"]["speech"].shape[0] == 2
    assert batch["recognition"]["speech"].shape[0] == 2


def test_collate_seg_only():
    ds = JointPRSegmentDataset([
        (DummySegDataset(3), "segmentation", 1.0),
    ])
    items = [ds[i] for i in range(3)]
    batch = joint_prseg_collate_fn(items)
    assert batch["segmentation"] is not None
    assert batch["recognition"] is None
    assert "target_start" in batch["segmentation"]


def test_collate_pr_only():
    ds = JointPRSegmentDataset([
        (DummyPRDataset(3), "recognition", 1.0),
    ])
    items = [ds[i] for i in range(3)]
    batch = joint_prseg_collate_fn(items)
    assert batch["segmentation"] is None
    assert batch["recognition"] is not None
    assert "lang_sym" in batch["recognition"]


# -------------------------------------------------------------------
# MODEL TESTS
# -------------------------------------------------------------------


def test_training_step_pr_only():
    model = _make_model()
    batch = {"segmentation": None, "recognition": _make_pr_sub_batch()}
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_seg_only():
    model = _make_model()
    batch = {"segmentation": _make_seg_sub_batch(), "recognition": None}
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_mixed():
    model = _make_model()
    batch = {
        "segmentation": _make_seg_sub_batch(),
        "recognition": _make_pr_sub_batch(),
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_backward_mixed():
    """Gradients flow through shared encoder from both tasks."""
    model = _make_model()
    batch = {
        "segmentation": _make_seg_sub_batch(),
        "recognition": _make_pr_sub_batch(),
    }
    loss = model.training_step(batch, 0)
    loss.backward()
    enc_grad = model.net._enc.weight.grad
    assert enc_grad is not None
    assert enc_grad.abs().sum() > 0
    bh_grad = model.boundary_head.weight.grad
    assert bh_grad is not None
    assert bh_grad.abs().sum() > 0


def test_normalize_seg_item():
    raw = DummySegDataset(1)[0]
    item = _normalize_seg_item(raw)
    assert item["type"] == "segmentation"
    assert "text" in item
    assert "target" not in item
    assert isinstance(item["text"], torch.Tensor)


def test_normalize_pr_item():
    raw = DummyPRDataset(1)[0]
    item = _normalize_pr_item(raw)
    assert item["type"] == "recognition"
    assert "text" in item
    assert isinstance(item["text"], torch.Tensor)
    assert item["text_length"] == 3


def test_on_validation_epoch_end_no_seg_updates():
    """on_validation_epoch_end does not crash when val_seg_loss was
    never updated -- i.e. no segmentation batches in the epoch
    (bug M39)."""
    model = _make_model()
    # val_seg_loss has not been updated: mean_value=0, weight=0.
    # The method should early-return instead of computing NaN.
    model.on_validation_epoch_end()
    # If we get here without error, the guard clause works.
    # Also verify the metric was not accidentally updated.
    assert model.val_seg_loss.weight.item() == 0
