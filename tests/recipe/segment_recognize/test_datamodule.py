"""Unit tests for SegmentRecognizeDataModule and helpers (src.data.mixed_prseg)."""

import pytest
import torch
from torch.utils.data import Dataset

from src.data.mixed_prseg import (
    MixedPRSegmentDataset,
    SegmentRecognizeDataModule,
    _collate_pr_items,
    _collate_seg_items,
    _normalize_pr_item,
    _normalize_seg_item,
    build_prseg_datamodule,
    collate_pr_seg_items,
)


# -- Fixtures ---------------------------------------------------


class _ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _raw_seg_item(utt_id="u", n=3, speech_len=1600):
    return {
        "speech": torch.zeros(speech_len, dtype=torch.float32),
        "speech_length": speech_len,
        "target": torch.arange(n, dtype=torch.long),
        "target_length": n,
        "phone_pointstamps": [(i * 160, i * 160 + 160) for i in range(n)],
        "lang_sym": "<eng>",
        "utt_id": utt_id,
    }


def _raw_pr_item(utt_id="u", n=3, speech_len=1600):
    return {
        "speech": torch.zeros(speech_len, dtype=torch.float32),
        "speech_length": speech_len,
        "target": torch.arange(n, dtype=torch.long),
        "lang_sym": "<eng>",
        "utt_id": utt_id,
    }


# -- Normalization ----------------------------------------------


def test_normalize_seg_item_canonical_keys():
    item = _raw_seg_item(n=3)
    norm = _normalize_seg_item(item)
    assert norm["type"] == "segmentation"
    assert norm["target"].tolist() == [0, 1, 2]
    assert norm["target_length"] == 3
    assert norm["target_start"] == [0.0, 160.0, 320.0]
    assert norm["target_end"] == [160.0, 320.0, 480.0]


def test_normalize_pr_item_canonical_keys():
    item = _raw_pr_item(n=4)
    norm = _normalize_pr_item(item)
    assert norm["type"] == "recognition"
    assert list(norm["target"]) == [0, 1, 2, 3]
    assert norm["target_length"] == 4
    assert norm["lang_sym"] == "<eng>"


def test_normalize_pr_item_missing_target_raises():
    """Missing ``target`` raises AssertionError."""
    item = {"speech": torch.zeros(100), "speech_length": 100}
    with pytest.raises(AssertionError):
        _normalize_pr_item(item)


# -- Collate ----------------------------------------------------


def test_collate_seg_items_shapes():
    items = [
        _normalize_seg_item(_raw_seg_item(utt_id="a", n=3, speech_len=1600)),
        _normalize_seg_item(_raw_seg_item(utt_id="b", n=2, speech_len=800)),
    ]
    out = _collate_seg_items(items)
    assert out["speech"].shape == (2, 1600)
    assert out["target"].shape == (2, 3)
    assert out["target_length"].tolist() == [3, 2]
    assert out["target"][1, 2].item() == -1
    assert out["target_start"][1, 2].item() == -1.0
    assert out["utt_id"] == ["a", "b"]


def test_collate_pr_items_shapes_and_padding():
    items = [
        _normalize_pr_item(_raw_pr_item(utt_id="a", n=3, speech_len=1600)),
        _normalize_pr_item(_raw_pr_item(utt_id="b", n=1, speech_len=800)),
    ]
    out = _collate_pr_items(items)
    assert out["speech"].shape == (2, 1600)
    assert out["target"].shape == (2, 3)
    assert out["target_length"].tolist() == [3, 1]
    assert out["target"][1, 1].item() == -1
    assert out["target"][1, 2].item() == -1


def test_joint_collate_splits_by_type():
    """Mixed items split into seg/recognition sub-batches."""
    batch = [
        _normalize_seg_item(_raw_seg_item(utt_id="s", n=3)),
        _normalize_pr_item(_raw_pr_item(utt_id="p", n=2)),
    ]
    out = collate_pr_seg_items(batch)
    assert out["segmentation"] is not None
    assert out["recognition"] is not None
    assert out["segmentation"]["utt_id"] == ["s"]
    assert out["recognition"]["utt_id"] == ["p"]


def test_joint_collate_single_type_nones_other():
    batch = [_normalize_pr_item(_raw_pr_item(utt_id="p", n=2))]
    out = collate_pr_seg_items(batch)
    assert out["segmentation"] is None
    assert out["recognition"] is not None


# -- MixedPRSegmentDataset --------------------------------------


def test_joint_dataset_index_map_weights():
    """Weight scales effective length."""
    ds_a = _ListDataset([_raw_seg_item(f"a{i}") for i in range(4)])
    ds_b = _ListDataset([_raw_pr_item(f"b{i}") for i in range(2)])
    joint = MixedPRSegmentDataset(
        [(ds_a, "segmentation", 1.0), (ds_b, "recognition", 2.5)],
    )
    # 4 seg + int(2 * 2.5) = 4 + 5 = 9
    assert len(joint) == 9
    assert joint[0]["type"] == "segmentation"
    assert joint[8]["type"] == "recognition"


def test_joint_dataset_rejects_unknown_type():
    ds = _ListDataset([_raw_seg_item("a")])
    with pytest.raises(AssertionError, match="Unknown type"):
        MixedPRSegmentDataset([(ds, "bogus", 1.0)])


# -- SegmentRecognizeDataModule ---------------------------------


def _make_dm(train_ds_list, val_ds_list, test_ds_list=None):
    return SegmentRecognizeDataModule(
        train_datasets=train_ds_list,
        validation_datasets=val_ds_list,
        test_datasets=test_ds_list,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )


def _seg_ds_list(n=4, weight=1.0):
    ds = _ListDataset([_raw_seg_item(f"s{i}") for i in range(n)])
    return [(ds, "segmentation", weight)]


def test_datamodule_setup_and_lengths():
    dm = _make_dm(_seg_ds_list(n=4), _seg_ds_list(n=2))
    dm.setup()
    assert len(dm.train_ds) == 4
    assert len(dm.validation_ds) == 2
    assert dm.test_ds is None


def test_datamodule_predict_dataloader_covers_val_and_test():
    dm = _make_dm(_seg_ds_list(n=4), _seg_ds_list(n=2), _seg_ds_list(n=3))
    dm.setup()
    loader = dm.predict_dataloader()
    assert sum(1 for _ in loader.dataset) == 5


def test_datamodule_predict_dataloader_val_only_when_no_test():
    dm = _make_dm(_seg_ds_list(n=4), _seg_ds_list(n=2))
    dm.setup()
    loader = dm.predict_dataloader()
    assert sum(1 for _ in loader.dataset) == 2


def test_val_dataloader_emits_canonical_batch():
    """End-to-end: val loader yields batches with expected keys."""
    dm = _make_dm(_seg_ds_list(), _seg_ds_list(n=2))
    dm.setup()
    batch = next(iter(dm.val_dataloader()))
    seg = batch.get("segmentation")
    assert seg is not None
    assert set(seg.keys()) >= {
        "speech", "speech_length", "target", "target_length",
        "target_start", "target_end",
    }
    assert seg["target_length"].shape[0] == 2


# -- build_prseg_datamodule -------------------------------------


def test_build_prseg_datamodule_train_lengths():
    ds_seg = _ListDataset([_raw_seg_item(f"s{i}") for i in range(4)])
    ds_pr = _ListDataset([_raw_pr_item(f"p{i}") for i in range(3)])
    dm = build_prseg_datamodule(
        train_datasets={
            "segmentation": [(ds_seg, 1.0)],
            "recognition": [(ds_pr, 0.5)],
        },
        validation_datasets={"segmentation": [ds_seg]},
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()
    # 4 seg + int(3 * 0.5) = 5 train items
    assert len(dm.train_ds) == 5
    assert len(dm.validation_ds) == 4
