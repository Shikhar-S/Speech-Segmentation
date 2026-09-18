"""Unit tests for src.data.mixed_prseg."""

import pytest
import torch
from torch.utils.data import Dataset

from src.data.mixed_prseg_dataset import (
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
    """Raw segmentation item as emitted by SegmentationDataset."""
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
    """Raw PR item where target is a token-id tensor (compatible format)."""
    return {
        "speech": torch.zeros(speech_len, dtype=torch.float32),
        "speech_length": speech_len,
        "target": torch.arange(n, dtype=torch.long),
        "lang_sym": "<eng>",
        "utt_id": utt_id,
    }


# -- Normalization ----------------------------------------------


def test_normalize_seg_item_keys_and_type():
    item = _raw_seg_item(n=3)
    norm = _normalize_seg_item(item)
    assert norm["type"] == "segmentation"
    assert set(norm.keys()) >= {
        "type", "speech", "speech_length", "target", "target_length",
        "target_start", "target_end", "lang_sym", "utt_id",
    }


def test_normalize_seg_item_pointstamps_to_start_end():
    item = _raw_seg_item(n=3)
    norm = _normalize_seg_item(item)
    assert norm["target_start"] == [0.0, 160.0, 320.0]
    assert norm["target_end"] == [160.0, 320.0, 480.0]


def test_normalize_seg_item_target_passthrough():
    item = _raw_seg_item(n=4)
    norm = _normalize_seg_item(item)
    assert norm["target"].tolist() == [0, 1, 2, 3]
    assert norm["target_length"] == 4


def test_normalize_seg_item_default_lang_sym():
    item = _raw_seg_item()
    del item["lang_sym"]
    norm = _normalize_seg_item(item)
    assert norm["lang_sym"] == "<eng>"


def test_normalize_seg_item_default_utt_id():
    item = _raw_seg_item()
    del item["utt_id"]
    norm = _normalize_seg_item(item)
    assert norm["utt_id"] == ""


def test_normalize_pr_item_keys_and_type():
    item = _raw_pr_item(n=3)
    norm = _normalize_pr_item(item)
    assert norm["type"] == "recognition"
    assert set(norm.keys()) >= {
        "type", "speech", "speech_length", "target", "target_length",
        "lang_sym", "utt_id",
    }


def test_normalize_pr_item_target_length():
    item = _raw_pr_item(n=5)
    norm = _normalize_pr_item(item)
    assert norm["target_length"] == 5


def test_normalize_pr_item_passthrough_target():
    item = _raw_pr_item(n=3)
    norm = _normalize_pr_item(item)
    # target is passed through unchanged (tensor or list)
    assert list(norm["target"]) == [0, 1, 2]


def test_normalize_pr_item_missing_target_raises():
    item = {"speech": torch.zeros(100), "speech_length": 100}
    with pytest.raises(AssertionError):
        _normalize_pr_item(item)


def test_normalize_pr_item_utt_id_fallback_to_key():
    item = _raw_pr_item()
    del item["utt_id"]
    item["key"] = "mykey"
    norm = _normalize_pr_item(item)
    assert norm["utt_id"] == "mykey"


def test_normalize_pr_item_default_lang_sym():
    item = _raw_pr_item()
    del item["lang_sym"]
    norm = _normalize_pr_item(item)
    assert norm["lang_sym"] == "<unk>"


# -- Collate: segmentation --------------------------------------


def test_collate_seg_items_shapes():
    items = [
        _normalize_seg_item(_raw_seg_item(utt_id="a", n=3, speech_len=1600)),
        _normalize_seg_item(_raw_seg_item(utt_id="b", n=2, speech_len=800)),
    ]
    out = _collate_seg_items(items)
    assert out["speech"].shape == (2, 1600)
    assert out["target"].shape == (2, 3)
    assert out["target_length"].tolist() == [3, 2]


def test_collate_seg_items_padding_value():
    items = [
        _normalize_seg_item(_raw_seg_item(utt_id="a", n=3, speech_len=1600)),
        _normalize_seg_item(_raw_seg_item(utt_id="b", n=2, speech_len=800)),
    ]
    out = _collate_seg_items(items)
    # Shorter item's last token and timestamps are padded with -1
    assert out["target"][1, 2].item() == -1
    assert out["target_start"][1, 2].item() == -1.0
    assert out["target_end"][1, 2].item() == -1.0


def test_collate_seg_items_speech_padding():
    items = [
        _normalize_seg_item(_raw_seg_item(speech_len=1600)),
        _normalize_seg_item(_raw_seg_item(speech_len=800)),
    ]
    out = _collate_seg_items(items)
    assert out["speech"][1, 800:].sum().item() == 0.0


def test_collate_seg_items_timestamps_correct():
    item = _raw_seg_item(n=2)
    norm = _normalize_seg_item(item)
    out = _collate_seg_items([norm])
    assert out["target_start"][0, 0].item() == pytest.approx(0.0)
    assert out["target_end"][0, 0].item() == pytest.approx(160.0)
    assert out["target_start"][0, 1].item() == pytest.approx(160.0)
    assert out["target_end"][0, 1].item() == pytest.approx(320.0)


def test_collate_seg_items_utt_id():
    items = [
        _normalize_seg_item(_raw_seg_item(utt_id="x")),
        _normalize_seg_item(_raw_seg_item(utt_id="y")),
    ]
    out = _collate_seg_items(items)
    assert out["utt_id"] == ["x", "y"]


# -- Collate: recognition ---------------------------------------


def test_collate_pr_items_shapes():
    items = [
        _normalize_pr_item(_raw_pr_item(utt_id="a", n=3, speech_len=1600)),
        _normalize_pr_item(_raw_pr_item(utt_id="b", n=1, speech_len=800)),
    ]
    out = _collate_pr_items(items)
    assert out["speech"].shape == (2, 1600)
    assert out["target"].shape == (2, 3)
    assert out["target_length"].tolist() == [3, 1]


def test_collate_pr_items_padding():
    items = [
        _normalize_pr_item(_raw_pr_item(utt_id="a", n=3)),
        _normalize_pr_item(_raw_pr_item(utt_id="b", n=1)),
    ]
    out = _collate_pr_items(items)
    assert out["target"][1, 1].item() == -1
    assert out["target"][1, 2].item() == -1


def test_collate_pr_items_lang_sym():
    items = [
        _normalize_pr_item(_raw_pr_item(utt_id="a")),
        _normalize_pr_item(_raw_pr_item(utt_id="b")),
    ]
    out = _collate_pr_items(items)
    assert out["lang_sym"] == ["<eng>", "<eng>"]


# -- collate_pr_seg_items ---------------------------------------


def test_joint_collate_splits_by_type():
    batch = [
        _normalize_seg_item(_raw_seg_item(utt_id="s", n=3)),
        _normalize_pr_item(_raw_pr_item(utt_id="p", n=2)),
    ]
    out = collate_pr_seg_items(batch)
    assert out["segmentation"] is not None
    assert out["recognition"] is not None
    assert out["segmentation"]["utt_id"] == ["s"]
    assert out["recognition"]["utt_id"] == ["p"]


def test_joint_collate_seg_only():
    batch = [_normalize_seg_item(_raw_seg_item(utt_id="s"))]
    out = collate_pr_seg_items(batch)
    assert out["segmentation"] is not None
    assert out["recognition"] is None


def test_joint_collate_pr_only():
    batch = [_normalize_pr_item(_raw_pr_item(utt_id="p"))]
    out = collate_pr_seg_items(batch)
    assert out["segmentation"] is None
    assert out["recognition"] is not None


def test_joint_collate_always_has_both_keys():
    batch = [_normalize_pr_item(_raw_pr_item())]
    out = collate_pr_seg_items(batch)
    assert "segmentation" in out
    assert "recognition" in out


# -- MixedPRSegmentDataset --------------------------------------


def test_mixed_dataset_len_with_unit_weight():
    ds_seg = _ListDataset([_raw_seg_item(f"s{i}") for i in range(4)])
    ds_pr = _ListDataset([_raw_pr_item(f"p{i}") for i in range(3)])
    joint = MixedPRSegmentDataset(
        [(ds_seg, "segmentation", 1.0), (ds_pr, "recognition", 1.0)]
    )
    assert len(joint) == 7


def test_mixed_dataset_weight_scales_effective_length():
    ds = _ListDataset([_raw_seg_item(f"a{i}") for i in range(4)])
    joint = MixedPRSegmentDataset([(ds, "segmentation", 2.5)])
    # int(4 * 2.5) = 10
    assert len(joint) == 10


def test_mixed_dataset_weight_wraps_with_modulo():
    ds = _ListDataset([_raw_seg_item("a"), _raw_seg_item("b")])
    joint = MixedPRSegmentDataset([(ds, "segmentation", 3.0)])
    # 6 items from 2-item ds: indices 0,1,0,1,0,1
    assert len(joint) == 6
    # Item at index 4 should wrap to ds[0]
    assert joint[4]["utt_id"] == "a"
    assert joint[5]["utt_id"] == "b"


def test_mixed_dataset_normalizes_items():
    ds_seg = _ListDataset([_raw_seg_item("s")])
    ds_pr = _ListDataset([_raw_pr_item("p")])
    joint = MixedPRSegmentDataset(
        [(ds_seg, "segmentation", 1.0), (ds_pr, "recognition", 1.0)]
    )
    seg_item = joint[0]
    pr_item = joint[1]
    assert seg_item["type"] == "segmentation"
    assert pr_item["type"] == "recognition"


def test_mixed_dataset_rejects_unknown_type():
    ds = _ListDataset([_raw_seg_item()])
    with pytest.raises(AssertionError, match="Unknown type"):
        MixedPRSegmentDataset([(ds, "bogus", 1.0)])


def test_mixed_dataset_zero_weight_empty():
    ds = _ListDataset([_raw_seg_item(f"a{i}") for i in range(5)])
    joint = MixedPRSegmentDataset([(ds, "segmentation", 0.0)])
    assert len(joint) == 0


# -- SegmentRecognizeDataModule ---------------------------------


def _make_dm(
    train_datasets,
    validation_datasets,
    test_datasets=None,
):
    return SegmentRecognizeDataModule(
        train_datasets=train_datasets,
        validation_datasets=validation_datasets,
        test_datasets=test_datasets,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )


def _seg_tuples(n=4, weight=1.0):
    ds = _ListDataset([_raw_seg_item(f"s{i}") for i in range(n)])
    return [(ds, "segmentation", weight)]


def _pr_tuples(n=4, weight=1.0):
    ds = _ListDataset([_raw_pr_item(f"p{i}") for i in range(n)])
    return [(ds, "recognition", weight)]


def test_datamodule_setup_creates_train_dataset():
    dm = _make_dm(_seg_tuples(), _seg_tuples(n=2))
    dm.setup()
    assert hasattr(dm, "train_ds")
    assert len(dm.train_ds) == 4


def test_datamodule_setup_creates_val_dataset():
    dm = _make_dm(_seg_tuples(), _seg_tuples(n=2))
    dm.setup()
    assert hasattr(dm, "validation_ds")
    assert len(dm.validation_ds) == 2


def test_datamodule_setup_no_test_datasets():
    dm = _make_dm(_seg_tuples(), _seg_tuples(n=2))
    dm.setup()
    assert dm.test_ds is None


def test_datamodule_train_dataloader_runs():
    dm = _make_dm(_seg_tuples() + _pr_tuples(), _seg_tuples(n=2))
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "segmentation" in batch or "recognition" in batch


def test_datamodule_val_dataloader_canonical_keys():
    dm = _make_dm(_seg_tuples(), _seg_tuples(n=2))
    dm.setup()
    batch = next(iter(dm.val_dataloader()))
    # Val uses seg items; collate wraps in segmentation/recognition dict
    assert "segmentation" in batch or "target" in batch


def test_datamodule_test_dataloader_none_when_no_test():
    dm = _make_dm(_seg_tuples(), _seg_tuples(n=2))
    dm.setup()
    assert dm.test_dataloader() is None


def test_datamodule_test_dataloader_runs_when_provided():
    dm = _make_dm(
        _seg_tuples(),
        _seg_tuples(n=2),
        test_datasets=_seg_tuples(n=3),
    )
    dm.setup()
    loader = dm.test_dataloader()
    assert loader is not None
    batch = next(iter(loader))
    assert batch is not None


def test_datamodule_predict_dataloader_covers_val_and_test():
    train = _seg_tuples(n=4)
    val = _seg_tuples(n=2)
    test = _seg_tuples(n=3)
    dm = _make_dm(train, val, test_datasets=test)
    dm.setup()
    loader = dm.predict_dataloader()
    # val (2) + test (3) = 5 items total across predict loader
    total = sum(1 for _ in loader.dataset)
    assert total == 5


def test_datamodule_predict_dataloader_val_only_when_no_test():
    dm = _make_dm(_seg_tuples(), _seg_tuples(n=2))
    dm.setup()
    loader = dm.predict_dataloader()
    assert sum(1 for _ in loader.dataset) == 2


# -- build_prseg_datamodule -------------------------------------


def test_build_prseg_datamodule_returns_dm():
    ds_seg = _ListDataset([_raw_seg_item(f"s{i}") for i in range(4)])
    ds_pr = _ListDataset([_raw_pr_item(f"p{i}") for i in range(3)])
    dm = build_prseg_datamodule(
        train_datasets={
            "segmentation": [(ds_seg, 1.0)],
            "recognition": [(ds_pr, 0.5)],
        },
        validation_datasets={
            "segmentation": [ds_seg],
        },
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()
    # train: 4 seg + int(3*0.5)=1 pr = 5
    assert len(dm.train_ds) == 5
    # val: 4 seg items
    assert len(dm.validation_ds) == 4


def test_build_prseg_datamodule_rejects_unknown_train_type():
    ds = _ListDataset([_raw_seg_item()])
    with pytest.raises(AssertionError, match="Unknown dataset type"):
        build_prseg_datamodule(
            train_datasets={"bogus": [(ds, 1.0)]},
            validation_datasets={"segmentation": [ds]},
        )


def test_build_prseg_datamodule_rejects_unknown_val_type():
    ds = _ListDataset([_raw_seg_item()])
    with pytest.raises(AssertionError, match="Unknown dataset type"):
        build_prseg_datamodule(
            train_datasets={"segmentation": [(ds, 1.0)]},
            validation_datasets={"bogus": [ds]},
        )
