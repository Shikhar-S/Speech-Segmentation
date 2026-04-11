"""Tests for src.data.segmentation.segmentation_dataset."""

from unittest.mock import MagicMock, patch

from src.data.segmentation.segmentation_dataset import SegmentationDataModule


def test_predict_split_string_normalized():
    """predict_dataloader must normalize a string predict_split to a
    single-element list so the loop iterates over split names, not
    individual characters.

    Regression test for bug M13.
    """
    tokenizer = MagicMock()
    dm = SegmentationDataModule(
        hf_repo="fake/repo",
        tokenizer=tokenizer,
        predict_split="test",
    )

    # __init__ stores the raw value; normalization is in predict_dataloader
    assert dm.predict_split == "test"

    # Simulate setup having run
    fake_ds = MagicMock()
    fake_ds.__len__ = MagicMock(return_value=3)
    dm.test_dataset = fake_ds
    dm.train_dataset = None
    dm.val_dataset = None

    with patch.object(dm, "_dl", return_value=MagicMock()) as mock_dl:
        dm.predict_dataloader()
        # Should call _dl once with a ConcatDataset, not crash
        mock_dl.assert_called_once()


def test_predict_split_list_passthrough():
    """A list predict_split must be kept as-is."""
    tokenizer = MagicMock()
    dm = SegmentationDataModule(
        hf_repo="fake/repo",
        tokenizer=tokenizer,
        predict_split=["train", "test"],
    )

    assert dm.predict_split == ["train", "test"]


def test_predict_split_none_defaults():
    """When predict_split is None, it defaults to
    [train_split, val_split, test_split].
    """
    tokenizer = MagicMock()
    dm = SegmentationDataModule(
        hf_repo="fake/repo",
        tokenizer=tokenizer,
        predict_split=None,
    )

    assert dm.predict_split == ["train", "val", "test"]


def test_predict_dataloader_string_does_not_iterate_chars():
    """End-to-end check: predict_dataloader must not try to access
    attributes like ``t_dataset``, ``e_dataset``, etc. when
    predict_split was originally a string.
    """
    tokenizer = MagicMock()
    dm = SegmentationDataModule(
        hf_repo="fake/repo",
        tokenizer=tokenizer,
        predict_split="test",
    )

    # Simulate setup having run by attaching split datasets
    fake_ds = MagicMock()
    fake_ds.__len__ = MagicMock(return_value=5)
    dm.test_dataset = fake_ds
    dm.train_dataset = None
    dm.val_dataset = None

    # Should not raise AttributeError from iterating chars
    with patch.object(dm, "_dl", return_value=MagicMock()) as mock_dl:
        dm.predict_dataloader()
        mock_dl.assert_called_once()
