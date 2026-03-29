"""Combined dataset and datamodule for joint phone recognition + segmentation.

``JointPRSegmentDataset`` interleaves items from multiple datasets with
weight-controlled sampling.  The collate function groups items by type
into sub-batches keyed by ``"segmentation"`` and ``"recognition"``.

Batch format returned by ``joint_prseg_collate_fn``::

    {
        "segmentation": {  # or None when no seg items in batch
            "speech":       (B_seg, T) float32,
            "speech_length": (B_seg,) int64,
            "text":         (B_seg, L) int64,
            "text_length":  (B_seg,) int64,
            "target_start": (B_seg, L) float32,
            "target_end":   (B_seg, L) float32,
        },
        "recognition": {   # or None when no PR items in batch
            "speech":       (B_pr, T) float32,
            "speech_length": (B_pr,) int64,
            "text":         (B_pr, L) int64,
            "text_length":  (B_pr,) int64,
            "lang_sym":     list[str],
        },
    }
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import lightning as L
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader, Dataset


# ── Per-item normalization ───────────────────────────────────────────


def _normalize_seg_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a segmentation dataset item to common format.

    Maps ``target`` → ``text``, ``target_length`` → ``text_length``,
    keeps ``phone_pointstamps`` for collation.
    """
    return {
        "type": "segmentation",
        "speech": item["speech"],
        "speech_length": item["speech_length"],
        "text": item["target"],
        "text_length": item["target_length"],
        "phone_pointstamps": item["phone_pointstamps"],
        "utt_id": item.get("utt_id", ""),
    }


def _normalize_pr_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a phone recognition dataset item to common format.

    Converts ``text_tokens`` list to a tensor as ``text``.
    """
    tokens = item.get("text_tokens")
    if tokens is not None:
        text = torch.tensor(tokens, dtype=torch.long)
        text_length = len(tokens)
    else:
        text = torch.tensor([], dtype=torch.long)
        text_length = 0
    return {
        "type": "recognition",
        "speech": item["speech"],
        "speech_length": item["speech_length"],
        "text": text,
        "text_length": text_length,
        "lang_sym": item.get("lang_sym", "<unk>"),
        "utt_id": item.get("utt_id", item.get("key", "")),
    }


_NORMALIZERS = {
    "segmentation": _normalize_seg_item,
    "recognition": _normalize_pr_item,
}


# ── Joint dataset ────────────────────────────────────────────────────


class JointPRSegmentDataset(Dataset):
    """Unified dataset interleaving seg and PR items with weighted sampling.

    Args:
        datasets: List of ``(dataset, type, weight)`` tuples.
            ``type`` is ``"segmentation"`` or ``"recognition"``.
            ``weight`` scales effective sample count (e.g. 2.0 = 2x items).
    """

    def __init__(
        self,
        datasets: List[Tuple[Dataset, str, float]],
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.index_map: List[Tuple[int, int]] = []
        self._build_index_map()

    def _build_index_map(self) -> None:
        for ds_idx, (ds, ds_type, weight) in enumerate(self.datasets):
            assert ds_type in ("segmentation", "recognition"), (
                f"Unknown type {ds_type!r}"
            )
            effective_len = int(len(ds) * weight)
            ds_len = len(ds)
            for i in range(effective_len):
                self.index_map.append((ds_idx, i % ds_len))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, local_idx = self.index_map[idx]
        ds, ds_type, _ = self.datasets[ds_idx]
        item = ds[local_idx]
        return _NORMALIZERS[ds_type](item)


# ── Collate ──────────────────────────────────────────────────────────


def _collate_seg_items(items: List[Dict]) -> Dict[str, Any]:
    """Collate segmentation items into a padded sub-batch."""
    B = len(items)
    T = max(item["speech_length"] for item in items)
    L = max(item["text"].shape[0] for item in items)

    speech = torch.zeros(B, T, dtype=torch.float32)
    speech_length = torch.zeros(B, dtype=torch.long)
    text = torch.full((B, L), -1, dtype=torch.long)
    text_length = torch.zeros(B, dtype=torch.long)
    target_start = torch.full((B, L), -1, dtype=torch.float32)
    target_end = torch.full((B, L), -1, dtype=torch.float32)

    for i, item in enumerate(items):
        sl = item["speech_length"]
        pl = item["text"].shape[0]
        speech[i, :sl] = item["speech"]
        speech_length[i] = sl
        text[i, :pl] = item["text"]
        text_length[i] = pl
        for j, (s, e) in enumerate(item["phone_pointstamps"][:pl]):
            target_start[i, j] = s
            target_end[i, j] = e

    return {
        "speech": speech,
        "speech_length": speech_length,
        "text": text,
        "text_length": text_length,
        "target_start": target_start,
        "target_end": target_end,
        "utt_id": [item["utt_id"] for item in items],
    }


def _collate_pr_items(items: List[Dict]) -> Dict[str, Any]:
    """Collate phone recognition items into a padded sub-batch."""
    B = len(items)
    speech_lengths = torch.tensor(
        [item["speech_length"] for item in items], dtype=torch.long,
    )
    max_T = int(speech_lengths.max())

    speech = torch.zeros(B, max_T, dtype=torch.float32)
    for i, item in enumerate(items):
        sl = item["speech_length"]
        speech[i, :sl] = item["speech"]

    texts = [item["text"] for item in items]
    max_L = max((t.shape[0] for t in texts), default=1) or 1
    text = torch.full((B, max_L), -1, dtype=torch.long)
    text_length = torch.zeros(B, dtype=torch.long)
    for i, t in enumerate(texts):
        if t.numel() > 0:
            text[i, : t.shape[0]] = t
            text_length[i] = t.shape[0]

    return {
        "speech": speech,
        "speech_length": speech_lengths,
        "text": text,
        "text_length": text_length,
        "lang_sym": [item.get("lang_sym", "<unk>") for item in items],
        "utt_id": [item["utt_id"] for item in items],
    }


def joint_prseg_collate_fn(
    batch: List[Dict],
) -> Dict[str, Optional[Dict]]:
    """Group items by type, collate each group independently.

    Returns dict with ``"segmentation"`` and ``"recognition"`` keys,
    each mapping to a padded sub-batch or ``None``.
    """
    seg_items = [x for x in batch if x["type"] == "segmentation"]
    pr_items = [x for x in batch if x["type"] == "recognition"]
    return {
        "segmentation": _collate_seg_items(seg_items) if seg_items else None,
        "recognition": _collate_pr_items(pr_items) if pr_items else None,
    }


# ── DataModule ───────────────────────────────────────────────────────


class JointPRSegDataModule(L.LightningDataModule):
    """Compose PR and Seg datamodules for joint multi-task training.

    Training uses ``JointPRSegmentDataset`` (single DataLoader with
    mixed items).  Validation uses ``CombinedLoader(sequential)``
    over each sub-datamodule's own validation loader.

    Args:
        pr_datamodule: Phone recognition datamodule (Kaldi-style).
        seg_datamodule: Segmentation datamodule (HF-style).
        pr_weight: Sampling weight for PR data (scales effective count).
        seg_weight: Sampling weight for Seg data.
        batch_size: Batch size for the joint training DataLoader.
        num_workers: Number of workers for the joint DataLoader.
        pin_memory: Pin memory for the joint DataLoader.
    """

    def __init__(
        self,
        pr_datamodule: L.LightningDataModule,
        seg_datamodule: L.LightningDataModule,
        pr_weight: float = 1.0,
        seg_weight: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.pr = pr_datamodule
        self.seg = seg_datamodule
        self.pr_weight = pr_weight
        self.seg_weight = seg_weight
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        self.pr.setup(stage)
        self.seg.setup(stage)
        datasets_list: List[Tuple[Dataset, str, float]] = []
        pr_train = getattr(
            self.pr, f"{self.pr.train_split}_dataset", None,
        )
        if pr_train is not None:
            datasets_list.append(
                (pr_train, "recognition", self.pr_weight),
            )
        seg_train = getattr(self.seg, "train_dataset", None)
        if seg_train is not None:
            datasets_list.append(
                (seg_train, "segmentation", self.seg_weight),
            )
        assert datasets_list, "No training datasets found."
        self.train_dataset = JointPRSegmentDataset(datasets_list)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=joint_prseg_collate_fn,
        )

    def val_dataloader(self):
        return self.seg.val_dataloader()

    def test_dataloader(self):
        seg_test = self.seg.test_dataloader()
        if seg_test is not None:
            return seg_test
        return self.pr.test_dataloader()

    def predict_dataloader(self):
        return self.seg.predict_dataloader()
