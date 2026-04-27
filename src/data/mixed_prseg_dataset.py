"""Joint PR + segmentation datamodule.

Batch schema:

    {
        "segmentation": {                # or None
            "speech":        (B, T) float32,   # raw waveform
            "speech_length": (B,)   long,
            "target":        (B, L) long,       # phone id sequence
            "target_length": (B,)   long,
            "target_start":   (B, L) float32,    # onset in audio points
            "target_end":     (B, L) float32,    # offset in audio points
            "lang_sym":      list[str],
            "utt_id":        list[str],
        },
        "recognition": {                 # or None
            "speech":        (B, T) float32,
            "speech_length": (B,)   long,
            "target":        (B, L) long,
            "target_length": (B,)   long,
            "lang_sym":      list[str],
            "utt_id":        list[str],
        },
    }

NOTE: This schema allows mixing multiple sources for pr and seg tasks.
Usage:
    python -m src.data.mixed_prseg
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import lightning as L
from torch.utils.data import ConcatDataset, DataLoader, Dataset


#####
# NOTE(shikhar): Legacy logic, rewrite on success.


def _normalize_seg_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw segmentation item into the canonical schema."""
    pointstamps = item["phone_pointstamps"]
    target_start = [float(s) for s, _ in pointstamps]
    target_end = [float(e) for _, e in pointstamps]
    return {
        "type": "segmentation",
        "speech": item["speech"],
        "speech_length": item["speech_length"],
        
        "target": item["target"],
        "target_length": item["target_length"],
        "target_start": target_start,
        "target_end": target_end,
        
        "lang_sym": item.get("lang_sym", "<eng>"),
        "utt_id": item.get("utt_id", ""),
    }


def _normalize_pr_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw phone-recognition item into the canonical schema."""
    target = item.get("target")
    assert target is not None, "PR item missing target field"
    target = torch.as_tensor(target, dtype=torch.long)
    target_length = len(target)
    return {
        "type": "recognition",
        "speech": item["speech"],
        "speech_length": item["speech_length"],
        
        "target": target,
        "target_length": target_length,
        
        "lang_sym": item.get("lang_sym", "<unk>"),
        "utt_id": item.get("utt_id", item.get("key", "")),
    }


_NORMALIZERS = {
    "segmentation": _normalize_seg_item,
    "recognition": _normalize_pr_item,
}

######

class MixedPRSegmentDataset(Dataset):
    """Dataset to mix seg and PR items with weighted sampling.

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
        for ds_idx, (ds, ds_type, weight) in enumerate(datasets):
            assert ds_type in _NORMALIZERS, f"Unknown type {ds_type!r}"
            n = len(ds)
            for i in range(int(n * weight)):
                self.index_map.append((ds_idx, i % n))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, local_idx = self.index_map[idx]
        ds, ds_type, _ = self.datasets[ds_idx]
        return _NORMALIZERS[ds_type](ds[local_idx])


def _collate_seg_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate segmentation items into a padded sub-batch."""
    B = len(items)
    T = max(item["speech_length"] for item in items)
    L = max(item["target"].shape[0] for item in items)

    speech = torch.zeros(B, T, dtype=torch.float32)
    speech_length = torch.zeros(B, dtype=torch.long)
    target = torch.full((B, L), -1, dtype=torch.long)
    target_length = torch.zeros(B, dtype=torch.long)
    target_start = torch.full((B, L), -1.0, dtype=torch.float32)
    target_end = torch.full((B, L), -1.0, dtype=torch.float32)

    for i, item in enumerate(items):
        sl = item["speech_length"]
        pl = item["target"].shape[0]
        speech[i, :sl] = item["speech"]
        speech_length[i] = sl
        target[i, :pl] = item["target"]
        target_length[i] = pl
        starts = item["target_start"][:pl]
        ends = item["target_end"][:pl]
        target_start[i, : len(starts)] = torch.tensor(
            starts,
            dtype=torch.float32,
        )
        target_end[i, : len(ends)] = torch.tensor(
            ends,
            dtype=torch.float32,
        )

    return {
        "speech": speech,
        "speech_length": speech_length,
        "target": target,
        "target_length": target_length,
        "target_start": target_start,
        "target_end": target_end,
        "lang_sym": [item.get("lang_sym", "<eng>") for item in items],
        "utt_id": [item["utt_id"] for item in items],
    }


def _collate_pr_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate phone-recognition items into a padded sub-batch."""
    B = len(items)
    T = max(item["speech_length"] for item in items)
    L = max(int(item["target_length"]) for item in items)

    speech = torch.zeros(B, T, dtype=torch.float32)
    speech_length = torch.zeros(B, dtype=torch.long)
    target = torch.full((B, L), -1, dtype=torch.long)
    target_length = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(items):
        sl = item["speech_length"]
        pl = int(item["target_length"])
        speech[i, :sl] = item["speech"]
        speech_length[i] = sl
        target[i, :pl] = item["target"]
        target_length[i] = pl

    return {
        "speech": speech,
        "speech_length": speech_length,
        "target": target,
        "target_length": target_length,
        "lang_sym": [item.get("lang_sym", "<unk>") for item in items],
        "utt_id": [item["utt_id"] for item in items],
    }


def collate_pr_seg_items(
    batch: List[Dict[str, Any]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Wrapper to group canonical items by type and collate each group."""
    seg_items = [x for x in batch if x["type"] == "segmentation"]
    pr_items = [x for x in batch if x["type"] == "recognition"]
    return {
        "segmentation": _collate_seg_items(seg_items) if seg_items else None,
        "recognition": _collate_pr_items(pr_items) if pr_items else None,
    }


class SegmentRecognizeDataModule(L.LightningDataModule):
    """Compose PR and Seg datasets for the segment_recognize recipe.
    Args:
        train_datasets: List of (dataset, type, weight) tuples for training.
        validation_datasets: List of (dataset, type, weight) tuples for validation.
        test_datasets: List of (dataset, type, weight) tuples for testing.
        prediction_dataset: A single dataset for running inference during prediction step.
        batch_size: Batch size for the joint training DataLoader.
        num_workers: Number of workers for the joint DataLoader.
        pin_memory: Pin memory for the joint DataLoader.
    """

    def __init__(
        self,
        train_datasets: List[Tuple[Dataset, str, float]],
        validation_datasets: List[Tuple[Dataset, str, float]],
        test_datasets: Optional[List[Tuple[Dataset, str, float]]] = None,
        prediction_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_datasets = train_datasets
        self.validation_datasets = validation_datasets
        self.test_datasets = test_datasets
        self.prediction_dataset = prediction_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _ds(self, split="train"):
        dataset_list = getattr(self, f"{split}_datasets")
        return MixedPRSegmentDataset(dataset_list)

    def setup(self, stage: Optional[str] = None):
        self.train_ds = self._ds("train")
        self.validation_ds = self._ds("validation")
        self.test_ds = (
            self._ds("test") if self.test_datasets is not None else None
        )

    def _dl(self, ds, shuffle=False):
        if ds is None:
            return None
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_pr_seg_items,
        )

    def train_dataloader(self):
        return self._dl(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.validation_ds, shuffle=False)

    def test_dataloader(self):
        return self._dl(self.test_ds, shuffle=False)

    def predict_dataloader(self):
        return self._dl(self.prediction_dataset, shuffle=False)


def build_prseg_datamodule(
    train_datasets: dict[str, List[Tuple[Dataset, float]]],
    validation_datasets: dict[str, List[Dataset]],
    prediction_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> SegmentRecognizeDataModule:
    """Helper to build a SegmentRecognizeDataModule with the given components.
    Args:
        train_datasets: Dict mapping dataset type to list of (dataset, weight) tuples for training.
        validation_datasets: Dict mapping dataset type to list of datasets for validation.
        prediction_dataset: A single dataset for running inference during prediction step.
    """
    # Conversion and checks
    train_ds_list = []
    for dstype, ds_list in train_datasets.items():
        assert dstype in ("segmentation", "recognition"), f"Unknown dataset type {dstype!r}"
        for ds, weight in ds_list:
            train_ds_list.append((ds, dstype, weight))

    val_ds_list = []
    for dstype, ds_list in validation_datasets.items():
        assert dstype in ("segmentation", "recognition"), f"Unknown dataset type {dstype!r}"
        for ds in ds_list:
            val_ds_list.append((ds, dstype, 1.0))

    return SegmentRecognizeDataModule(
        train_datasets=train_ds_list,
        validation_datasets=val_ds_list,
        prediction_dataset=prediction_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


if __name__ == "__main__":
    from src.data.segmentation.segmentation_dataset import build_segmentation_dataset
    from src.data.recognition.prism_preval import build_prism_preval_dataset

    eval_pr_ds = build_prism_preval_dataset(
        dataset_name="doreco",
        data_dir="/work/hdd/bbjs/shared/powsm/s2t1/dump/raw",
        portable_wavscp=False
    )
    DATASET='changelinglab/timit-segment'
    tokenizer = type("DummyTokenizer", (), {"tokens2ids": lambda self, target: [i for i in range(len(target))]})()
    timit_train_ds=build_segmentation_dataset(hf_repo=DATASET, split="train", tokenizer=tokenizer)
    timit_val_ds=build_segmentation_dataset(hf_repo=DATASET, split="val", tokenizer=tokenizer)
    timit_test_ds=build_segmentation_dataset(hf_repo=DATASET, split="test", tokenizer=tokenizer)

    dm = build_prseg_datamodule(
        train_datasets={
            "segmentation": [(timit_train_ds, 1.0)],
            "recognition": [(timit_train_ds, 1.0)], # reuse
        },
        validation_datasets={
            "segmentation": [timit_val_ds],
            "recognition": [eval_pr_ds],
        },
        prediction_dataset=timit_test_ds,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()

    def sanity_check_dataloader(dl):
        for batch in dl:
            seg_sub = batch["segmentation"]
            if seg_sub is None:
                print('=='*20)
                print("No segmentation items in batch!!.")
                print('=='*20)
            else:
                for k in (
                    "speech",
                    "speech_length",
                    "target",
                    "target_length",
                    "target_start",
                    "target_end",
                ):
                    assert k in seg_sub, f"missing {k}"
                print("  keys:", list(seg_sub.keys()))
                print("  target shape:", seg_sub["target"].shape)
                print('Speech')
                print(seg_sub['speech'])
                print('target')
                print(seg_sub['target'])
                print('Start times')
                print(seg_sub['target_start'])
                print('End times')
                print(seg_sub['target_end'])

            pr_batch = batch["recognition"]
            if pr_batch is None:
                print('=='*20)
                print("No recognition items in batch!!.")
                print('=='*20)
                break
            for k in (
                "speech",
                "speech_length",
                "target",
                "target_length",
            ):
                assert k in pr_batch, f"missing {k}"
            print("  recognition keys:", list(pr_batch.keys()))
            print("  recognition target shape:", pr_batch["target"].shape)
            print('Recognition Speech')
            print(pr_batch['speech'])
            print('Recognition target')
            print(pr_batch['target'])
            break
    
    # print('train batch:')
    # sanity_check_dataloader(dm.train_dataloader())
    # print("val batch:")
    # sanity_check_dataloader(dm.val_dataloader())
    print("predict batch:")
    sanity_check_dataloader(dm.predict_dataloader())