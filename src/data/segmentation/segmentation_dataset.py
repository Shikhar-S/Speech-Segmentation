"""Dataloader for segmentation datasets.

The dataset on HF must follow the schema:
    utt_id, audio, text, phones, phone_starts, phone_ends,
    language, speaker_id, duration, split
The loader converts each row into a list of ``SegmentationUnit`` objects
keyed by ``utt_id``.
"""

from typing import List, Optional, Union

import datasets
import torch
import torchaudio
import lightning as L
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from src.data.segmentation.dataset_processing_transforms import (
    HF_REPO_TRANSFORMS,
)
from src.data.segmentation.dataset_splitting_transforms import (
    HF_REPO_SPLIT_TRANSFORMS,
)


class DummyTokenizer:
    """Trivial tokenizer for tasks that don't need real phone IDs.

    Returns sequential ints of the right length.
    """

    def tokens2ids(self, phones):
        return list(range(len(phones)))


class SegmentationDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        target_sr=16000,
        max_speech_length=None,
        transform=None,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        audio = row["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        if audio["sampling_rate"] != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, audio["sampling_rate"], self.target_sr
            )
        if self.max_speech_length is not None:
            waveform = waveform[: int(self.max_speech_length * self.target_sr)]

        # All per-dataset symbol normalization (e.g. ARPABET->IPA, closure
        # merging) goes through the registered transform — see
        # dataset_processing_transforms.HF_REPO_TRANSFORMS.
        phones_ipa = list(row["phones"])
        phone_timestamps = list(zip(row["phone_starts"], row["phone_ends"]))
        if self.transform is not None:
            phone_timestamps, phones_ipa = self.transform(
                phone_timestamps, phones_ipa
            )
        phone_pointstamps = [
            (int(s * self.target_sr), int(e * self.target_sr))
            for s, e in phone_timestamps
        ]
        target = self.tokenizer.tokens2ids(phones_ipa)

        return {
            "speech": waveform,
            "speech_length": len(waveform),
            "target": torch.tensor(target, dtype=torch.long),
            "target_length": len(target),
            "phone_pointstamps": phone_pointstamps,
            "phone_timestamps": phone_timestamps,
            "phones": phones_ipa,
            "text": row["text"],
            "utt_id": row["utt_id"],
            "duration": row["duration"],
            "speaker_id": row["speaker_id"],
            "language": row["language"],
            "split": row["split"],
        }


def collate_fn(batch):
    B = len(batch)
    T = max(item["speech_length"] for item in batch)
    L = max(len(item["target"]) for item in batch)

    speech = torch.full((B, T), -1.0, dtype=torch.float32)
    speech_length = torch.zeros(B, dtype=torch.long)
    phone_id = torch.full((B, L), -1, dtype=torch.long)
    target_length = torch.zeros(B, dtype=torch.long)
    target_start = torch.full((B, L), -1, dtype=torch.float32)
    target_end = torch.full((B, L), -1, dtype=torch.float32)

    for i, item in enumerate(batch):
        sl, pl = item["speech_length"], len(item["target"])
        speech[i, :sl] = item["speech"]
        speech_length[i] = sl
        phone_id[i, :pl] = item["target"]
        target_length[i] = pl
        for j, (s, e) in enumerate(item["phone_pointstamps"][:pl]):
            target_start[i, j] = s
            target_end[i, j] = e

    return {
        "speech": speech,
        "speech_length": speech_length,
        "target": phone_id,
        "target_text": [item["phones"] for item in batch],
        "target_length": target_length,
        "target_start": target_start,
        "target_end": target_end,
        "ground_truth_timestamps": [item["phone_timestamps"] for item in batch],
        "utt_id": [item["utt_id"] for item in batch],
        "split": [item.get("split", "unknown") for item in batch],
    }


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        hf_repo: str,
        tokenizer,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        tune_split: str = "tune",
        predict_split: Optional[Union[str, List[str]]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sr: int = 16000,
        max_speech_length: Optional[float] = None,
        cache_dir: str = "exp/cache/hf",
        train_fraction: float = 1.0,
        seed: int = 42,
        transform=None,  # Pass False explicitly to avoid transform
        split_transform=None,  # Pass False explicitly to avoid transform
    ):
        super().__init__()
        self.hf_repo = hf_repo
        assert (
            tokenizer is not None
        ), "Tokenizer must be provided for segmentation dataset."
        self.tokenizer = tokenizer
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.tune_split = tune_split
        self.predict_split = predict_split or [
            train_split,
            val_split,
            test_split,
        ]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length
        self.cache_dir = cache_dir
        self.train_fraction = train_fraction
        self.seed = seed
        self.transform = transform
        if transform is None:
            self.transform = HF_REPO_TRANSFORMS.get(hf_repo)
        elif transform is False:
            self.transform = None
        # transform to create a disjoint 'tune' split out of
        # train or test from hf dataset.
        self.split_transform = split_transform
        if split_transform is None:
            self.split_transform = HF_REPO_SPLIT_TRANSFORMS.get(hf_repo)
        elif split_transform is False:
            self.split_transform = None

    def _maybe_sample_subset(self, dataset, fraction):
        if fraction >= 1.0:
            return dataset
        total_size = len(dataset[self.train_split])
        subset_size = int(total_size * fraction)
        indices = torch.randperm(
            total_size, generator=torch.Generator().manual_seed(self.seed)
        )[:subset_size].tolist()
        dataset[self.train_split] = dataset[self.train_split].select(indices)
        return dataset

    def _maybe_apply_split_transform(self, ddict):
        if self.split_transform is None:
            return ddict
        return self.split_transform(ddict)

    def setup(self, stage: Optional[str] = None):
        ddict = datasets.load_dataset(self.hf_repo, cache_dir=self.cache_dir)
        ddict = self._maybe_sample_subset(ddict, self.train_fraction)
        ddict = self._maybe_apply_split_transform(ddict)
        self.train_dataset = self._ds(ddict, self.train_split)
        self.val_dataset = self._ds(ddict, self.val_split)
        self.test_dataset = self._ds(ddict, self.test_split)
        self.tune_dataset = self._ds(ddict, self.tune_split)

    def _ds(self, ddict, key):
        return (
            SegmentationDataset(
                ddict[key],
                self.tokenizer,
                self.target_sr,
                self.max_speech_length,
                transform=self.transform,
            )
            if key in ddict
            else None
        )

    def _dl(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError(
                f"Train split '{self.train_split}' not found in dataset."
            )
        return self._dl(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError(
                f"Val split '{self.val_split}' not found in dataset."
            )
        return self._dl(self.val_dataset)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError(
                f"Test split '{self.test_split}' not found in dataset."
            )
        return self._dl(self.test_dataset)

    def predict_dataloader(self):
        predict_split = self.predict_split
        if isinstance(predict_split, str):
            predict_split = [predict_split]
        predict_datasets = []
        for split in predict_split:
            ds = getattr(self, f"{split}_dataset")
            if ds is not None:
                predict_datasets.append(ds)
        assert (
            len(predict_datasets) > 0
        ), f"None of the predict splits {self.predict_split} found in dataset."
        return self._dl(ConcatDataset(predict_datasets))


# builders
def build_segmentation_dataset(
    hf_repo: str,
    split: str,
    tokenizer,
    train_fraction: float = 1.0,
    **kwargs,
) -> SegmentationDataset:
    """Build a SegmentationDataset for one split, applying any
    registered DatasetDict-level split transform first
    (to mirror SegmentationDataModule's behaviour).
    """
    cache_dir = kwargs.get("cache_dir", "exp/cache/hf")
    target_sr = kwargs.get("target_sr", 16000)
    max_speech_length = kwargs.get("max_speech_length", None)
    transform = kwargs.get("transform", HF_REPO_TRANSFORMS.get(hf_repo))
    if transform is False:
        transform = None
    split_transform = kwargs.get(
        "split_transform", HF_REPO_SPLIT_TRANSFORMS.get(hf_repo)
    )
    if split_transform is False:
        split_transform = None
    if split_transform is not None:
        ddict = datasets.load_dataset(hf_repo, cache_dir=cache_dir)
        ddict = split_transform(ddict)
        if split not in ddict:
            raise KeyError(
                f"Split {split!r} not present after split_transform; "
                f"available: {list(ddict.keys())}"
            )
        ds = ddict[split]
    else:
        ds = datasets.load_dataset(hf_repo, split=split, cache_dir=cache_dir)
    return SegmentationDataset(
        ds,
        tokenizer,
        target_sr=target_sr,
        max_speech_length=max_speech_length,
        transform=transform,
    )


if __name__ == "__main__":
    # export HF_HOME="exp/cache/hf"
    # cp -f /u/sbharadwaj/.cache/huggingface/token exp/cache/hf/token [copy token]
    # python -m src.data.segmentation.segmentation_dataset
    tokenizer = type(
        "DummyTokenizer",
        (),
        {"tokens2ids": lambda self, phones: [i for i in range(len(phones))]},
    )()
    # DATASET = "changelinglab/buckeye-segment"
    DATASET = "changelinglab/timit-segment"
    # DATASET = "changelinglab/torgo-segment"
    dataset = build_segmentation_dataset(
        hf_repo=DATASET,
        split="test",
        tokenizer=tokenizer,
        train_fraction=1.0,
        cache_dir="exp/cache/hf",
        transform=False,
    )
    for item in dataset:
        print(item)
        break
    # dl = SegmentationDataModule(
    #     hf_repo=DATASET, tokenizer=tokenizer, batch_size=2
    # )
    # dl.setup()
    # for batch in dl.train_dataloader():
    #     print(batch)
    #     break
