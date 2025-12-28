import heapq
import torch
import torchaudio
import kaldiio
import numpy as np
import lightning as L
import json
import random
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from typing import List, Optional
import yaml

from tqdm import tqdm


class PackedKaldiDataset(Dataset):
    def __init__(
        self,
        wav_scp,
        text_file,
        lang_file,
        length_npy,
        vocab_file,
        max_duration=20.0,
        pack_factor=1.0,
        use_packing=1.0,
        sampling_rate=16000,
    ):
        self.sr, self.max_duration, self.use_packing = (
            sampling_rate,
            max_duration,
            use_packing,
        )
        self.pack_limit = max_duration * pack_factor

        self.wav_map = {l.split()[0]: l.split()[1] for l in open(wav_scp)}
        self.text_map = {l.split()[0]: " ".join(l.split()[1:]) for l in open(text_file)}
        self.lang_map = {
            l.split()[0]: l.split()[1].split("><")[0][1:].strip()
            for l in open(lang_file)
        }
        self.lengths = np.load(length_npy, allow_pickle=True).item()
        self.vocab = json.load(open(vocab_file))

        self.keys = [
            k for k in self.wav_map if k in self.lengths and k in self.text_map
        ]
        self.packs = self._generate_packs()
        self.pack_durations = [
            sum(self.lengths[k] for k in p) + (len(p) - 1) * 0.3 for p in self.packs
        ]

    def _generate_packs(self):
        L_filtered = {k: v for k, v in self.lengths.items() if v <= self.max_duration}
        all_keys = list(L_filtered.keys())
        random.seed(42)
        random.shuffle(all_keys)

        num_to_pack = int(len(all_keys) * self.use_packing)
        to_pack, standalone = all_keys[:num_to_pack], all_keys[num_to_pack:]
        final_packs = [[k] for k in standalone]

        if to_pack:
            # Sort descending to perform First-Fit Decreasing
            sorted_keys = sorted(to_pack, key=lambda k: L_filtered[k], reverse=True)
            # bin_contents: list of lists [key1, key2...]
            # heap: tracks (current_total_duration, bin_index)
            bin_contents = []
            heap = []
            for k in tqdm(sorted_keys, desc="Packing"):
                dur = L_filtered[k]
                # Check the bin with the smallest current duration (most available space)
                if heap and (heap[0][0] + dur + 0.3 <= self.pack_limit):
                    curr_dur, idx = heapq.heappop(heap)
                    bin_contents[idx].append(k)
                    heapq.heappush(heap, (curr_dur + dur + 0.3, idx))
                else:
                    # Create new bin
                    idx = len(bin_contents)
                    bin_contents.append([k])
                    heapq.heappush(heap, (dur, idx))

            final_packs.extend(bin_contents)

        return final_packs

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, idx):
        group = self.packs[idx]
        wavs, txts = [], []
        for i, key in enumerate(group):
            p = self.wav_map[key]
            p = p if p.startswith("/work") else f"/work/hdd/bbjs/shared/powsm/s2t1/{p}"
            w = (
                torch.from_numpy(kaldiio.load_mat(p)[1]).float().view(-1)
                if ".ark" in p
                else torchaudio.load(p)[0].view(-1)
            )
            wavs.append(w)
            txts.append(self.text_map[key])
            if i < len(group) - 1:
                wavs.append(torch.zeros(int(random.uniform(0.1, 0.4) * self.sr)))
                txts.append("<sep>")

        full_txt = " ".join(txts)
        return {
            "speech": torch.cat(wavs)[: int(self.max_duration * self.sr)],
            "tokens": torch.tensor(
                [
                    self.vocab.get(t.strip("/"), self.vocab.get("<unk>", -1))
                    for t in full_txt.replace("<sep>", " <sep> ").split()
                    if t.strip("/")
                ]
            ),
            "key": group[0],
        }


class DistributedDynamicSampler(Sampler):
    def __init__(
        self,
        durations: List[float],
        max_units: float,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
    ):
        self.durations = np.array(durations)
        self.max_units, self.num_replicas, self.rank, self.shuffle = (
            max_units,
            num_replicas,
            rank,
            shuffle,
        )
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.durations), generator=g).tolist()
        indices = indices[self.rank :: self.num_replicas]

        batch, current_units = [], 0
        for idx in indices:
            dur = self.durations[idx]
            if current_units + dur > self.max_units and batch:
                yield batch
                batch, current_units = [], 0
            batch.append(idx)
            current_units += dur
        if batch:
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch


class PackedKaldiDataModule(L.LightningDataModule):
    def __init__(
        self,
        scp_dict,
        txt_dict,
        lng_dict,
        len_dict,
        vocab_file,
        batch_size=16,
        max_units=200.0,
        dynamic_batching=True,
        pack_factor=1.0,
        use_packing=1.0,
        num_workers=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.files = {
            "wav": scp_dict,
            "txt": txt_dict,
            "lng": lng_dict,
            "len": len_dict,
        }

    def setup(self, stage=None):
        self.ds = {s: self._ds(s) for s in self.files["wav"].keys()}

    def _ds(self, split):
        # Packing only for train. Test/Val use standalone mode.
        packing_ratio = self.hparams.use_packing if split == "train" else 0.0
        return PackedKaldiDataset(
            self.files["wav"][split],
            self.files["txt"][split],
            self.files["lng"][split],
            self.files["len"][split],
            self.hparams.vocab_file,
            pack_factor=self.hparams.pack_factor,
            use_packing=packing_ratio,
        )

    def _dl(self, split, shuffle=False):
        ds = self.ds[split]
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if self.hparams.dynamic_batching:
            sampler = DistributedDynamicSampler(
                ds.pack_durations, self.hparams.max_units, world_size, rank, shuffle
            )
            return DataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=self.hparams.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=shuffle
            )
            return DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                sampler=sampler,
                num_workers=self.hparams.num_workers,
                collate_fn=self.collate_fn,
            )

    def collate_fn(self, batch):
        return {
            "speech": torch.nn.utils.rnn.pad_sequence(
                [x["speech"] for x in batch], batch_first=True
            ),
            "tokens": torch.nn.utils.rnn.pad_sequence(
                [x["tokens"] for x in batch], batch_first=True, padding_value=-1
            ),
            "speech_length": torch.tensor([len(x["speech"]) for x in batch]),
            "token_length": torch.tensor([len(x["tokens"]) for x in batch]),
        }

    def train_dataloader(self):
        return self._dl("train", shuffle=True)

    def val_dataloader(self):
        return self._dl("dev1k")


def build_kaldi_datamodule(
    dataset_name,
    dataset_config_path="configs/data/powsm_evalset_index.yaml",
    batch_size=16,
    num_workers=4,
    vocab_file: Optional[str] = None,
):
    with open(dataset_config_path) as f:
        config = yaml.safe_load(f)

    if dataset_name not in config["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds_config = config["datasets"][dataset_name]

    if "wav_scp" in ds_config:
        # we are missing the split level.
        # typically for eval datasets
        splits = ["predict"]
        ds_config = {splits[0]: ds_config}
    else:
        splits = list(ds_config.keys())
    wav_scp_file, text_file, lang_file, task_set = {}, {}, {}, {}
    for split in splits:
        wav_scp_file[split] = ds_config[split]["wav_scp"]
        text_file[split] = ds_config[split]["text_phoneme"]
        lang_file[split] = ds_config[split]["language"]
        task_set[split] = ds_config[split].get("task_set", None)

    sampling_rate = config.get("sampling_rate", 16000)

    return PackedKaldiDataModule(
        wav_scp_file=wav_scp_file,
        text_file=text_file,
        lang_file=lang_file,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        num_workers=num_workers,
        task_set=task_set,
        vocab_file=vocab_file,
    )


if __name__ == "__main__":
    # Test with: python -m src.data.packed_kaldi_dataset
    datamodule = build_kaldi_datamodule(
        dataset_name="pr_fixed",
        dataset_config_path="configs/data/ipapack_index.yaml",
        batch_size=2,
        num_workers=1,
        vocab_file="src/model/xeusphoneme/resources/ipa_vocab.json",
    )
    datamodule.setup()
    for i in datamodule.train_dataloader().dataset:
        print(i["speech_length"])
        break
    # print(len(datamodule.predict_dataloader().dataset))
    # for batch in datamodule.predict_dataloader().dataset:
    #     print(batch)
    #     break
