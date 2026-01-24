"""
Tag accent and langid (multi-worker loading via TorchDataModule).

Examples:
  # Lang
  python -m src.core.tools.data_tagger \
      --wav_scp /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/train_1k_fixed_1task/wav.scp \
      --hf_repo "speechbrain/lang-id-voxlingua107-ecapa" \
      --out exp/runs/tags/lang --batch_size 24 --num_workers 8

      python -m src.core.tools.data_tagger \
      --wav_scp /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/train_1k_fixed_1task/wav.scp \
      --hf_repo "speechbrain/lang-id-voxlingua107-ecapa" \
      --out exp/runs/tags/langwscore --batch_size 24 --num_workers 8 --max_utts 1000
      
      
      
  # Accent
  python -m src.core.tools.data_tagger \
      --wav_scp /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/train_1k_fixed_1task/wav.scp \
      --hf_repo "Jzuluaga/accent-id-commonaccent_ecapa" \
      --out exp/runs/tags/accent --batch_size 24 --num_workers 8

Output:
  <out>.shard<SLURM_ARRAY_TASK_ID>.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import kaldiio
import numpy as np
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def to_mono_float32(x: Any) -> np.ndarray:
    """Accept np.ndarray or torch.Tensor; return mono float32 numpy (T,)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim >= 2:
        # works for (C, T) or (T, C)
        channel_dim = 0 if x.shape[0] < x.shape[1] else 1
        x = x.mean(axis=channel_dim)
    return x.astype(np.float32, copy=False)


def filter_ds(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        utt, _ = ln.strip().split(None, 1)
        if "_cv_" in utt:
            out.append(ln)
    return out


def label_to_str(lab: Any) -> str:
    if isinstance(lab, torch.Tensor):
        if lab.numel() == 1:
            return str(lab.item())
        return str(lab.detach().cpu().tolist())
    return str(lab)


class WavScpDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]], base_dir: str, default_sr: int):
        self.items = items
        self.base_dir = base_dir
        self.default_sr = int(default_sr)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        utt, wav_path = self.items[idx]
        wp = wav_path
        if not Path(wp).is_absolute():
            wp = os.path.join(self.base_dir, wp)

        if ".ark" in wp:
            val = kaldiio.load_mat(wp)
            if isinstance(val, tuple) and len(val) == 2:
                sr, wav = val
            else:
                sr, wav = self.default_sr, val
            w = to_mono_float32(wav)
        else:
            w_t, sr = torchaudio.load(wp)  # (C,T), int
            w = to_mono_float32(w_t)

        return {
            "utt_id": utt,
            "wavpath": wp,
            "sr": int(sr),
            "wav": w,  # np.float32 (T,)
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = [b for b in batch if b["wav"].size > 0]
    if len(batch) == 0:
        return {"empty": True}

    lens = torch.tensor([b["wav"].shape[0] for b in batch], dtype=torch.float32)
    T = int(lens.max().item())
    wavs = torch.zeros((len(batch), T), dtype=torch.float32)
    for i, b in enumerate(batch):
        w = torch.from_numpy(b["wav"])
        wavs[i, : w.numel()] = w

    lens_rel = lens / float(T)  # SpeechBrain expects relative lengths in [0, 1]
    return {
        "empty": False,
        "wavs": wavs,  # (B,T)
        "lens_rel": lens_rel,  # (B,)
        "lens_abs": lens,  # (B,)
        "utts": [b["utt_id"] for b in batch],
        "srs": [b["sr"] for b in batch],
        "paths": [b["wavpath"] for b in batch],
    }


def dataloader(batch_size: int, num_workers: int, dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_batch,
        drop_last=False,
    )


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_scp", required=True)
    ap.add_argument("--hf_repo", required=True)
    ap.add_argument(
        "--out", required=True, help="output prefix (writes <out>.shardK.jsonl)"
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_utts", type=int, default=0, help="0 means all")
    ap.add_argument(
        "--default_sr", type=int, default=16000, help="used when .ark has no sr"
    )
    args = ap.parse_args()

    BASE_DIR = "/work/hdd/bbjs/shared/powsm/s2t1"

    sid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    nsh = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    out_jsonl = f"{args.out}.shard{sid}.jsonl"
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = EncoderClassifier.from_hparams(
        source=args.hf_repo,
        savedir=f"exp/cache/{args.hf_repo}",
        run_opts={"device": device},
    )

    lines = [ln for ln in open(args.wav_scp) if ln.strip()]
    lines = filter_ds(lines)
    if args.max_utts > 0:
        lines = lines[: args.max_utts]
    lines = lines[sid::nsh]

    items: List[Tuple[str, str]] = []
    for ln in lines:
        utt, wav_path = ln.strip().split(None, 1)
        items.append((utt, wav_path))

    ds = WavScpDataset(items, base_dir=BASE_DIR, default_sr=args.default_sr)
    loader = dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers, dataset=ds
    )

    total_batches = (len(ds) + args.batch_size - 1) // args.batch_size
    print(f"Shard {sid}/{nsh}: {len(ds)} utts, ~{total_batches} batches -> {out_jsonl}")

    with open(out_jsonl, "w") as fo:
        for batch in tqdm(
            loader, total=total_batches, desc=f"Tagging {Path(out_jsonl).name}"
        ):
            if batch.get("empty", False):
                continue

            wavs = batch["wavs"].to(device, non_blocking=True)
            lens_rel = batch["lens_rel"].to(device, non_blocking=True)

            _, score, _, pred = clf.classify_batch(wavs, lens_rel)
            pred_iter = list(pred)
            score_iter = list(score)
            for u, sr, L, lab, labscr, path in zip(
                batch["utts"],
                batch["srs"],
                batch["lens_abs"].tolist(),
                pred_iter,
                score_iter,
                batch["paths"],
            ):
                fo.write(
                    json.dumps(
                        {
                            "utt_id": u,
                            "wavpath": str(path),
                            "duration_sec": float(L) / float(sr),
                            "tag": label_to_str(lab),
                            "tag_score": label_to_str(labscr),
                        }
                    )
                    + "\n"
                )

    print(f"Wrote {out_jsonl}")


if __name__ == "__main__":
    main()
