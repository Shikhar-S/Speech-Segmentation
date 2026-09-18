"""
Buckeye Corpus Data Preparation

This script segments long recordings into chunks at pauses, writes per-split
metadata JSON and the clipped WAVs (``<output_dir>/speech_clips/``), and
produces a seeded speaker-disjoint 80/10/10 split.

Usage:
    python scripts/data_prep/buckeye_data_prep.py \
        --buckeye_root $BUCKEYE_ROOT     # directory holding s01.zip ... s40.zip
        --output_dir   exp/data/buckeye_meta
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict
import argparse, json, logging, numpy as np, pandas as pd
import buckeye
from tqdm import tqdm

log = logging.getLogger("buckeye_prep")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class BuckeyeAlignmentSegment:
    segment_id: str
    speaker_id: str
    track_id: str
    start_time: float
    end_time: float
    text: str
    phones: List[str]
    phone_timestamps: List[Tuple[float, float]]
    speaker_id: str


def pause_segments(words, min_pause: float) -> List[Tuple[float, float]]:
    """Return [(seg_start, seg_end), ...] by splitting at pauses ≥ min_pause."""
    segs, cur_start, last_end = [], None, None
    for w in words:
        is_pause = getattr(w, "__class__", type("X", (), {})).__name__ == "Pause"
        if not is_pause:  # Word
            cur_start = w.beg if cur_start is None else cur_start
            last_end = w.end
        elif cur_start is not None and last_end is not None and w.dur >= min_pause:
            segs.append((cur_start, last_end))
            cur_start = last_end = None
    if cur_start is not None and last_end is not None:
        segs.append((cur_start, last_end))
    return segs


def clip_phones(
    phones, t0: float, t1: float
) -> Tuple[List[str], List[Tuple[float, float]]]:
    out_p, out_t = [], []
    for p in phones:
        if p.beg >= t0 and p.end <= t1:
            lab = getattr(p, "seg", None)
            if lab and lab.upper() not in {"SIL", "<SIL>", "SP", ""}:
                out_p.append(lab)
                out_t.append((p.beg - t0, p.end - t0))
    return out_p, out_t


def clip_words(words, t0: float, t1: float) -> str:
    toks = []
    for w in words:
        if (
            getattr(w, "__class__", type("X", (), {})).__name__ == "Word"
            and w.beg >= t0
            and w.end <= t1
        ):
            orth = getattr(w, "orthography", "") or ""
            if orth and not orth.startswith("<") and not orth.startswith("{"):
                toks.append(orth.strip())
    return " ".join(toks)


def split_if_long(t0: float, t1: float, max_dur: float) -> List[Tuple[float, float]]:
    dur = t1 - t0
    if dur <= max_dur:
        return [(t0, t1)]
    n = int(np.ceil(dur / max_dur))
    step = dur / n
    return [(t0 + i * step, min(t0 + (i + 1) * step, t1)) for i in range(n)]


# ----------------------------- main pipeline ----------------------------- #


def segment_track(
    track, spk: str, min_pause: float, max_seg: float, min_seg: float
) -> List[Tuple[Tuple[float, float], str, List[str], List[Tuple[float, float]], str]]:
    segs = []
    track_duration = track.wav.getnframes() / float(track.wav.getframerate())
    for i, (a, b) in enumerate(pause_segments(track.words, min_pause)):
        for j, (t0, t1) in enumerate(split_if_long(a, b, max_seg)):
            if (t1 - t0) < min_seg or t0 >= track_duration or t1 > track_duration:
                continue
            ph, ph_t = clip_phones(track.phones, t0, t1)
            tx = clip_words(track.words, t0, t1)
            if ph and tx:
                segs.append(
                    (
                        (t0, t1),
                        tx,
                        ph,
                        ph_t,
                        f"{spk}_{track.name}_{i:04d}" + (f"_{j:02d}" if j else ""),
                    )
                )
    return segs


def process_corpus(
    root: Path, min_pause: float, min_seg: float, max_seg: float, clips_dir: Path
) -> List[BuckeyeAlignmentSegment]:
    all_segments: List[BuckeyeAlignmentSegment] = []
    clips_dir.mkdir(parents=True, exist_ok=True)
    for spk in tqdm(buckeye.corpus(str(root), load_wavs=True), desc="speakers"):
        spk_id = spk.name
        log.info(f"Speaker {spk_id} ({spk.sex}, {spk.age})")
        for tr_idx, tr in tqdm(
            enumerate(spk.tracks), desc=f"{spk_id} tracks", leave=False
        ):
            seg_list = list(segment_track(tr, spk_id, min_pause, max_seg, min_seg))
            for (t0, t1), txt, ph, ph_t, seg_stub in tqdm(
                seg_list, desc=f"{spk_id}:{tr.name}", leave=False
            ):
                tr.clip_wav(str(clips_dir / f"{seg_stub}.wav"), t0, t1)
                all_segments.append(
                    BuckeyeAlignmentSegment(
                        segment_id=seg_stub,
                        speaker_id=spk_id,
                        track_id=tr_idx,
                        start_time=t0,
                        end_time=t1,
                        text=txt,
                        phones=ph,
                        phone_timestamps=ph_t,
                    )
                )

    log.info(f"Total segments: {len(all_segments)}")
    return all_segments


def speaker_disjoint_splits(
    segments: List[BuckeyeAlignmentSegment], ratio: Dict[str, float]
) -> Dict[str, List[BuckeyeAlignmentSegment]]:
    spk2segs: Dict[str, List[BuckeyeAlignmentSegment]] = {}
    for s in segments:
        spk2segs.setdefault(s.speaker_id, []).append(s)
    spks = np.array(sorted(spk2segs.keys()))
    rng = np.random.default_rng(42)
    rng.shuffle(spks)

    n = len(spks)
    n_tr = int(n * ratio["train"])
    n_va = int(n * ratio["val"])
    split_spks = {
        "train": spks[:n_tr],
        "val": spks[n_tr : n_tr + n_va],
        "test": spks[n_tr + n_va :],
    }
    print("Split sizes:", {k: len(v) for k, v in split_spks.items()})
    return {
        k: [ss for sp in split_spks[k] for ss in spk2segs[sp]]
        for k in ("train", "val", "test")
    }


def save_metadata(
    out_dir: Path, name: str, segs: List[BuckeyeAlignmentSegment]
) -> None:
    with open(out_dir / f"{name}_metadata.json", "w") as f:
        json.dump(
            [asdict(s) | {"duration": s.end_time - s.start_time} for s in segs],
            f,
            indent=2,
        )
    df = pd.DataFrame(
        [
            {
                "segment_id": s.segment_id,
                "track_id": s.track_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.end_time - s.start_time,
                "n_phones": len(s.phones),
                "n_words": len(s.text.split()),
                "speaker_id": s.speaker_id,
            }
            for s in segs
        ]
    )
    df.to_csv(out_dir / f"{name}_metadata.csv", index=False)
    log.info(f"Saved {name}: {len(segs)} segments")


def main():
    p = argparse.ArgumentParser(description="Prepare Buckeye corpus for CTC alignment")
    p.add_argument(
        "--buckeye_root",
        required=True,
        type=Path,
        help="Path to Buckeye root (sXX.zip files unpacked/accessible)",
    )
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument(
        "--min_pause",
        type=float,
        default=1.0,
        help="Pause ≥ this (s) defines a boundary",
    )
    p.add_argument("--min_segment", type=float, default=0.5)
    p.add_argument("--max_segment", type=float, default=20.0)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"Loading Buckeye from: {args.buckeye_root}")
    segs = process_corpus(
        args.buckeye_root,
        args.min_pause,
        args.min_segment,
        args.max_segment,
        out / "speech_clips",
    )

    if not segs:
        log.error("No segments extracted. Check corpus path/structure.")
        return

    splits = speaker_disjoint_splits(segs, {"train": 0.8, "val": 0.1, "test": 0.1})
    for k, v in splits.items():
        save_metadata(out, k, v)

    durs = np.array([s.end_time - s.start_time for s in segs])
    log.info(
        f"Dur (s): mean={durs.mean():.2f} std={durs.std():.2f} min={durs.min():.2f} max={durs.max():.2f} total={durs.sum()/3600:.2f}h"
    )


if __name__ == "__main__":
    main()
