"""Corpus forced alignment under native MFA 2.2.17.

The only Tamil acoustic model, ``tamil_cv``, is an MFA **v2.0.0** model. 
It needs a dedicated ``mfa2`` micromamba env. 
MFA 2.x has no ``align_one``, so this uses the corpus ``mfa align``
workflow on a one-shot corpus built from a chosen phone source:

  - ``gt``      gold phones (the SSNCE topline)
  - ``cascade`` PhoneticXEUS predicted phones

Output is written as eval-format JSONL ({"start","end","label"} units +
``phone_timestamps`` passthrough) for ``scripts/eval_segmentation.py``.

Run (CPU only, no GPU):
    PYTHONPATH=. .venv/bin/python -m src.model.mfa.mfa2_align --source gt
    PYTHONPATH=. .venv/bin/python scripts/eval_segmentation.py \
        exp/runs/mfa2_tamil/gt/pred.jsonl \
        --tolerance-ms 20 --mode strict --strip-outer-silences
"""

import argparse
import dataclasses
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.segmentation.segmentation_dataset import (
    DummyTokenizer,
    build_segmentation_dataset,
)
from src.metrics.types import SegmentationUnit
from src.model.mfa.utils import (
    build_phone_dict,
    is_mfa_silence,
    normalize_phones_for_mfa_tamil,
    strip_outer_silence_units,
    _save_utterance,
)

# The mfa2 env + the cached v2 model. Overridable via env vars / CLI.
MFA2_ENV = "mfa2"
MAMBA_BIN = os.environ.get("MAMBA_BIN", shutil.which("micromamba") or "micromamba")
MAMBA_ROOT = os.environ.get("MAMBA_ROOT_PREFIX", os.path.expanduser("~/micromamba"))
DEFAULT_MODEL_ZIP = "exp/cache/mfa/pretrained_models/acoustic/tamil_cv.zip"
DEFAULT_MFA2_ROOT = "exp/cache/mfa2"
DEFAULT_RECOG_JSONL = (
    "exp/runs/mfa2_tamil/recog/ssnce_xeuspr_tamil_phones.*.jsonl"
)
DEFAULT_HF_REPO = "changelinglab/ssnce-segment"
# Must match the data config used for the recognition step
# (configs/data/segmentation_phonvec.yaml) so the corpus audio is bit-identical
# to what the recognizer saw.
SR = 16000


def run_mfa2_align(
    corpus_dir: Path,
    dict_path: Path,
    model_zip: str,
    out_dir: Path,
    *,
    mfa2_root: str,
    beam: int = 100,
    retry_beam: int = 400,
) -> subprocess.CompletedProcess:
    """Run ``mfa align`` (MFA 2.x) in the mfa2 env over a prepared corpus.

    Args:
        corpus_dir: Directory of ``<uid>.wav`` + ``<uid>.lab`` files.
        dict_path: Phone-to-phone pronunciation dictionary.
        model_zip: Path to the cached ``tamil_cv.zip`` acoustic model.
        out_dir: Destination for the per-utterance JSON alignments.
        mfa2_root: ``MFA_ROOT_DIR`` for the v2 model store (kept separate from
            the v3 cache).
        beam: Alignment beam (wide for the weak tamil_cv acoustic match).
        retry_beam: Retry beam.

    Returns:
        The completed subprocess (``returncode``/``stderr`` for diagnostics).
    """
    cmd = [
        MAMBA_BIN,
        "run",
        "-n",
        MFA2_ENV,
        "mfa",
        "align",
        str(corpus_dir),
        str(dict_path),
        str(model_zip),
        str(out_dir),
        "--beam",
        str(beam),
        "--retry_beam",
        str(retry_beam),
        "--output_format",
        "json",
        "--clean",
        "--single_speaker",
    ]
    env = {
        "MFA_ROOT_DIR": str(mfa2_root),
        "MAMBA_ROOT_PREFIX": MAMBA_ROOT,
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", ""),
    }
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def parse_mfa_json(out_dir: Path, uid: str) -> Optional[List[SegmentationUnit]]:
    """Parse one MFA 2.x JSON export's phone tier into SegmentationUnits."""
    jp = Path(out_dir) / f"{uid}.json"
    if not jp.exists():
        return None
    with open(jp) as f:
        tiers = json.load(f).get("tiers", {})
    tier = tiers.get("phones") or next(iter(tiers.values()), {})
    return [
        SegmentationUnit(start=float(s), end=float(e), label=lab)
        for s, e, lab in tier.get("entries", [])
    ]


def _split_transcript(transcript: str) -> List[str]:
    """Split a slash-joined ``predicted_transcript`` into phone tokens.

    Drops empty/whitespace tokens (the bare ``split('/')`` leaves these for
    leading/trailing/doubled slashes) and any special ``<…>`` vocab tokens.
    """
    return [
        p
        for p in (s.strip() for s in transcript.split("/"))
        if p and not (p.startswith("<") and p.endswith(">"))
    ]


def load_recog_phones(glob_pat: str) -> Dict[str, List[str]]:
    """utt_id -> predicted phone tokens from a distributed_inference recognition
    dump (the masked PhoneticXEUS recognizer's ``predicted_transcript``)."""
    out: Dict[str, List[str]] = {}
    for fp in sorted(glob.glob(glob_pat)):
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = next(iter(json.loads(line).values()))
                pred = rec.get("pred")
                if not isinstance(pred, list) or not pred:
                    continue
                transcript = pred[0].get("predicted_transcript", "")
                out[rec["passthrough"]["utt_id"]] = _split_transcript(
                    transcript
                )
    return out


def build_corpus(
    source: str, corpus_dir: Path, recog_jsonl: str, limit: int = 0
) -> Tuple[Dict[str, Tuple[list, list]], List[List[str]]]:
    """Write the MFA corpus (wav + label per utt) for the chosen phone source.

    Audio load, resample (to ``SR``), and the ``process_ssnce_symbols`` transform
    come from the shared ``SegmentationDataset`` — the same pipeline the
    recognition step uses — so the corpus audio matches what the recognizer saw.
    The raw phones (gold for ``gt``, recognizer predictions for ``cascade``) go
    through the *same* ``normalize_phones_for_mfa_tamil`` + silence filter; only
    the source differs. Returns ``(refs, all_content)`` where ``refs`` maps
    utt_id -> ``(phone_timestamps, gold_phones)`` for eval passthrough, and
    ``all_content`` is the list of per-utt phone sequences (for the dictionary).
    """
    recog = load_recog_phones(recog_jsonl) if source == "cascade" else None
    dataset = build_segmentation_dataset(
        DEFAULT_HF_REPO, "test", DummyTokenizer(), target_sr=SR
    )
    gt: Dict[str, Tuple[list, list]] = {}
    all_content: List[List[str]] = []
    for item in dataset:
        if limit and len(gt) >= limit:
            break
        uid = item["utt_id"]
        ts, phones = item["phone_timestamps"], item["phones"]
        raw = phones if source == "gt" else recog.get(uid, [])
        content = [
            p
            for p in normalize_phones_for_mfa_tamil(raw)
            if p and not is_mfa_silence(p)
        ]
        if not content:
            continue
        _save_utterance(
            item["speech"], " ".join(content), corpus_dir / f"{uid}.wav", SR
        )
        gt[uid] = (ts, phones)
        all_content.append(content)
    return gt, all_content


def write_pred_jsonl(
    refs: Dict[str, Tuple[list, list]], out_dir: Path, pred_path: Path
) -> int:
    """Parse all alignments into the eval JSONL wire format; return count."""
    n = 0
    with open(pred_path, "w") as f:
        for uid, (ts, phones) in refs.items():
            units = parse_mfa_json(out_dir, uid)
            if units is None:
                continue
            units = strip_outer_silence_units(units)
            rec = {
                uid: {
                    "pred": [dataclasses.asdict(u) for u in units],
                    "passthrough": {
                        "utt_id": uid,
                        "phone_timestamps": ts,
                        "phones": phones,
                        "split": "test",
                    },
                }
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    """CLI: build the SSNCE corpus, align under mfa2, write eval JSONL."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["gt", "cascade"], required=True)
    ap.add_argument("--out-root", default="exp/runs/mfa2_tamil")
    ap.add_argument("--model-zip", default=DEFAULT_MODEL_ZIP)
    ap.add_argument("--mfa2-root", default=DEFAULT_MFA2_ROOT)
    ap.add_argument(
        "--recog-jsonl",
        default=DEFAULT_RECOG_JSONL,
        help="glob of the recognition dump jsonl (for source=cascade)",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.out_root) / args.source
    corpus, out = root / "corpus", root / "out"
    if root.exists():
        shutil.rmtree(root)
    corpus.mkdir(parents=True)
    out.mkdir(parents=True)
    Path(args.mfa2_root).mkdir(parents=True, exist_ok=True)

    refs, all_content = build_corpus(
        args.source, corpus, args.recog_jsonl, args.limit
    )
    build_phone_dict(all_content, root / "dict.txt")
    print(f"corpus: {len(refs)} utts", flush=True)

    model_zip = str(Path(args.model_zip).resolve())
    mfa2_root = str(Path(args.mfa2_root).resolve())
    r = run_mfa2_align(
        corpus, root / "dict.txt", model_zip, out, mfa2_root=mfa2_root
    )
    print("align rc:", r.returncode, "| stderr tail:\n", r.stderr[-800:])

    n = write_pred_jsonl(refs, out, root / "pred.jsonl")
    print(f"parsed {n}/{len(refs)} -> {root / 'pred.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
