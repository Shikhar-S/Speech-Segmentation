"""MFA topline.

This implementation is for the topline MFA numbers, which uses both speech and text to do
text-dependent alignment (aka Forced Alignment). It also considers speaker adaptation.
Check MFA documentation (concepts) for more details.
MFA documentation: https://montreal-forced-aligner.readthedocs.io/en/latest/

Installation commands:
    micromamba create -n mfa -c conda-forge python=3.10 montreal-forced-aligner
    micromamba activate mfa
    micromamba install pip
    pip install datasets==3.6.0 torchcodec==0.11.1 espnet torchaudio panphon
    #TODO(shikhar): remove espnet dependency

Usage:
    python -m src.model.mfa.inference_topline \
        --hf_repo changelinglab/timit-segment \
        --split test \
        --mfa_cache_dir exp/cache/mfa \
        --run_dir exp/runs/mfa/topline_timit
    
    python -m src.model.mfa.inference_topline \
        --hf_repo changelinglab/buckeye-segment \
        --split test \
        --mfa_cache_dir exp/cache/mfa \
        --run_dir exp/runs/mfa/topline_buckeye

    python -m scripts.eval_segmentation "exp/runs/mfa/topline_timit/results.jsonl" --strip-outer-silences
    
    To align with orthographic transcript, use --units words and 
    specify a dictionary for the language with --dictionary (e.g. --dictionary english_mfa).
    All dicts are available here:
        mfa list models --type dictionary --cache-dir exp/cache/mfa
"""

import argparse
import contextlib
import dataclasses
import json
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

from src.data.segmentation.segmentation_dataset import (
    DummyTokenizer,
    build_segmentation_dataset,
)
from src.model.mfa.utils import (
    _phones_from_mfa_json,
    _save_utterance,
    build_phone_dict,
    ensure_mfa_model,
    mfa_env,
)


def _build_corpus(
    dataset,
    corpus_dir: Path,
    sr: int = 16000,
    units: str = "phones",
) -> tuple[dict[str, int], set[str]]:
    """Write a Prosodylab-format corpus from a SegmentationDataset.

    Creates corpus_dir/{speaker_id}/{utt_id}.wav and .lab for each item.
    Skips existing WAV files so the function is safe to resume.

    Args:
        dataset: A SegmentationDataset instance.
        corpus_dir: Destination directory (must exist).
        sr: Sample rate for saved WAV files.
        units: The units to align, either "phones" or "words".

    Returns:
        Tuple of ({utt_id: dataset_index}, set of all phone symbols seen).
        The phone set is empty when units="words".
    """
    utt_idx_map: dict[str, int] = {}
    all_phones: set[str] = set()
    for i in tqdm(range(len(dataset)), desc="Building corpus"):
        item = dataset[i]
        utt_id: str = item["utt_id"]
        spk_dir = corpus_dir / str(item["speaker_id"])
        spk_dir.mkdir(parents=True, exist_ok=True)
        wav_path = spk_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            speech = item["speech"][: item["speech_length"]].float()
            if units == "phones":
                phones = item["phones"]
                transcript = " ".join(phones)
                all_phones.update(p for p in phones if p)
            else:
                transcript = item["text"]
            _save_utterance(speech, transcript, wav_path, sr)
        utt_idx_map[utt_id] = i
    return utt_idx_map, all_phones


def _collect_results(
    output_dir: Path,
    utt_idx_map: dict[str, int],
    dataset,
) -> dict[int, dict]:
    """Walk MFA output JSONs and assemble per-utterance result records.

    Args:
        output_dir: Root of MFA output (mirrors corpus speaker-subdir layout).
        utt_idx_map: Mapping {utt_id: dataset_index} from _build_corpus.
        dataset: SegmentationDataset for ground-truth passthrough data.

    Returns:
        {dataset_index: {"pred": List[SegmentationUnit], "passthrough": dict}}.
        Utterances that MFA failed to align (no output JSON) are omitted.
    """
    records: dict[int, dict] = {}
    for spk_dir in output_dir.iterdir():
        for json_path in spk_dir.rglob("*.json"):
            utt_id = json_path.relative_to(spk_dir).with_suffix("").as_posix()
            if utt_id not in utt_idx_map:
                continue
            i = utt_idx_map[utt_id]
            item = dataset[i]
            records[i] = {
                "pred": _phones_from_mfa_json(json_path),
                "passthrough": {
                    "utt_id": item["utt_id"],
                    "phone_timestamps": item["phone_timestamps"],
                    "phones": item["phones"],
                    "split": item["split"],
                },
            }
    return records


def _write_jsonl(records: dict[int, dict], out_file: Path) -> None:
    """Write the records dict to a JSONL file."""
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, record in records.items():
            pred_dicts = [dataclasses.asdict(u) for u in record["pred"]]
            line = json.dumps(
                {
                    str(i): {
                        "pred": pred_dicts,
                        "passthrough": record["passthrough"],
                    }
                },
                ensure_ascii=False,
            )
            f.write(line + "\n")
    print(f"Wrote {len(records)} utterances to {out_path}.", flush=True)


def run_mfa_batch_inference(
    hf_repo: str,
    split: str,
    run_dir: str,
    dictionary: str = "english_mfa",
    acoustic_model: str = "english_mfa",
    sr: int = 16000,
    cache_dir: str = "exp/cache/hf",
    mfa_cache_dir: str | None = None,
    units: str = "phones",
) -> None:
    """Build a speaker corpus, run mfa align, write eval-compatible JSONL.

    The JSONL schema matches distributed_inference output so
    scripts/eval_segmentation.py can read it without modification.

    Args:
        hf_repo: HuggingFace dataset repository.
        split: Dataset split to process (e.g. "test").
        run_dir: Directory to store intermediate corpus and output files.
        dictionary: MFA dictionary name or path. Ignored when units="phones".
        acoustic_model: MFA acoustic model name or path.
        sr: Target sample rate for saved WAV files.
        cache_dir: HuggingFace dataset cache directory.
        mfa_cache_dir: Directory for MFA pretrained models
            (sets MFA_ROOT_DIR). Models are downloaded here if absent.
        units: Whether to align to phones or words. Phones are more common and
            provide finer-grained alignment.
    """
    env = mfa_env(mfa_cache_dir)
    ensure_mfa_model(
        acoustic_model,
        dictionary=None if units == "phones" else dictionary,
        env=env,
    )

    dataset = build_segmentation_dataset(
        hf_repo, split, tokenizer=DummyTokenizer(), cache_dir=cache_dir
    )

    run_dir = Path(run_dir)
    corpus_dir = run_dir / "corpus"
    output_dir = run_dir / "output"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building corpus in {corpus_dir} ...", flush=True)
    utt_idx_map, all_phones = _build_corpus(
        dataset, corpus_dir, sr=sr, units=units
    )

    if units == "phones":
        dict_path = run_dir / "phone_dict.txt"
        build_phone_dict([list(all_phones)], dict_path)
        effective_dictionary = str(dict_path)
    else:
        effective_dictionary = dictionary

    print("Running mfa align ...", flush=True)
    subprocess.run(
        [
            "mfa",
            "align",
            str(corpus_dir),
            effective_dictionary,
            acoustic_model,
            str(output_dir),
            "--clean",
            "--overwrite",
            "--output_format",
            "json",
        ],
        env=env,
        check=True,
    )

    print("Collecting results ...", flush=True)
    records = _collect_results(output_dir, utt_idx_map, dataset)
    _write_jsonl(records, run_dir / "results.jsonl")


def main() -> None:
    """CLI entry point for MFA batch forced-alignment inference."""
    parser = argparse.ArgumentParser(
        description="MFA batch forced-alignment inference → JSONL"
    )
    parser.add_argument("--hf_repo", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Store intermediate corpus, mfa output and final evaluation compatible output here.",
    )
    parser.add_argument("--dictionary", default="english_mfa")
    parser.add_argument("--acoustic_model", default="english_mfa")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--cache_dir", default="exp/cache/hf")
    parser.add_argument("--mfa_cache_dir", default=None)
    parser.add_argument(
        "--units",
        choices=["phones", "words"],
        default="phones",
        help="Whether to align to phones or words",
    )
    args = parser.parse_args()
    run_mfa_batch_inference(
        hf_repo=args.hf_repo,
        split=args.split,
        run_dir=args.run_dir,
        dictionary=args.dictionary,
        acoustic_model=args.acoustic_model,
        sr=args.sr,
        cache_dir=args.cache_dir,
        mfa_cache_dir=args.mfa_cache_dir,
        units=args.units,
    )


if __name__ == "__main__":
    main()
