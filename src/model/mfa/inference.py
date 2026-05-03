"""MFA batch inference: build speaker corpus, run mfa align, parse JSON outputs.

MFA documentation: https://montreal-forced-aligner.readthedocs.io/en/latest/

Usage:
    micromamba activate mfa
    python -m src.model.mfa.inference \\
        --hf_repo changelinglab/timit-segment \\
        --split test \\
        --out_file exp/runs/mfa_timit/mfa.jsonl \\
        --mfa_cache_dir exp/cache/mfa
"""

import argparse
import contextlib
import dataclasses
import json
import subprocess
import tempfile
from pathlib import Path

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
    use_phones: bool = True,
) -> tuple[dict[str, int], set[str]]:
    """Write a Prosodylab-format corpus from a SegmentationDataset.

    Creates corpus_dir/{speaker_id}/{utt_id}.wav and .lab for each item.
    Skips existing WAV files so the function is safe to resume.

    Args:
        dataset: A SegmentationDataset instance.
        corpus_dir: Destination directory (must exist).
        sr: Sample rate for saved WAV files.
        use_phones: If True, write the phone sequence to .lab instead of
            the word transcript.

    Returns:
        Tuple of ({utt_id: dataset_index}, set of all phone symbols seen).
        The phone set is empty when use_phones is False.
    """
    utt_idx_map: dict[str, int] = {}
    all_phones: set[str] = set()
    for i in range(len(dataset)):
        item = dataset[i]
        utt_id: str = item["utt_id"]
        spk_dir = corpus_dir / str(item["speaker_id"])
        spk_dir.mkdir(parents=True, exist_ok=True)
        wav_path = spk_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            speech = item["speech"][: item["speech_length"]].float()
            if use_phones:
                phones = item["phones"]
                transcript = " ".join(phones)
                all_phones.update(p for p in phones if p)
            else:
                transcript = item["text"]
            _save_utterance(speech, transcript, wav_path, sr)
        utt_idx_map[utt_id] = i
    return utt_idx_map, all_phones


def _run_align(
    corpus_dir: Path,
    output_dir: Path,
    dictionary: str,
    acoustic_model: str,
    env: dict[str, str] | None = None,
) -> None:
    """Run mfa align on the corpus directory.

    Args:
        corpus_dir: Prosodylab-format corpus directory.
        output_dir: Directory where MFA writes output JSON files.
        dictionary: MFA dictionary name or path to a custom dictionary file.
        acoustic_model: MFA acoustic model name or path.
        env: Environment dict from ``mfa_env``; controls MFA_ROOT_DIR.
    """
    subprocess.run(
        [
            "mfa", "align",
            str(corpus_dir), dictionary, acoustic_model, str(output_dir),
            "--clean", "--overwrite", "--output_format", "json",
        ],
        env=env,
        check=True,
    )


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
    for json_path in output_dir.rglob("*.json"):
        utt_id = json_path.stem
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


def run_mfa_batch_inference(
    hf_repo: str,
    split: str,
    out_file: str,
    dictionary: str = "english_mfa",
    acoustic_model: str = "english_mfa",
    sr: int = 16000,
    cache_dir: str = "exp/cache/hf",
    mfa_cache_dir: str | None = None,
    temp_dir: str | None = None,
    use_phones: bool = True,
) -> None:
    """Build a speaker corpus, run mfa align, write eval-compatible JSONL.

    The JSONL schema matches distributed_inference output so
    scripts/eval_segmentation.py can read it without modification.

    Args:
        hf_repo: HuggingFace dataset repository.
        split: Dataset split to process (e.g. "test").
        out_file: Output JSONL path.
        dictionary: MFA dictionary name or path. Ignored when use_phones=True.
        acoustic_model: MFA acoustic model name or path.
        sr: Target sample rate for saved WAV files.
        cache_dir: HuggingFace dataset cache directory.
        mfa_cache_dir: Directory for MFA pretrained models
            (sets MFA_ROOT_DIR). Models are downloaded here if absent.
        temp_dir: If given, use this directory for intermediate corpus and
            output files instead of a managed temporary directory.
        use_phones: If True (default), write the dataset's phone sequence to
            .lab and generate a phone-to-phone dictionary on the fly.
            If False, use the word transcript and the named dictionary.
    """
    env = mfa_env(mfa_cache_dir)
    ensure_mfa_model(
        acoustic_model,
        dictionary=None if use_phones else dictionary,
        env=env,
    )

    dataset = build_segmentation_dataset(
        hf_repo, split, tokenizer=DummyTokenizer(), cache_dir=cache_dir
    )

    ctx = (
        tempfile.TemporaryDirectory()
        if temp_dir is None
        else contextlib.nullcontext(temp_dir)
    )
    with ctx as tmp:
        tmp = Path(tmp)
        corpus_dir = tmp / "corpus"
        output_dir = tmp / "output"
        corpus_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        print(f"Building corpus in {corpus_dir} ...", flush=True)
        utt_idx_map, all_phones = _build_corpus(
            dataset, corpus_dir, sr=sr, use_phones=use_phones
        )

        if use_phones:
            dict_path = tmp / "phone_dict.txt"
            build_phone_dict([list(all_phones)], dict_path)
            effective_dictionary = str(dict_path)
        else:
            effective_dictionary = dictionary

        print("Running mfa align ...", flush=True)
        _run_align(
            corpus_dir, output_dir, effective_dictionary, acoustic_model,
            env=env,
        )

        print("Collecting results ...", flush=True)
        records = _collect_results(output_dir, utt_idx_map, dataset)

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
    print(f"Wrote {len(records)} utterances to {out_file}.", flush=True)


def main() -> None:
    """CLI entry point for MFA batch forced-alignment inference."""
    parser = argparse.ArgumentParser(
        description="MFA batch forced-alignment inference → JSONL"
    )
    parser.add_argument("--hf_repo", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--dictionary", default="english_mfa")
    parser.add_argument("--acoustic_model", default="english_mfa")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--cache_dir", default="exp/cache/hf")
    parser.add_argument("--mfa_cache_dir", default=None)
    parser.add_argument(
        "--temp_dir",
        default=None,
        help="Keep intermediate corpus/output here instead of a temp dir.",
    )
    parser.add_argument(
        "--no_use_phones",
        dest="use_phones",
        action="store_false",
        help="Use word transcripts + named dictionary instead of phones.",
    )
    parser.set_defaults(use_phones=True)
    args = parser.parse_args()
    run_mfa_batch_inference(
        hf_repo=args.hf_repo,
        split=args.split,
        out_file=args.out_file,
        dictionary=args.dictionary,
        acoustic_model=args.acoustic_model,
        sr=args.sr,
        cache_dir=args.cache_dir,
        mfa_cache_dir=args.mfa_cache_dir,
        temp_dir=args.temp_dir,
        use_phones=args.use_phones,
    )


if __name__ == "__main__":
    main()
