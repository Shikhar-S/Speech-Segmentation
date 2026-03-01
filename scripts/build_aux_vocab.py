"""Build a custom SentencePiece vocabulary from training data ASR text.

Usage:
    python scripts/build_aux_vocab.py \\
        --dataset_config configs/data/ipapack_index.yaml \\
        --splits train_accentmix_multi \\
        --vocab_size 4000 \\
        --out exp/aux_vocab/bpe4k

Outputs: <out>.model and <out>.vocab
"""

import argparse
import tempfile
from pathlib import Path

import yaml


def extract_asr_text(lang_file: str, out_f) -> int:
    """Extract orthographic ASR text from a language file and write to out_f.

    Lines with keys ending in '_asr' contain orthographic text formatted as:
        <utt_id>_asr <tag> <word1> <word2> ...
    The tag (e.g. '<cmn><asr><notimestamps>') is skipped; remaining fields are written.

    Returns number of lines written.
    """
    count = 0
    with open(lang_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            if not parts[0].endswith("_asr"):
                continue
            # parts[1] is the tag token; parts[2:] is the actual text
            text = " ".join(parts[2:])
            if text:
                out_f.write(text + "\n")
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Build SentencePiece vocab from training ASR text."
    )
    parser.add_argument(
        "--dataset_config",
        default="configs/data/ipapack_index.yaml",
        help="Path to ipapack_index.yaml dataset config.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        help="One or more training split names from the dataset config.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=4000,
        help="SentencePiece vocabulary size.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path prefix (e.g. exp/aux_vocab/bpe4k). Writes .model and .vocab.",
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=0.9995,
        help="Character coverage for SentencePiece training (default 0.9995 for multilingual).",
    )
    args = parser.parse_args()

    with open(args.dataset_config) as f:
        config = yaml.safe_load(f)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Collect all ASR text into a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
        total_lines = 0
        for split in args.splits:
            if split not in config["datasets"]:
                raise ValueError(
                    f"Split '{split}' not found in {args.dataset_config}. "
                    f"Available: {list(config['datasets'].keys())}"
                )
            lang_file = config["datasets"][split]["language"]
            print(f"Extracting ASR text from split '{split}': {lang_file}")
            n = extract_asr_text(lang_file, tmp)
            print(f"  {n} sentences written")
            total_lines += n

    print(f"\nTotal sentences: {total_lines}")
    print(f"Training SentencePiece model (vocab_size={args.vocab_size}, model_type=unigram)...")

    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=str(out_prefix),
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=args.character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    Path(tmp_path).unlink()
    print(f"\nDone.")
    print(f"  Model : {out_prefix}.model")
    print(f"  Vocab : {out_prefix}.vocab")


if __name__ == "__main__":
    main()
