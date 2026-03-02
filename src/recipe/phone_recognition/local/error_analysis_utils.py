"""
error_analysis_utils.py
=======================
Utility functions for phonetic error analysis of PER outputs.

Analyzes Phone Error Rate outputs to decompose errors into:
  1. Accent Deafness — model collapses accent variation
  2. Diacritic / Style Gap — narrow vs broad transcription mismatch
  3. Systematic Substitution Patterns — convention vs genuine confusion
  4. Articulatory Feature Error Rates (FER) — per-feature breakdown
  5. Annotation Inconsistency — within-language reference variation
  6. Per-phone accuracy — best/worst recognized phones

No main(), no argparse, no load_inputs().  Import this module and call functions directly.
"""

import json
import re
import string
import unicodedata
import yaml
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import panphon
    from panphon.distance import Distance

    HAS_PANPHON = True
except ImportError:
    HAS_PANPHON = False

try:
    from iso639 import Lang as _IsoLang

    HAS_ISO639 = True
except ImportError:
    HAS_ISO639 = False


# ===========================================================================
# Section 1 — IPA / Phonetics Utilities
# ===========================================================================

COMBINING_RANGE = (
    set(range(0x0300, 0x0370)) | set(range(0x1AB0, 0x1B00)) | set(range(0x1DC0, 0x1E00))
)
MODIFIER_LETTERS = {
    "\u02d0",
    "\u02d1",
    "\u02b0",
    "\u02b2",
    "\u02b7",
    "\u02e0",
    "\u02e4",
    "\u0306",
    "\u02de",
    "\u0303",
    "\u0325",
    "\u032a",
    "\u0329",
    "\u031d",
    "\u031e",
    "\u0339",
    "\u031c",
    "\u0324",
    "\u0330",
    "\u0334",
    "\u032f",
    "\u0320",
    "\u0308",
    "\u033d",
    "\u02bc",
    "\u0318",
    "\u0319",
}
TIE_BARS = {"\u0361", "\u035c"}

DIACRITIC_NAMES = {
    "\u02d0": "Length (ː)",
    "\u02d1": "Half-length (ˑ)",
    "\u0325": "Voicelessness (̥)",
    "\u032a": "Dental (̪)",
    "\u0306": "Extra-short (˘)",
    "\u0329": "Syllabic (̩)",
    "\u031d": "Raised (̝)",
    "\u031e": "Lowered (̞)",
    "\u02b2": "Palatalized (ʲ)",
    "\u02e0": "Velarized (ˠ)",
    "\u02e4": "Pharyngealized (ˤ)",
    "\u02b0": "Aspiration (ʰ)",
    "\u02de": "Rhoticity (˞)",
    "\u0303": "Nasalization (̃)",
    "\u0324": "Breathy (̤)",
    "\u0330": "Creaky (̰)",
    "\u0334": "Velarized/dark (̴)",
    "\u032f": "Non-syllabic (̯)",
    "\u0320": "Retracted (̠)",
    "\u0339": "More rounded (̹)",
    "\u031c": "Less rounded (̜)",
    "\u0318": "ATR (̘)",
    "\u0319": "RTR (̙)",
    "\u0308": "Centralized (̈)",
    "\u033d": "Mid-centralized (̽)",
    "\u02bc": "Ejective (ʼ)",
}

FEATURE_GROUPS = {
    "Voicing": ["voi"],
    "Place": ["ant", "cor", "distr", "lab"],
    "Manner": ["cont", "delrel", "lat", "nas", "strid"],
    "Height": ["hi", "lo"],
    "Backness": ["back"],
    "Rounding": ["round"],
    "Nasality": ["nas"],
    "Syllabicity": ["syl"],
    "Sonorancy": ["son", "cons"],
}


def is_diacritic_char(char: str) -> bool:
    """Return True if char is an IPA diacritic (combining mark or modifier letter)."""
    cp = ord(char)
    return (
        unicodedata.category(char).startswith("M")
        or cp in COMBINING_RANGE
        or char in MODIFIER_LETTERS
    )


def strip_diacritics(phone: str) -> str:
    """Remove all diacritic characters from a phone string."""
    return "".join(c for c in phone if not is_diacritic_char(c))


def count_diacritics(text: str) -> Counter:
    """Count named diacritics in an IPA string."""
    counts = Counter()
    for char in text:
        if is_diacritic_char(char):
            name = DIACRITIC_NAMES.get(
                char, f"U+{ord(char):04X} ({unicodedata.name(char, '?')})"
            )
            counts[name] += 1
    return counts


def segment_ipa(text: str) -> list:
    """Segment IPA string into phones (base + diacritics). Uses panphon if available."""
    if not text:
        return []
    if HAS_PANPHON:
        ft = panphon.FeatureTable()
        try:
            segs = ft.ipa_segs(text)
            if segs:
                return segs
        except Exception:
            pass
    segments, current = [], ""
    for char in text:
        if is_diacritic_char(char) or char in TIE_BARS:
            current += char
        else:
            if current:
                segments.append(current)
            current = char
    if current:
        segments.append(current)
    return segments


def parse_predicted_transcript(pred_str: str) -> list:
    """Split a slash-delimited predicted transcript into a phone list."""
    return [p for p in pred_str.split("/") if p and p != "▁"]


def lang_name(code: str) -> str:
    """Resolve an ISO 639-3 code to its English name; falls back to the code itself."""
    if HAS_ISO639:
        try:
            return _IsoLang(code).name
        except Exception:
            pass
    return code


def clean_ipa(text: str) -> str:
    """Normalize IPA text: remove spaces/punct, NFD-normalize, fix Latin g → IPA ɡ.

    Mirrors PhoneRecognitionEvaluator.clean_text.
    """
    text = text.replace(" ", "").translate(str.maketrans("", "", string.punctuation))
    text = unicodedata.normalize("NFD", text)
    return text.replace("g", "ɡ").strip()


# ===========================================================================
# Section 2 — Data Loading
# ===========================================================================

# Project root: src/recipe/phone_recognition/local/ → up 4 levels
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Module-level path constants (absolute, anchored to project root via __file__)
IPAPACK_YAML = _PROJECT_ROOT / "configs/data/ipapack_index.yaml"
RUNS_DIR = _PROJECT_ROOT / "exp/runs"

DEFAULT_TRAIN_SPLITS = {
    "train",
    "train_accentmix_multi",
    "train_accentmix_en",
    "huper_librispeech",
}

DATASETS = {
    "voxangeles": "decodedv3.voxangeles",
    "doreco": "decodedv3.doreco",
    "tusom": "decodedv3.tusom2021",
    "buckeye": "decodedv3.buckeye",
}

# Audio path patterns (read-only locations)
VA_AUDIO_PATTERN = (
    "/work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_voxangeles"
    "/recording/{langcode}/{utt_id}.wav"
)
TUSOM_AUDIO_PATTERN = (
    "/work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_tusom2021/data/wav/{utt_id}.wav"
)

_TASK_TOKENS = {"pr", "asr", "g2p", "p2g", "notimestamps"}


# Named helpers (not lambdas) so they are pickling-safe
def _va_lang(p: dict) -> str:
    return p["utt_id"].split("-")[0]


def _sym_lang(p: dict) -> str:
    return p.get("lang_sym", "unk")


def _buckeye_lang(p: dict) -> str:
    return p.get("lang_sym", "eng")


LANG_FN: dict = {
    "voxangeles": _va_lang,
    "doreco": _sym_lang,
    "tusom": _sym_lang,
    "buckeye": _buckeye_lang,
}


def _extract_langs_from_text_file(path: str) -> set:
    """Read a Kaldi text file and return unique language tags (ISO 639-3)."""
    langs = set()
    with open(path) as f:
        for line in f:
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            tags = re.findall(r"<([a-z0-9]+)>", parts[1])
            for tag in tags:
                if tag not in _TASK_TOKENS:
                    langs.add(tag)
                    break
    return langs


def load_train_langs(
    ipapack_yaml: Path = IPAPACK_YAML,
    train_splits: set = DEFAULT_TRAIN_SPLITS,
) -> list:
    """Return sorted list of ISO 639-3 training language codes from ipapack YAML."""
    with open(ipapack_yaml) as f:
        ipapack = yaml.safe_load(f)
    langs: set = set()
    for split_name, split_cfg in ipapack["datasets"].items():
        if split_name not in train_splits:
            continue
        lang_file = split_cfg.get("language")
        if lang_file is None:
            continue
        print(f"  Reading {split_name}: {lang_file}")
        langs |= _extract_langs_from_text_file(lang_file)
    result = sorted(langs)
    print(f"\nTotal training languages: {len(result)}")
    return result


def load_jsonl(filepath: str) -> list:
    """Load entries from a single JSONL file in the standard decode output format."""
    entries = []
    with open(filepath) as f:
        for line_num, line in tqdm(enumerate(f)):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: Skipping malformed JSON at line {line_num + 1}")
                continue
            for idx_key, entry in obj.items():
                pred_info = entry.get("pred", [{}])[0]
                passthrough = entry.get("passthrough", {})
                pred_transcript = pred_info.get("predicted_transcript", "")
                pred_phones = (
                    parse_predicted_transcript(pred_transcript) if pred_transcript else []
                )
                ref_str = passthrough.get("target", "")
                ref_phones = segment_ipa(ref_str)
                entries.append(
                    {
                        "idx": idx_key,
                        "utt_id": passthrough.get("utt_id", f"line{line_num}_{idx_key}"),
                        "lang": passthrough.get("lang_sym", "unk"),
                        "split": passthrough.get("split", ""),
                        "ref_str": ref_str,
                        "pred_str": pred_info.get("processed_transcript", ""),
                        "ref_phones": ref_phones,
                        "pred_phones": pred_phones,
                    }
                )
    return entries


def load_jsonl_shards(model_dir: Path) -> dict:
    """Merge all transcription.*.jsonl shards in model_dir into a single dict."""
    records = {}
    for shard in sorted(model_dir.glob("transcription.*.jsonl")):
        with open(shard) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.update(obj)
    return records


def load_dataset_predictions(
    dataset: str,
    models: list,
    gt_field: str = "target",
    pred_field: str = "processed_transcript",
    key_field: str = "utt_id",
    lang_fn: Optional[Callable] = None,
) -> dict:
    """Load decoded outputs for a dataset from decodedv3 JSONL files.

    Returns
    -------
    dict
        {model_name -> {lang -> {utt_id -> {"prediction": str, "transcription": str}}}}
    """
    dataset_dir = RUNS_DIR / DATASETS[dataset]
    if lang_fn is None:
        lang_fn = LANG_FN[dataset]

    result = {}
    for model_name in models:
        model_dir = dataset_dir / model_name
        if not model_dir.exists():
            print(f"[warn] {model_dir} does not exist, skipping")
            continue
        records = load_jsonl_shards(model_dir)
        by_lang: dict = defaultdict(dict)
        for _, item in records.items():
            passthrough = item["passthrough"]
            utt_id = passthrough[key_field]
            lang = lang_fn(passthrough)
            prediction = item["pred"][0][pred_field]
            transcription = passthrough[gt_field]
            by_lang[lang][utt_id] = {"prediction": prediction, "transcription": transcription}

        result[model_name] = dict(by_lang)
        n_utts = sum(len(v) for v in by_lang.values())
        print(f"  {model_name}: {n_utts} utts across {len(by_lang)} lang(s)")
    return result


# ===========================================================================
# Section 3 — Alignment & Confusion
# ===========================================================================

DEL_SYM = "<DEL>"
INS_SYM = "<INS>"


def align_phones(ref_segs: list, hyp_segs: list) -> list:
    """Levenshtein alignment with backtrace.

    Returns
    -------
    list of (op, ref_phone, hyp_phone) where op ∈ {"C", "S", "D", "I"}.
    Deletions have hyp_phone=None; insertions have ref_phone=None.
    Uses pure-Python DP (no numpy).
    """
    n, m = len(ref_segs), len(hyp_segs)
    # Build DP table as list-of-lists
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_segs[i - 1] == hyp_segs[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + cost)

    # Backtrace
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_segs[i - 1] == hyp_segs[j - 1]:
            alignment.append(("C", ref_segs[i - 1], hyp_segs[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            alignment.append(("S", ref_segs[i - 1], hyp_segs[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            alignment.append(("D", ref_segs[i - 1], None))
            i -= 1
        else:
            alignment.append(("I", None, hyp_segs[j - 1]))
            j -= 1
    alignment.reverse()
    return alignment


def normalized_edit_distance(seq1: list, seq2: list) -> float:
    """Normalized edit distance between two phone sequences (∈ [0, 1])."""
    if not seq1 and not seq2:
        return 0.0
    alignment = align_phones(seq1, seq2)
    errors = sum(1 for op, _, _ in alignment if op != "C")
    total = max(len(seq1), len(seq2))
    return errors / total if total > 0 else 0.0


def phone_confusion_matrix(
    df: pd.DataFrame,
    evaluator,
    ref_col: str = "reference",
    hyp_col: str = "predicted",
    include_del: bool = True,
    include_ins: bool = True,
) -> pd.DataFrame:
    """Compute phone-level confusion counts from a per-utterance DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ref_col and hyp_col with IPA strings.
    evaluator : PhoneRecognitionEvaluator
        Used for text normalization and panphon segmentation.
    include_del : bool
        Include deletion pairs (ref → DEL_SYM) in the output.
    include_ins : bool
        Include insertion pairs (INS_SYM → hyp) in the output.

    Returns
    -------
    pd.DataFrame  columns: ref, hyp, count — sorted by count descending.
    """
    counter: Counter = Counter()
    for _, row in df.iterrows():
        ref = evaluator._prepare(row[ref_col])
        hyp = evaluator._prepare(row[hyp_col])
        ref_segs = evaluator.dst.fm.ipa_segs(ref)
        hyp_segs = evaluator.dst.fm.ipa_segs(hyp)
        for op, ref_p, hyp_p in align_phones(ref_segs, hyp_segs):
            if op == "C":
                continue
            if op == "I":
                if not include_ins:
                    continue
                counter[(INS_SYM, hyp_p)] += 1
            elif op == "D":
                if not include_del:
                    continue
                counter[(ref_p, DEL_SYM)] += 1
            else:  # "S"
                counter[(ref_p, hyp_p)] += 1

    rows = [{"ref": r, "hyp": h, "count": c} for (r, h), c in counter.most_common()]
    return pd.DataFrame(rows)


# ===========================================================================
# Section 4 — Metrics Computation
# ===========================================================================


def compute_metrics(
    dataset_name: str,
    preds: dict,
    compute_inventory: bool = False,
) -> pd.DataFrame:
    """Compute phone recognition metrics for all models and languages in preds.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (used as a column in the output DataFrame).
    preds : dict
        {model_name -> {lang_sym -> {utt_id -> {"prediction": str, "transcription": str}}}}

    Returns
    -------
    pd.DataFrame
        One row per (model, language) with columns: dataset, model, language,
        N, phones, PER, FER, FED, PFER, SUB, INS, DEL.
        Also includes a "macro" row per model averaging over languages (if >1 lang).
    """
    from src.metrics.phone_recognition import PhoneRecognitionEvaluator

    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
    rows = []

    for model_name, lang_data in tqdm(preds.items(), desc=f"{dataset_name} models"):
        per_lang_summaries = []
        for lang, utt_data in tqdm(lang_data.items(), desc=model_name, leave=False):
            summary, _ = evaluator.evaluate(
                utt_data,
                compute_inventory=compute_inventory,
                tqdm_enabled=False,
            )
            per_lang_summaries.append(summary)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "language": lang,
                    "N": summary.N,
                    "phones": summary.phones,
                    "PER": summary.PER,
                    "FER": summary.FER,
                    "FED": summary.FED,
                    "PFER": summary.PFER,
                    "SUB": summary.SUB,
                    "INS": summary.INS,
                    "DEL": summary.DEL,
                }
            )

        if len(per_lang_summaries) > 1:
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "language": "__macro__",
                    "N": sum(s.N for s in per_lang_summaries),
                    "phones": sum(s.phones for s in per_lang_summaries),
                    "PER": sum(s.PER for s in per_lang_summaries) / len(per_lang_summaries),
                    "FER": sum(s.FER for s in per_lang_summaries) / len(per_lang_summaries),
                    "FED": sum(s.FED for s in per_lang_summaries),
                    "PFER": sum(s.PFER for s in per_lang_summaries) / len(per_lang_summaries),
                    "SUB": sum(s.SUB for s in per_lang_summaries) / len(per_lang_summaries),
                    "INS": sum(s.INS for s in per_lang_summaries) / len(per_lang_summaries),
                    "DEL": sum(s.DEL for s in per_lang_summaries) / len(per_lang_summaries),
                }
            )

    return pd.DataFrame(rows)


# ===========================================================================
# Section 5 — Per-Utterance DataFrame Construction
# ===========================================================================


def audio_duration(utt_id: str, pattern: str) -> Optional[float]:
    """Return audio duration in seconds for utt_id, or None on failure.

    The pattern may contain {utt_id} and/or {langcode} (extracted from utt_id
    as the portion before the first '-').
    """
    if not HAS_SOUNDFILE:
        return None
    try:
        langcode = utt_id.split("-")[0]
        path = pattern.format(utt_id=utt_id, langcode=langcode)
        return sf.info(path).duration
    except Exception:
        return None


def build_utt_dataframe(
    preds: dict,
    evaluator,
    audio_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Build a per-utterance DataFrame with metrics and optional audio duration.

    Parameters
    ----------
    preds : dict
        {model_name -> {lang -> {utt_id -> {"prediction": str, "transcription": str}}}}
    evaluator : PhoneRecognitionEvaluator
        Used to compute per-utterance metrics and phone segmentation.
    audio_fn : callable or None
        If provided, called as audio_fn(utt_id) -> float | None to add a
        "duration" column.  When None, "duration" is filled with None values.

    Returns
    -------
    pd.DataFrame
        Columns: utt_id, langcode, langname, reference, predicted, pfer, fer,
        fed, per, n_phones, duration.
    """
    rows = []
    for _model_name, lang_data in preds.items():
        for lang, utt_data in lang_data.items():
            _, instance_metrics = evaluator.evaluate(
                utt_data, compute_inventory=False, tqdm_enabled=False
            )
            name = lang_name(lang)
            for utt_id, metrics in instance_metrics.items():
                ref = utt_data[utt_id]["transcription"]
                n_phones = len(
                    evaluator.dst.fm.ipa_segs(evaluator._prepare(ref))
                )
                rows.append(
                    {
                        "utt_id": utt_id,
                        "langcode": lang,
                        "langname": name,
                        "reference": ref,
                        "predicted": utt_data[utt_id]["prediction"],
                        "pfer": metrics["pfer"],
                        "fer": metrics["fer"],
                        "fed": metrics["fed"],
                        "per": metrics["per"],
                        "n_phones": n_phones,
                        "duration": audio_fn(utt_id) if audio_fn is not None else None,
                    }
                )
    return pd.DataFrame(rows)


def df_to_entries(df: pd.DataFrame) -> list:
    """Convert a per-utterance DataFrame to the entries list format used by analysis functions.

    Segments reference and prediction via clean_ipa + segment_ipa.
    Columns mapped: reference→ref_str, predicted→pred_str, langcode→lang.
    """
    entries = []
    for _, row in df.iterrows():
        ref_clean = clean_ipa(str(row["reference"]))
        pred_clean = clean_ipa(str(row.get("predicted", "")))
        entries.append(
            {
                "idx": row.get("idx", ""),
                "utt_id": row["utt_id"],
                "lang": row.get("langcode", "unk"),
                "split": row.get("split", ""),
                "ref_str": row["reference"],
                "pred_str": row.get("predicted", ""),
                "ref_phones": segment_ipa(ref_clean),
                "pred_phones": segment_ipa(pred_clean),
            }
        )
    return entries


# ===========================================================================
# Section 6 — Error Analysis
# ===========================================================================


def analyze_accent_deafness(entries: list) -> dict:
    """Measure how much the model collapses accent variation relative to reference."""
    if len(entries) < 2:
        return {"error": "Need at least 2 utterances"}
    n = len(entries)
    ref_dists, pred_dists = [], []
    for i, j in combinations(range(n), 2):
        ref_dists.append(
            normalized_edit_distance(entries[i]["ref_phones"], entries[j]["ref_phones"])
        )
        pred_dists.append(
            normalized_edit_distance(
                entries[i]["pred_phones"], entries[j]["pred_phones"]
            )
        )
    avg_ref = float(np.mean(ref_dists))
    avg_pred = float(np.mean(pred_dists))
    ratio = avg_ref / avg_pred if avg_pred > 0 else float("inf")

    lang_groups = defaultdict(list)
    for e in entries:
        lang_groups[e["lang"]].append(e)
    within_lang = {}
    for lang, group in lang_groups.items():
        if len(group) < 2:
            continue
        lr, lp = [], []
        for i, j in combinations(range(len(group)), 2):
            lr.append(
                normalized_edit_distance(group[i]["ref_phones"], group[j]["ref_phones"])
            )
            lp.append(
                normalized_edit_distance(
                    group[i]["pred_phones"], group[j]["pred_phones"]
                )
            )
        ar, ap = float(np.mean(lr)), float(np.mean(lp))
        within_lang[lang] = {
            "n_speakers": len(group),
            "ref_variation": round(ar, 3),
            "pred_variation": round(ap, 3),
            "ratio": round(ar / ap, 1) if ap > 0 else float("inf"),
        }
    return {
        "avg_ref_distance": round(avg_ref, 3),
        "avg_pred_distance": round(avg_pred, 3),
        "accent_deafness_ratio": round(ratio, 1),
        "n_pairs": len(ref_dists),
        "within_language": within_lang,
    }


def analyze_diacritic_gap(entries: list) -> dict:
    """Quantify the gap between reference and predicted diacritic usage."""
    ref_counts, pred_counts = Counter(), Counter()
    for e in entries:
        ref_counts += count_diacritics(e["ref_str"])
        pred_counts += count_diacritics(e["pred_str"])
    all_d = sorted(set(ref_counts.keys()) | set(pred_counts.keys()))
    gap_table = [
        {
            "diacritic": d,
            "in_ref": ref_counts.get(d, 0),
            "in_pred": pred_counts.get(d, 0),
            "gap": ref_counts.get(d, 0) - pred_counts.get(d, 0),
        }
        for d in all_d
    ]
    gap_table.sort(key=lambda x: abs(x["gap"]), reverse=True)
    return {
        "gap_table": gap_table,
        "ref_only_diacritics": [
            d for d in all_d if pred_counts.get(d, 0) == 0 and ref_counts.get(d, 0) > 0
        ],
        "pred_only_diacritics": [
            d for d in all_d if ref_counts.get(d, 0) == 0 and pred_counts.get(d, 0) > 0
        ],
        "total_ref_diacritics": sum(ref_counts.values()),
        "total_pred_diacritics": sum(pred_counts.values()),
    }


def analyze_substitutions(entries: list) -> dict:
    """Classify substitution errors by articulatory distance."""
    sub_counts, del_counts, ins_counts = Counter(), Counter(), Counter()
    correct_counts, total_ref_counts, total_ops = Counter(), Counter(), Counter()

    for entry in entries:
        for op, ref_p, hyp_p in align_phones(entry["ref_phones"], entry["pred_phones"]):
            total_ops[op] += 1
            if op == "C":
                correct_counts[ref_p] += 1
                total_ref_counts[ref_p] += 1
            elif op == "S":
                sub_counts[(ref_p, hyp_p)] += 1
                total_ref_counts[ref_p] += 1
            elif op == "D":
                del_counts[ref_p] += 1
                total_ref_counts[ref_p] += 1
            elif op == "I":
                ins_counts[hyp_p] += 1

    dst = Distance() if HAS_PANPHON else None
    classified_subs = []
    for (ref_p, hyp_p), count in sub_counts.most_common():
        dist = -1
        if dst:
            try:
                dist = dst.feature_edit_distance(ref_p, hyp_p)
            except Exception:
                dist = -1
        ref_base, hyp_base = strip_diacritics(ref_p), strip_diacritics(hyp_p)
        if ref_base == hyp_base and ref_p != hyp_p:
            sub_type = "diacritic_only"
        elif dist >= 0 and dist < 0.1:
            sub_type = "very_close (likely convention)"
        elif dist >= 0 and dist < 0.25:
            sub_type = "close"
        elif dist >= 0:
            sub_type = "distant (likely acoustic)"
        else:
            sub_type = "unknown"
        classified_subs.append(
            {
                "ref": ref_p,
                "pred": hyp_p,
                "count": count,
                "articulatory_distance": round(dist, 3) if dist >= 0 else None,
                "type": sub_type,
            }
        )

    phone_confusions = defaultdict(Counter)
    for (ref_p, hyp_p), count in sub_counts.items():
        phone_confusions[ref_p][hyp_p] += count

    phone_accuracy = {}
    for phone in total_ref_counts:
        total = total_ref_counts[phone]
        correct = correct_counts.get(phone, 0)
        top_conf = None
        if phone_confusions[phone]:
            top = phone_confusions[phone].most_common(1)[0]
            top_conf = f"→{top[0]} ({top[1]}×)"
        phone_accuracy[phone] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
            "top_confusion": top_conf,
        }

    best = sorted(phone_accuracy.items(), key=lambda x: (-x[1]["accuracy"], -x[1]["total"]))
    worst = sorted(phone_accuracy.items(), key=lambda x: (x[1]["accuracy"], -x[1]["total"]))
    total_ref = total_ops["C"] + total_ops["S"] + total_ops["D"]
    total_errors = total_ops["S"] + total_ops["D"] + total_ops["I"]
    per = total_errors / total_ref * 100 if total_ref > 0 else 0

    return {
        "per": round(per, 1),
        "total_ref_phones": total_ref,
        "operation_counts": dict(total_ops),
        "substitutions": classified_subs[:50],
        "deletions": del_counts.most_common(20),
        "insertions": ins_counts.most_common(20),
        "best_phones": [(p, i) for p, i in best[:15] if i["total"] >= 5],
        "worst_phones": [(p, i) for p, i in worst[:15] if i["total"] >= 5],
        "phone_accuracy": phone_accuracy,
    }


def analyze_feature_errors(entries: list) -> dict:
    """Compute articulatory feature error rates (FER) per feature and feature group."""
    if not HAS_PANPHON:
        return {"error": "panphon not available"}
    ft = panphon.FeatureTable()
    feature_names = ft.names
    feature_errors = {f: 0 for f in feature_names}
    feature_totals = {f: 0 for f in feature_names}
    n_aligned, n_skipped = 0, 0

    for entry in entries:
        for op, ref_p, hyp_p in align_phones(entry["ref_phones"], entry["pred_phones"]):
            if op == "S":
                try:
                    rf = ft.fts(ref_p)
                    hf = ft.fts(hyp_p)
                    if rf is None or hf is None:
                        n_skipped += 1
                        continue
                    rv, hv = rf.numeric(), hf.numeric()
                except Exception:
                    n_skipped += 1
                    continue
                n_aligned += 1
                for i, fn in enumerate(feature_names):
                    if rv[i] != 0 or hv[i] != 0:
                        feature_totals[fn] += 1
                        if rv[i] != hv[i]:
                            feature_errors[fn] += 1
            elif op == "C":
                try:
                    rf = ft.fts(ref_p)
                    if rf is None:
                        continue
                    rv = rf.numeric()
                except Exception:
                    continue
                for i, fn in enumerate(feature_names):
                    if rv[i] != 0:
                        feature_totals[fn] += 1

    feature_fer = {}
    for fn in feature_names:
        t, e = feature_totals[fn], feature_errors[fn]
        feature_fer[fn] = {
            "fer": round(e / t, 3) if t > 0 else 0,
            "errors": e,
            "total": t,
        }

    group_fer = {}
    for group, features in FEATURE_GROUPS.items():
        if not features:
            continue
        ge = sum(feature_errors.get(f, 0) for f in features)
        gt = sum(feature_totals.get(f, 0) for f in features)
        group_fer[group] = {
            "fer": round(ge / gt, 3) if gt > 0 else 0,
            "errors": ge,
            "total": gt,
        }

    return {
        "per_feature": dict(sorted(feature_fer.items(), key=lambda x: -x[1]["fer"])),
        "per_group": dict(sorted(group_fer.items(), key=lambda x: -x[1]["fer"])),
        "n_aligned_substitutions": n_aligned,
        "n_skipped": n_skipped,
    }


def analyze_annotation_consistency(entries: list) -> dict:
    """Measure within-language reference variation as a proxy for annotation consistency."""
    lang_groups = defaultdict(list)
    for e in entries:
        lang_groups[e["lang"]].append(e)
    results = {}
    for lang, group in lang_groups.items():
        if len(group) < 2:
            continue
        ref_dists = [
            normalized_edit_distance(group[i]["ref_phones"], group[j]["ref_phones"])
            for i, j in combinations(range(len(group)), 2)
        ]
        inventories = [set(e["ref_phones"]) for e in group]
        jaccard_sims = []
        for i, j in combinations(range(len(inventories)), 2):
            inter = len(inventories[i] & inventories[j])
            union = len(inventories[i] | inventories[j])
            if union > 0:
                jaccard_sims.append(inter / union)
        all_phones = set()
        for inv in inventories:
            all_phones.update(inv)
        inconsistent = [
            {
                "phone": p,
                "present_in": f"{sum(1 for inv in inventories if p in inv)}/{len(inventories)}",
            }
            for p in sorted(all_phones)
            if 0 < sum(1 for inv in inventories if p in inv) < len(inventories)
        ]
        results[lang] = {
            "n_utterances": len(group),
            "avg_ref_distance": round(float(np.mean(ref_dists)), 3) if ref_dists else 0,
            "std_ref_distance": round(float(np.std(ref_dists)), 3) if ref_dists else 0,
            "avg_inventory_jaccard": (
                round(float(np.mean(jaccard_sims)), 3) if jaccard_sims else 0
            ),
            "inconsistent_phones": inconsistent[:20],
        }
    return results


# ===========================================================================
# Section 7 — Report Generation
# ===========================================================================


def format_report(entries, accent, diacritics, substitutions, features, consistency) -> str:
    """Format analysis results as a Markdown report string."""
    L = []
    langs = set(e["lang"] for e in entries)
    L.append("# PER Decomposition Analysis Report\n")
    L.append("## Data Summary")
    L.append(f"- **Utterances**: {len(entries)}")
    L.append(f"- **Languages**: {len(langs)} ({', '.join(sorted(langs))})")
    L.append(f"- **Overall PER**: {substitutions.get('per', '?')}%")
    L.append(f"- **Total ref phones**: {substitutions.get('total_ref_phones', '?')}\n")

    # 1. Accent Deafness
    L.append("---\n## 1. Accent Deafness\n")
    if "error" not in accent and "skipped" not in accent:
        L.append("| Metric | Value |")
        L.append("|--------|-------|")
        L.append(
            f"| Avg normalized **prediction** distance | {accent['avg_pred_distance']} |"
        )
        L.append(
            f"| Avg normalized **reference** distance | {accent['avg_ref_distance']} |"
        )
        L.append(
            f"| **Accent deafness ratio** | **{accent['accent_deafness_ratio']}×** |\n"
        )
        w = accent.get("within_language", {})
        if w:
            L.append("### Within-language variation:\n")
            L.append("| Language | N | Ref Var | Pred Var | Ratio |")
            L.append("|----------|---|---------|----------|-------|")
            for lang, info in sorted(w.items(), key=lambda x: -x[1].get("ratio", 0)):
                L.append(
                    f"| {lang} | {info['n_speakers']} | {info['ref_variation']} |"
                    f" {info['pred_variation']} | **{info['ratio']}×** |"
                )
            L.append("")
    else:
        L.append("*Skipped or insufficient data.*\n")

    # 2. Diacritic Gap
    L.append("---\n## 2. Diacritic / Style Gap\n")
    L.append(
        f"Total diacritics — Ref: {diacritics['total_ref_diacritics']},"
        f" Pred: {diacritics['total_pred_diacritics']}\n"
    )
    gt = diacritics.get("gap_table", [])
    if gt:
        L.append("| Diacritic | In Ref | In Pred | Gap |")
        L.append("|-----------|--------|---------|-----|")
        for r in gt[:20]:
            L.append(
                f"| {r['diacritic']} | {r['in_ref']} | {r['in_pred']} |"
                f" {'+' if r['gap'] > 0 else ''}{r['gap']} |"
            )
        L.append("")
    ro = diacritics.get("ref_only_diacritics", [])
    if ro:
        L.append(f"**Diacritics in ref but never predicted**: {', '.join(ro)}\n")

    # 3. Substitution Patterns
    L.append("---\n## 3. Systematic Substitution Patterns\n")
    ops = substitutions.get("operation_counts", {})
    L.append(
        f"Operations: C={ops.get('C', 0)}, S={ops.get('S', 0)},"
        f" D={ops.get('D', 0)}, I={ops.get('I', 0)}\n"
    )
    subs = substitutions.get("substitutions", [])
    if subs:
        for label, filt in [
            (
                "Convention / style mismatches",
                lambda s: "convention" in s.get("type", "") or "diacritic" in s.get("type", ""),
            ),
            ("Close substitutions", lambda s: s.get("type") == "close"),
            (
                "Likely acoustic confusions",
                lambda s: "acoustic" in s.get("type", "") or "distant" in s.get("type", ""),
            ),
        ]:
            matched = [s for s in subs if filt(s)]
            if matched:
                L.append(f"### {label}:\n")
                L.append("| Ref → Pred | Count | Art. Dist | Type |")
                L.append("|------------|-------|-----------|------|")
                for s in matched[:15]:
                    L.append(
                        f"| {s['ref']} → {s['pred']} | {s['count']} |"
                        f" {s.get('articulatory_distance', '?')} | {s['type']} |"
                    )
                L.append("")

    # 4. Feature Error Rates
    L.append("---\n## 4. Articulatory Feature Error Rates\n")
    if "error" not in features:
        for label, key in [
            ("By feature group", "per_group"),
            ("By individual feature", "per_feature"),
        ]:
            data = features.get(key, {})
            if data:
                L.append(f"### {label}:\n")
                L.append("| Feature | FER | Errors | Total |")
                L.append("|---------|-----|--------|-------|")
                for fn, info in data.items():
                    if info["total"] > 0:
                        L.append(
                            f"| {fn} | {info['fer']} | {info['errors']} | {info['total']} |"
                        )
                L.append("")
    else:
        L.append(f"*{features['error']}*\n")

    # 5. Annotation Consistency
    L.append("---\n## 5. Annotation Consistency\n")
    if consistency:
        L.append(
            "| Language | N | Avg Ref Dist | Std | Jaccard | Inconsistent Phones |"
        )
        L.append("|----------|---|-------------|-----|---------|---------------------|")
        for lang, info in sorted(consistency.items()):
            ic = ", ".join(
                f"{p['phone']}({p['present_in']})"
                for p in info.get("inconsistent_phones", [])[:5]
            )
            L.append(
                f"| {lang} | {info['n_utterances']} | {info['avg_ref_distance']} |"
                f" {info['std_ref_distance']} | {info['avg_inventory_jaccard']} | {ic} |"
            )
        L.append("")
    else:
        L.append("*Not enough within-language samples.*\n")

    # 6. Per-Phone Accuracy
    L.append("---\n## 6. Per-Phone Accuracy\n")
    for label, key in [
        ("Best-recognized (≥5 occ)", "best_phones"),
        ("Worst-recognized (≥5 occ)", "worst_phones"),
    ]:
        phones = substitutions.get(key, [])
        if phones:
            L.append(f"### {label}:\n")
            L.append("| Phone | Accuracy | Top Confusion | Total |")
            L.append("|-------|----------|---------------|-------|")
            for phone, info in phones[:10]:
                L.append(
                    f"| {phone} | {info['accuracy']}% |"
                    f" {info.get('top_confusion', '-')} | {info['total']} |"
                )
            L.append("")

    return "\n".join(L)
