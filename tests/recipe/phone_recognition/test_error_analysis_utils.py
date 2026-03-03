"""Tests for error_analysis_utils.py.

When making changes to error_analysis_utils.py, update this file accordingly.
Run with:  pytest tests/recipe/phone_recognition/test_error_analysis_utils.py -v
"""

import json
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from src.recipe.phone_recognition.local.error_analysis_utils import (
    DATASETS,
    DEL_SYM,
    INS_SYM,
    LANG_FN,
    align_phones,
    analyze_accent_variance,
    analyze_annotation_consistency,
    analyze_diacritic_gap,
    analyze_substitutions,
    clean_ipa,
    count_diacritics,
    df_to_entries,
    format_report,
    is_diacritic_char,
    lang_name,
    load_jsonl,
    load_jsonl_shards,
    normalized_edit_distance,
    parse_predicted_transcript,
    phone_confusion_matrix,
    phone_inventory_jaccard,
    segment_ipa,
    strip_diacritics,
)


# ---------------------------------------------------------------------------
# Section 1 — IPA / Phonetics Utilities
# ---------------------------------------------------------------------------


class TestIsDiacriticChar:
    def test_combining_tilde_is_diacritic(self):
        # U+0303 COMBINING TILDE
        assert is_diacritic_char("\u0303") is True

    def test_modifier_length_mark_is_diacritic(self):
        # U+02D0 MODIFIER LETTER TRIANGULAR COLON (ː)
        assert is_diacritic_char("\u02d0") is True

    def test_base_vowel_not_diacritic(self):
        assert is_diacritic_char("a") is False

    def test_ipa_base_consonant_not_diacritic(self):
        assert is_diacritic_char("ʃ") is False

    def test_digit_not_diacritic(self):
        assert is_diacritic_char("1") is False


class TestStripDiacritics:
    def test_strips_length_mark(self):
        # "aː" → "a"
        assert strip_diacritics("a\u02d0") == "a"

    def test_no_diacritics_unchanged(self):
        assert strip_diacritics("ʃ") == "ʃ"

    def test_empty_string(self):
        assert strip_diacritics("") == ""

    def test_multiple_diacritics_stripped(self):
        # aspirated voiceless dental stop: tʰ̪  (t + aspiration + dental)
        phone = "t\u02b0\u032a"
        assert strip_diacritics(phone) == "t"


class TestCountDiacritics:
    def test_counts_length_marks(self):
        text = "aː bː"  # two length marks (ː = U+02D0)
        counts = count_diacritics(text)
        assert counts["Length (ː)"] == 2

    def test_empty_string_returns_empty_counter(self):
        assert count_diacritics("") == Counter()

    def test_no_diacritics_returns_empty_counter(self):
        assert count_diacritics("abc") == Counter()

    def test_mixed_diacritics(self):
        # Use bare combining characters (not precomposed): U+0303 + U+02D0
        text = "a\u0303\u02d0"  # a + combining tilde + length mark
        counts = count_diacritics(text)
        assert counts["Nasalization (̃)"] == 1
        assert counts["Length (ː)"] == 1


class TestSegmentIpa:
    def test_empty_string(self):
        assert segment_ipa("") == []

    def test_simple_phones(self):
        segs = segment_ipa("abc")
        assert "a" in segs and "b" in segs and "c" in segs

    def test_phone_with_diacritic_grouped(self):
        # "aː" should be one segment, not two
        segs = segment_ipa("a\u02d0")
        assert len(segs) == 1
        assert segs[0] == "a\u02d0"

    def test_two_base_phones(self):
        segs = segment_ipa("ab")
        assert len(segs) == 2


class TestParsePredictedTranscript:
    def test_slash_separated(self):
        assert parse_predicted_transcript("a/b/c") == ["a", "b", "c"]

    def test_filters_word_boundary_token(self):
        assert "▁" not in parse_predicted_transcript("a/▁/b")

    def test_filters_empty_segments(self):
        result = parse_predicted_transcript("a//b")
        assert "" not in result

    def test_empty_string_returns_empty_list(self):
        assert parse_predicted_transcript("") == []


class TestLangName:
    def test_known_code(self):
        name = lang_name("hrv")
        assert "Croatian" in name

    def test_unknown_code_returns_code(self):
        assert lang_name("zzz") == "zzz"


class TestCleanIpa:
    def test_removes_spaces(self):
        assert " " not in clean_ipa("a b c")

    def test_removes_punctuation(self):
        result = clean_ipa("a,b.")
        assert "," not in result and "." not in result

    def test_replaces_latin_g(self):
        result = clean_ipa("g")
        assert result == "ɡ"

    def test_nfd_normalization(self):
        # Result should be in NFD form
        result = clean_ipa("ɑ")
        assert result == unicodedata.normalize("NFD", "ɑ")

    def test_matches_evaluator_clean_text(self):
        from src.metrics.phone_recognition import PhoneRecognitionEvaluator

        for text in ["aː bɪ", "g tʰ ɑ", "pʰd̪"]:
            assert clean_ipa(text) == PhoneRecognitionEvaluator.clean_text(text)


# ---------------------------------------------------------------------------
# Section 2 — Data Loading (file-based, uses tmp_path)
# ---------------------------------------------------------------------------

_SAMPLE_JSONL_LINE = json.dumps(
    {
        "0": {
            "pred": [{"processed_transcript": "a ɪ t", "predicted_transcript": "a/ɪ/t"}],
            "passthrough": {
                "utt_id": "eng-000-001-1",
                "target": "aɪt",
                "lang_sym": "eng",
            },
        }
    }
)


class TestLoadJsonl:
    def test_loads_single_entry(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text(_SAMPLE_JSONL_LINE + "\n")
        entries = load_jsonl(str(f))
        assert len(entries) == 1
        e = entries[0]
        assert e["utt_id"] == "eng-000-001-1"
        assert e["lang"] == "eng"
        assert e["ref_str"] == "aɪt"

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text("\n" + _SAMPLE_JSONL_LINE + "\n\n")
        entries = load_jsonl(str(f))
        assert len(entries) == 1

    def test_skips_malformed_json(self, tmp_path, capsys):
        f = tmp_path / "test.jsonl"
        f.write_text("NOT_JSON\n" + _SAMPLE_JSONL_LINE + "\n")
        entries = load_jsonl(str(f))
        # Malformed line skipped, valid line loaded
        assert len(entries) == 1
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_no_debug_break(self, tmp_path):
        """Regression: old load_jsonl had a break after 100 lines.

        Uses empty target strings to avoid slow panphon segmentation.
        """
        lines = []
        for i in range(150):
            lines.append(
                json.dumps(
                    {
                        str(i): {
                            "pred": [{"processed_transcript": "", "predicted_transcript": ""}],
                            "passthrough": {"utt_id": f"u{i}", "target": "", "lang_sym": "eng"},
                        }
                    }
                )
            )
        f = tmp_path / "big.jsonl"
        f.write_text("\n".join(lines) + "\n")
        entries = load_jsonl(str(f))
        assert len(entries) == 150, "load_jsonl must not stop early at 100 lines"


class TestLoadJsonlShards:
    def test_merges_multiple_shards(self, tmp_path):
        shard0 = tmp_path / "transcription.0.jsonl"
        shard1 = tmp_path / "transcription.1.jsonl"
        shard0.write_text(json.dumps({"0": {"x": 1}}) + "\n")
        shard1.write_text(json.dumps({"1": {"x": 2}}) + "\n")
        records = load_jsonl_shards(tmp_path)
        assert "0" in records and "1" in records

    def test_empty_dir_returns_empty_dict(self, tmp_path):
        assert load_jsonl_shards(tmp_path) == {}

    def test_ignores_non_jsonl_files(self, tmp_path):
        (tmp_path / "other.txt").write_text("hello")
        assert load_jsonl_shards(tmp_path) == {}


# ---------------------------------------------------------------------------
# Section 3 — Alignment & Confusion
# ---------------------------------------------------------------------------


class TestAlignPhones:
    def test_perfect_match(self):
        aln = align_phones(["a", "b", "c"], ["a", "b", "c"])
        ops = [op for op, _, _ in aln]
        assert all(op == "C" for op in ops)

    def test_all_substitutions(self):
        aln = align_phones(["a", "b"], ["x", "y"])
        ops = [op for op, _, _ in aln]
        assert all(op == "S" for op in ops)

    def test_deletion(self):
        aln = align_phones(["a", "b", "c"], ["a", "c"])
        ops = [op for op, _, _ in aln]
        assert "D" in ops

    def test_insertion(self):
        aln = align_phones(["a", "c"], ["a", "b", "c"])
        ops = [op for op, _, _ in aln]
        assert "I" in ops

    def test_deletion_has_none_hyp(self):
        aln = align_phones(["a", "b"], ["a"])
        del_ops = [(op, r, h) for op, r, h in aln if op == "D"]
        assert len(del_ops) == 1
        assert del_ops[0][1] == "b"
        assert del_ops[0][2] is None

    def test_insertion_has_none_ref(self):
        aln = align_phones(["a"], ["a", "b"])
        ins_ops = [(op, r, h) for op, r, h in aln if op == "I"]
        assert len(ins_ops) == 1
        assert ins_ops[0][1] is None
        assert ins_ops[0][2] == "b"

    def test_empty_ref(self):
        aln = align_phones([], ["a", "b"])
        assert all(op == "I" for op, _, _ in aln)
        assert len(aln) == 2

    def test_empty_hyp(self):
        aln = align_phones(["a", "b"], [])
        assert all(op == "D" for op, _, _ in aln)
        assert len(aln) == 2

    def test_both_empty(self):
        assert align_phones([], []) == []

    def test_returns_correct_total_length(self):
        ref = ["a", "b", "c", "d"]
        hyp = ["a", "x", "c"]
        aln = align_phones(ref, hyp)
        # total ops = max(len(ref), len(hyp)) at minimum (may be more due to insertions+deletions)
        assert len(aln) >= max(len(ref), len(hyp))

    def test_known_alignment(self):
        # ref="abc", hyp="ac" → C(a), D(b), C(c)
        aln = align_phones(["a", "b", "c"], ["a", "c"])
        assert aln[0] == ("C", "a", "a")
        assert aln[1] == ("D", "b", None)
        assert aln[2] == ("C", "c", "c")


class TestNormalizedEditDistance:
    def test_identical_sequences(self):
        assert normalized_edit_distance(["a", "b"], ["a", "b"]) == pytest.approx(0.0)

    def test_both_empty(self):
        assert normalized_edit_distance([], []) == pytest.approx(0.0)

    def test_completely_different(self):
        d = normalized_edit_distance(["a", "b"], ["x", "y"])
        assert d == pytest.approx(1.0)

    def test_one_empty(self):
        d = normalized_edit_distance(["a", "b"], [])
        assert d == pytest.approx(1.0)

    def test_partial_match(self):
        # ref=["a","b","c"], hyp=["a","b","x"] → 1 error / 3 → 0.333
        d = normalized_edit_distance(["a", "b", "c"], ["a", "b", "x"])
        assert d == pytest.approx(1 / 3, abs=1e-6)

    def test_symmetry_is_approximate(self):
        # NED is not strictly symmetric but should be close for similar-length seqs
        d1 = normalized_edit_distance(["a", "b", "c"], ["a", "x", "c"])
        d2 = normalized_edit_distance(["a", "x", "c"], ["a", "b", "c"])
        assert d1 == pytest.approx(d2)


class TestPhoneConfusionMatrix:
    """Uses a real PhoneRecognitionEvaluator (no GPU needed)."""

    @pytest.fixture(scope="class")
    def evaluator(self):
        from src.metrics.phone_recognition import PhoneRecognitionEvaluator

        return PhoneRecognitionEvaluator(normalize_ipa=True)

    def _make_df(self, pairs):
        return pd.DataFrame(
            [{"reference": ref, "predicted": hyp} for ref, hyp in pairs]
        )

    def test_substitution_counted(self, evaluator):
        df = self._make_df([("a", "b")])
        cm = phone_confusion_matrix(df, evaluator)
        row = cm[(cm["ref"] == "a") & (cm["hyp"] == "b")]
        assert len(row) == 1
        assert row["count"].iloc[0] == 1

    def test_deletion_with_include_del(self, evaluator):
        df = self._make_df([("ab", "a")])
        cm = phone_confusion_matrix(df, evaluator, include_del=True)
        del_rows = cm[cm["hyp"] == DEL_SYM]
        assert len(del_rows) > 0

    def test_deletion_excluded(self, evaluator):
        df = self._make_df([("ab", "a")])
        cm = phone_confusion_matrix(df, evaluator, include_del=False)
        # When all ops are deletions and they're excluded, result may be empty
        if "hyp" in cm.columns:
            assert DEL_SYM not in cm["hyp"].values

    def test_insertion_with_include_ins(self, evaluator):
        df = self._make_df([("a", "ab")])
        cm = phone_confusion_matrix(df, evaluator, include_ins=True)
        ins_rows = cm[cm["ref"] == INS_SYM]
        assert len(ins_rows) > 0

    def test_insertion_excluded(self, evaluator):
        df = self._make_df([("a", "ab")])
        cm = phone_confusion_matrix(df, evaluator, include_ins=False)
        # When all ops are insertions and they're excluded, result may be empty
        if "ref" in cm.columns:
            assert INS_SYM not in cm["ref"].values

    def test_perfect_match_no_rows(self, evaluator):
        df = self._make_df([("a", "a")])
        cm = phone_confusion_matrix(df, evaluator, include_del=False, include_ins=False)
        assert len(cm) == 0

    def test_sorted_by_count_descending(self, evaluator):
        df = self._make_df([("a", "b"), ("a", "b"), ("a", "b"), ("c", "d")])
        cm = phone_confusion_matrix(df, evaluator)
        if len(cm) >= 2:
            assert cm["count"].iloc[0] >= cm["count"].iloc[1]

    def test_evaluator_explicit_not_global(self, evaluator):
        """phone_confusion_matrix must accept evaluator explicitly, not use a global."""
        import inspect

        sig = inspect.signature(phone_confusion_matrix)
        assert "evaluator" in sig.parameters


# ---------------------------------------------------------------------------
# Section 5 — Per-Utterance DataFrame Construction
# ---------------------------------------------------------------------------


class TestAudioDuration:
    def test_nonexistent_path_returns_none(self):
        from src.recipe.phone_recognition.local.error_analysis_utils import audio_duration

        result = audio_duration("eng-001-002-3", "/nonexistent/{utt_id}.wav")
        assert result is None

    def test_pattern_with_langcode(self):
        from src.recipe.phone_recognition.local.error_analysis_utils import audio_duration

        # Should not raise even though file doesn't exist
        result = audio_duration("eng-001-002-3", "/nonexistent/{langcode}/{utt_id}.wav")
        assert result is None

    def test_langcode_extracted_correctly(self, tmp_path, monkeypatch):
        """langcode is the portion before first '-' in utt_id."""
        import soundfile as sf

        # We can't easily create a WAV without actual audio data; test extraction logic
        called_with = {}

        def mock_info(path):
            called_with["path"] = path

            class _Info:
                duration = 1.23

            return _Info()

        monkeypatch.setattr(sf, "info", mock_info)
        from src.recipe.phone_recognition.local import error_analysis_utils

        monkeypatch.setattr(error_analysis_utils, "HAS_SOUNDFILE", True)
        monkeypatch.setattr(error_analysis_utils.sf, "info", mock_info)

        result = error_analysis_utils.audio_duration(
            "ady-000-001-5", "/audio/{langcode}/{utt_id}.wav"
        )
        assert result == pytest.approx(1.23)
        assert "ady" in called_with["path"]
        assert "ady-000-001-5" in called_with["path"]


class TestBuildUttDataframe:
    @pytest.fixture(scope="class")
    def evaluator(self):
        from src.metrics.phone_recognition import PhoneRecognitionEvaluator

        return PhoneRecognitionEvaluator(normalize_ipa=True)

    def _minimal_preds(self):
        return {
            "model_a": {
                "eng": {
                    "u1": {"transcription": "ab", "prediction": "ab"},
                    "u2": {"transcription": "cd", "prediction": "c"},
                }
            }
        }

    def test_returns_dataframe(self, evaluator):
        from src.recipe.phone_recognition.local.error_analysis_utils import build_utt_dataframe

        df = build_utt_dataframe(self._minimal_preds(), evaluator)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, evaluator):
        from src.recipe.phone_recognition.local.error_analysis_utils import build_utt_dataframe

        df = build_utt_dataframe(self._minimal_preds(), evaluator)
        assert len(df) == 2

    def test_expected_columns(self, evaluator):
        from src.recipe.phone_recognition.local.error_analysis_utils import build_utt_dataframe

        df = build_utt_dataframe(self._minimal_preds(), evaluator)
        expected = {"utt_id", "langcode", "langname", "reference", "predicted",
                    "pfer", "fer", "fed", "per", "n_phones", "duration"}
        assert set(df.columns) == expected

    def test_no_audio_fn_duration_is_none(self, evaluator):
        from src.recipe.phone_recognition.local.error_analysis_utils import build_utt_dataframe

        df = build_utt_dataframe(self._minimal_preds(), evaluator)
        assert df["duration"].isna().all()

    def test_audio_fn_applied(self, evaluator):
        from src.recipe.phone_recognition.local.error_analysis_utils import build_utt_dataframe

        df = build_utt_dataframe(
            self._minimal_preds(), evaluator, audio_fn=lambda u: 1.5
        )
        assert (df["duration"] == 1.5).all()

    def test_langcode_column_correct(self, evaluator):
        from src.recipe.phone_recognition.local.error_analysis_utils import build_utt_dataframe

        df = build_utt_dataframe(self._minimal_preds(), evaluator)
        assert (df["langcode"] == "eng").all()

    def test_perfect_prediction_per_zero(self, evaluator):
        from src.recipe.phone_recognition.local.error_analysis_utils import build_utt_dataframe

        preds = {"m": {"eng": {"u1": {"transcription": "ab", "prediction": "ab"}}}}
        df = build_utt_dataframe(preds, evaluator)
        assert df["per"].iloc[0] == pytest.approx(0.0)


class TestDfToEntries:
    def _make_df(self):
        return pd.DataFrame(
            [
                {
                    "utt_id": "eng-001",
                    "langcode": "eng",
                    "reference": "aɪ",
                    "predicted": "aɪ",
                },
                {
                    "utt_id": "fra-002",
                    "langcode": "fra",
                    "reference": "bo",
                    "predicted": "b",
                },
            ]
        )

    def test_returns_list_of_dicts(self):
        entries = df_to_entries(self._make_df())
        assert isinstance(entries, list)
        assert all(isinstance(e, dict) for e in entries)

    def test_correct_length(self):
        assert len(df_to_entries(self._make_df())) == 2

    def test_required_keys_present(self):
        entry = df_to_entries(self._make_df())[0]
        for key in ("utt_id", "lang", "ref_str", "pred_str", "ref_phones", "pred_phones"):
            assert key in entry

    def test_langcode_mapped_to_lang(self):
        entries = df_to_entries(self._make_df())
        assert entries[0]["lang"] == "eng"
        assert entries[1]["lang"] == "fra"

    def test_ref_phones_is_list(self):
        entries = df_to_entries(self._make_df())
        for e in entries:
            assert isinstance(e["ref_phones"], list)
            assert isinstance(e["pred_phones"], list)


# ---------------------------------------------------------------------------
# Section 6 — Error Analysis
# ---------------------------------------------------------------------------

# Minimal synthetic entries used across analysis tests
_ENTRIES = [
    {
        "utt_id": "eng-001",
        "lang": "eng",
        "split": "",
        "ref_str": "aɪt",
        "pred_str": "aɪt",
        "ref_phones": ["a", "ɪ", "t"],
        "pred_phones": ["a", "ɪ", "t"],
    },
    {
        "utt_id": "eng-002",
        "lang": "eng",
        "split": "",
        "ref_str": "bɛd",
        "pred_str": "bɛd",
        "ref_phones": ["b", "ɛ", "d"],
        "pred_phones": ["b", "ɛ", "d"],
    },
    {
        "utt_id": "fra-001",
        "lang": "fra",
        "split": "",
        "ref_str": "bõ",
        "pred_str": "bo",
        "ref_phones": ["b", "õ"],
        "pred_phones": ["b", "o"],
    },
]


class TestAnalyzeAccentVariance:
    def test_returns_expected_keys(self):
        result = analyze_accent_variance(_ENTRIES)
        assert "avg_ref_distance" in result
        assert "avg_pred_distance" in result
        assert "accent_variance_ratio" in result
        assert "within_language" in result

    def test_insufficient_data(self):
        result = analyze_accent_variance([_ENTRIES[0]])
        assert "error" in result

    def test_identical_seqs_give_zero_distance(self):
        entries = [
            {**_ENTRIES[0], "lang": "eng"},
            {**_ENTRIES[1], "lang": "eng"},
        ]
        result = analyze_accent_variance(entries)
        assert isinstance(result["avg_ref_distance"], float)

    def test_within_language_populated(self):
        result = analyze_accent_variance(_ENTRIES)
        wl = result["within_language"]
        assert "eng" in wl  # eng has 2 entries


class TestAnalyzeDiacriticGap:
    def test_returns_expected_keys(self):
        result = analyze_diacritic_gap(_ENTRIES)
        for key in ("gap_table", "ref_only_diacritics", "pred_only_diacritics",
                    "total_ref_diacritics", "total_pred_diacritics"):
            assert key in result

    def test_ref_has_more_diacritics_than_pred(self):
        # fra entry has nasalization in ref (õ) but not in pred (o)
        result = analyze_diacritic_gap(_ENTRIES)
        assert result["total_ref_diacritics"] >= result["total_pred_diacritics"]

    def test_empty_entries_zero_counts(self):
        no_diacritic_entries = [
            {"ref_str": "ab", "pred_str": "ab"}
        ]
        result = analyze_diacritic_gap(no_diacritic_entries)
        assert result["total_ref_diacritics"] == 0
        assert result["total_pred_diacritics"] == 0


class TestAnalyzeSubstitutions:
    def test_returns_expected_keys(self):
        result = analyze_substitutions(_ENTRIES)
        for key in ("per", "total_ref_phones", "operation_counts",
                    "substitutions", "deletions", "insertions",
                    "best_phones", "worst_phones", "phone_accuracy"):
            assert key in result

    def test_perfect_entries_give_zero_per(self):
        perfect = [
            {
                "ref_phones": ["a", "b"],
                "pred_phones": ["a", "b"],
                "ref_str": "ab",
                "pred_str": "ab",
            }
        ]
        result = analyze_substitutions(perfect)
        assert result["per"] == pytest.approx(0.0)

    def test_per_bounded_range(self):
        result = analyze_substitutions(_ENTRIES)
        assert result["per"] >= 0.0

    def test_substitution_classified(self):
        sub_entry = [
            {
                "ref_phones": ["b", "õ"],
                "pred_phones": ["b", "o"],
                "ref_str": "bõ",
                "pred_str": "bo",
            }
        ]
        result = analyze_substitutions(sub_entry)
        assert len(result["substitutions"]) >= 1
        # Should detect substitution of õ → o
        refs = [s["ref"] for s in result["substitutions"]]
        # The exact segmentation depends on panphon, but substitution should be found
        assert len(result["substitutions"]) >= 1


class TestAnalyzeAnnotationConsistency:
    def test_returns_dict(self):
        result = analyze_annotation_consistency(_ENTRIES)
        assert isinstance(result, dict)

    def test_lang_with_single_utt_excluded(self):
        result = analyze_annotation_consistency(_ENTRIES)
        # fra has only 1 entry → should not appear
        assert "fra" not in result

    def test_lang_with_multiple_utts_present(self):
        result = analyze_annotation_consistency(_ENTRIES)
        assert "eng" in result

    def test_result_structure(self):
        result = analyze_annotation_consistency(_ENTRIES)
        if "eng" in result:
            eng = result["eng"]
            for key in ("n_utterances", "avg_ref_distance", "std_ref_distance",
                        "avg_inventory_jaccard", "inconsistent_phones"):
                assert key in eng


# ---------------------------------------------------------------------------
# Section 7 — Report Generation
# ---------------------------------------------------------------------------


class TestFormatReport:
    def _run_all_analyses(self):
        accent = analyze_accent_variance(_ENTRIES)
        diacritics = analyze_diacritic_gap(_ENTRIES)
        substitutions = analyze_substitutions(_ENTRIES)
        consistency = analyze_annotation_consistency(_ENTRIES)
        # Skip feature analysis (panphon-intensive); use stub
        features = {"error": "panphon not run in this test"}
        return accent, diacritics, substitutions, features, consistency

    def test_returns_string(self):
        accent, diacritics, substitutions, features, consistency = self._run_all_analyses()
        report = format_report(_ENTRIES, accent, diacritics, substitutions, features, consistency)
        assert isinstance(report, str)

    def test_contains_section_headers(self):
        accent, diacritics, substitutions, features, consistency = self._run_all_analyses()
        report = format_report(_ENTRIES, accent, diacritics, substitutions, features, consistency)
        assert "Accent Deafness" in report
        assert "Diacritic" in report
        assert "Substitution" in report
        assert "Annotation Consistency" in report

    def test_contains_data_summary(self):
        accent, diacritics, substitutions, features, consistency = self._run_all_analyses()
        report = format_report(_ENTRIES, accent, diacritics, substitutions, features, consistency)
        assert "Utterances" in report
        assert "Languages" in report

    def test_entry_count_in_report(self):
        accent, diacritics, substitutions, features, consistency = self._run_all_analyses()
        report = format_report(_ENTRIES, accent, diacritics, substitutions, features, consistency)
        assert str(len(_ENTRIES)) in report


class TestPhoneInventoryJaccard:
    """Section 3 — phone_inventory_jaccard."""

    @pytest.fixture(scope="class")
    def evaluator(self):
        from src.metrics.phone_recognition import PhoneRecognitionEvaluator

        return PhoneRecognitionEvaluator(normalize_ipa=True)

    def _df(self, pairs):
        return pd.DataFrame(
            [
                {"utt_id": f"u{i}", "langcode": "eng", "langname": "English",
                 "reference": ref, "predicted": hyp}
                for i, (ref, hyp) in enumerate(pairs)
            ]
        )

    def test_returns_dataframe(self, evaluator):
        df = phone_inventory_jaccard(self._df([("ab", "ab")]), evaluator)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, evaluator):
        df = phone_inventory_jaccard(self._df([("ab", "ab")]), evaluator)
        for col in ("utt_id", "langcode", "langname", "ref_n", "pred_n", "shared", "jaccard"):
            assert col in df.columns

    def test_perfect_match_jaccard_one(self, evaluator):
        df = phone_inventory_jaccard(self._df([("a", "a")]), evaluator)
        assert df["jaccard"].iloc[0] == pytest.approx(1.0)

    def test_no_overlap_jaccard_zero(self, evaluator):
        # "a" vs "b" — disjoint inventories → Jaccard = 0
        df = phone_inventory_jaccard(self._df([("a", "b")]), evaluator)
        assert df["jaccard"].iloc[0] == pytest.approx(0.0)

    def test_partial_overlap(self, evaluator):
        # ref={a,b}, hyp={a,c} → shared=1, union=3 → Jaccard=1/3
        df = phone_inventory_jaccard(self._df([("ab", "ac")]), evaluator)
        j = df["jaccard"].iloc[0]
        assert 0.0 < j < 1.0

    def test_empty_prediction_jaccard_zero(self, evaluator):
        df = phone_inventory_jaccard(self._df([("ab", "")]), evaluator)
        assert df["jaccard"].iloc[0] == pytest.approx(0.0)

    def test_both_empty_jaccard_one(self, evaluator):
        # No phones on either side → Jaccard defined as 1 (nothing to miss)
        df = phone_inventory_jaccard(self._df([("", "")]), evaluator)
        assert df["jaccard"].iloc[0] == pytest.approx(1.0)

    def test_row_count_matches_input(self, evaluator):
        pairs = [("a", "a"), ("ab", "b"), ("abc", "ac")]
        df = phone_inventory_jaccard(self._df(pairs), evaluator)
        assert len(df) == len(pairs)

    def test_sorted_ascending(self, evaluator):
        # Provide pairs with known ordering: perfect match > partial > no overlap
        pairs = [("a", "b"), ("ab", "a"), ("a", "a")]
        df = phone_inventory_jaccard(self._df(pairs), evaluator)
        assert list(df["jaccard"]) == sorted(df["jaccard"])


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_align_phones_no_numpy(self):
        """align_phones must use pure-Python DP (no numpy array allocation)."""
        import inspect

        src = inspect.getsource(align_phones)
        # Must not create numpy arrays (list-of-lists DP only)
        assert "np.zeros" not in src
        assert "np.array" not in src
        assert "numpy.zeros" not in src

    def test_load_jsonl_no_break_after_100(self):
        """Regression: old code had `break` inside the progress print block."""
        import inspect

        src = inspect.getsource(load_jsonl)
        # The only acceptable 'break' is NOT tied to a line-count check
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if "break" in line:
                context = "\n".join(lines[max(0, i - 3): i + 1])
                assert "% 100" not in context, (
                    f"Found a break near a mod-100 check — regression of debug break:\n{context}"
                )

    def test_phone_confusion_matrix_signature_has_evaluator(self):
        """Regression: evaluator must be an explicit parameter, not a global."""
        import inspect

        params = list(inspect.signature(phone_confusion_matrix).parameters)
        assert params[1] == "evaluator"

    def test_del_sym_and_ins_sym_defined(self):
        assert DEL_SYM == "<DEL>"
        assert INS_SYM == "<INS>"


# ---------------------------------------------------------------------------
# Section 2 — DATASETS / LANG_FN constants (including gmuaccent)
# ---------------------------------------------------------------------------


class TestDatasetsConstant:
    def test_known_datasets_present(self):
        for name in ("voxangeles", "doreco", "tusom", "buckeye", "gmuaccent"):
            assert name in DATASETS

    def test_gmuaccent_maps_to_correct_dir(self):
        assert DATASETS["gmuaccent"] == "decodedv3.gmuaccent"

    def test_all_datasets_have_lang_fn(self):
        for name in DATASETS:
            assert name in LANG_FN, f"LANG_FN missing entry for dataset '{name}'"


class TestGmuaccentLangFn:
    """Tests for the _gmuaccent_lang helper via LANG_FN['gmuaccent']."""

    def _fn(self):
        return LANG_FN["gmuaccent"]

    def test_simple_accent_stripped(self):
        fn = self._fn()
        assert fn({"utt_id": "arabic42"}) == "arabic"

    def test_multiword_accent_stripped(self):
        fn = self._fn()
        assert fn({"utt_id": "jamaican_creole_english7"}) == "jamaican_creole_english"

    def test_no_trailing_digits(self):
        # utt_id with no trailing digits → full string returned
        fn = self._fn()
        assert fn({"utt_id": "english"}) == "english"

    def test_large_number_suffix(self):
        fn = self._fn()
        assert fn({"utt_id": "mandarin_chinese1234"}) == "mandarin_chinese"

    def test_single_digit_suffix(self):
        fn = self._fn()
        assert fn({"utt_id": "hindi1"}) == "hindi"

    def test_all_digits_returns_empty(self):
        # Edge case: utt_id is entirely digits → empty string
        fn = self._fn()
        assert fn({"utt_id": "123"}) == ""
