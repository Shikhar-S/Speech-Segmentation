"""Tests for src.data.kaldi_dataset, focused on _extract_language."""

from src.data.kaldi_dataset import KaldiDataset


def test_extract_language_blank_lines(tmp_path):
    """Blank lines interspersed with valid entries must not cause errors.

    Regression test for bug M9: _extract_language used to crash on blank
    lines because it called ``line.split()[:2]`` without guarding.
    """
    lang_file = tmp_path / "lang.txt"
    lang_file.write_text(
        "\n"
        "utt001 <eng><pr>\n"
        "\n"
        "\n"
        "utt002 <cmn><pr>\n"
        "\n"
    )

    result = KaldiDataset._extract_language(None, str(lang_file))

    assert result == {"utt001": "eng", "utt002": "cmn"}


def test_extract_language_valid(tmp_path):
    """Standard lang file with no blank lines parses correctly."""
    lang_file = tmp_path / "lang.txt"
    lang_file.write_text(
        "utt_a <fra><pr>\n"
        "utt_b <deu><pr>\n"
        "utt_c <jpn><pr>\n"
    )

    result = KaldiDataset._extract_language(None, str(lang_file))

    assert result == {"utt_a": "fra", "utt_b": "deu", "utt_c": "jpn"}


def test_extract_language_pr_suffix_stripped(tmp_path):
    """Keys ending with ``_pr`` should have the suffix removed."""
    lang_file = tmp_path / "lang.txt"
    lang_file.write_text("utt001_pr <eng><pr>\n")

    result = KaldiDataset._extract_language(None, str(lang_file))

    assert "utt001" in result
    assert "utt001_pr" not in result
    assert result["utt001"] == "eng"
