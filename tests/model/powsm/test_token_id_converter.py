import pytest
import yaml

from src.model.powsm import token_id_converter as tic


def test_token_id_converter_converts_tokens_and_ids():
    converter = tic.TokenIDConverter(["<unk>", "<pad>", "HELLO"])

    assert converter.get_num_vocabulary_size() == 3
    assert converter.tokens2ids(["HELLO", "<pad>"]) == [2, 1]
    # Unknown tokens should map to unk id
    assert converter.tokens2ids(["UNKNOWN"]) == [0]

    assert converter.ids2tokens([2, 1, 0]) == ["HELLO", "<pad>", "<unk>"]


def test_token_id_converter_duplicate_tokens_raise():
    with pytest.raises(RuntimeError):
        tic.TokenIDConverter(["<unk>", "<unk>"])


def test_build_powsm_tokenizer_from_files(tmp_path):
    tokens_file = tmp_path / "tokens.txt"
    tokens_file.write_text("<unk>\nA\nB\n", encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump({"token_list": str(tokens_file)}), encoding="utf-8"
    )

    converter = tic.build_powsm_tokenizer_from_files(str(config_file))

    assert converter.token_list[:3] == ["<unk>", "A", "B"]
    assert converter.unk_symbol == "<unk>"


def test_build_powsm_tokenizer_invokes_snapshot(monkeypatch, tmp_path):
    tokens_file = tmp_path / "tokens.txt"
    tokens_file.write_text("<unk>\nfoo\n", encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump({"token_list": str(tokens_file)}), encoding="utf-8"
    )

    called = {}

    def fake_snapshot_download(*, repo_id, force_download, local_dir, local_dir_use_symlinks):
        called["params"] = {
            "repo_id": repo_id,
            "force_download": force_download,
            "local_dir": local_dir,
            "local_dir_use_symlinks": local_dir_use_symlinks,
        }

    monkeypatch.setattr(tic, "snapshot_download", fake_snapshot_download)

    converter = tic.build_powsm_tokenizer(
        work_dir=str(tmp_path), hf_repo="espnet/test", config_file=str(config_file)
    )

    assert called["params"] == {
        "repo_id": "espnet/test",
        "force_download": False,
        "local_dir": str(tmp_path),
        "local_dir_use_symlinks": False,
    }
    assert converter.token_list[1] == "foo"


def test_whitespace_token_preserved(tmp_path):
    """Space character as a token is preserved (bug M25 fix)."""
    tokens_file = tmp_path / "tokens.txt"
    # Line with a single space: first char kept, rest rstripped -> " "
    tokens_file.write_text("<unk>\n \nA\n", encoding="utf-8")

    converter = tic.TokenIDConverter(token_list=tokens_file)

    assert " " in converter.token_list
    assert converter.token2id[" "] == 1
    assert converter.get_num_vocabulary_size() == 3


def test_build_powsm_tokenizer_whitespace(tmp_path):
    """build_powsm_tokenizer_from_files preserves space token (M25)."""
    tokens_file = tmp_path / "tokens.txt"
    tokens_file.write_text("<unk>\n \nB\n", encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump({"token_list": str(tokens_file)}),
        encoding="utf-8",
    )

    converter = tic.build_powsm_tokenizer_from_files(str(config_file))

    assert " " in converter.token_list
    assert converter.token_list[1] == " "
    assert converter.get_num_vocabulary_size() == 3
