from src.model.powsm.builders_common import load_token_list


def test_load_token_list_preserves_whitespace(tmp_path):
    """Space character as a token is preserved by load_token_list (M25)."""
    tokens_file = tmp_path / "tokens.txt"
    # Line " \n": first char kept, rest rstripped -> " "
    tokens_file.write_text("<unk>\n \nA\n", encoding="utf-8")

    token_list = load_token_list(str(tokens_file))

    assert token_list == ["<unk>", " ", "A"]
    assert token_list[1] == " "


def test_load_token_list_from_list():
    """Passing a list returns a copy."""
    src = ["<unk>", "X", "Y"]
    result = load_token_list(src)

    assert result == src
    assert result is not src


def test_load_token_list_from_tuple():
    """Passing a tuple returns a list."""
    src = ("<unk>", "X")
    result = load_token_list(src)

    assert result == ["<unk>", "X"]
    assert isinstance(result, list)
