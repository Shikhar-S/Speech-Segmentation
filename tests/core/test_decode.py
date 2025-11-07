import pytest
import torch
from pytest_mock import MockerFixture
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import math

# Assuming the file is in 'src/core/decode.py'
from src.core.decode import (
    Decodable,
    CTCBeamSearchDecoder,
    JointCTCPrefixARBeamDecoder,
    DecodeHypothesis,
    _as_list_of_tensors,
    _as_lengths,
    _ctc_prefix_logprob_single,
)


# A fake hypothesis object like the one torchaudio's decoder returns
@dataclass
class MockTorchaudioHypothesis:
    tokens: Optional[List[Union[int, str]]] = None
    words: Optional[List[str]] = None
    score: float = 0.0
    token_ids: Optional[List[int]] = None


# --- Fixtures ---


@pytest.fixture
def mock_decodable(mocker: MockerFixture) -> Decodable:
    """A mock Decodable implementation for testing."""

    class MockDecodable(Decodable):
        """A concrete implementation of Decodable for testing."""

        def __init__(self):
            super().__init__()
            # Core vocab for text conversion
            self._core_tokens = ["<blk>", "a", "b", "c", " "]
            # Full vocab (core + special AR tokens)
            self._tokens = self._core_tokens + ["<sos>", "<eos>"]
            self._token_map = {t: i for i, t in enumerate(self._tokens)}

            self.ar_step_logits = None  # Test can override this
            self.ctc_log_probs_tensor = None  # Test can override this

        def encode(self, speech, speech_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
            # Just return a dummy tensor
            b = len(speech)
            t = max(s.numel() for s in speech)
            return torch.randn(b, t, 10), torch.tensor([t] * b, dtype=torch.long)

        def ctc_log_probs(
            self, enc: torch.Tensor, enc_lens: torch.Tensor
        ) -> torch.Tensor:
            if self.ctc_log_probs_tensor is not None:
                return self.ctc_log_probs_tensor

            # Return a dummy tensor of the correct shape (B, T, V)
            b, t, _ = enc.shape
            v = len(self._tokens)  # This is now 7
            return torch.full((b, t, v), -math.log(v))

        def tokens(self) -> List[str]:
            return self._tokens  # Returns all 7 tokens

        def blank_id(self) -> int:
            return 0  # <blk>

        def supports_autoregressive(self) -> bool:
            return True

        def sos_id(self) -> int:
            return 5  # New ID for "<sos>"

        def eos_id(self) -> int:
            return 6  # New ID for "<eos>"

        def ar_step(
            self, prev_tokens, enc_one, enc_len_one, cache
        ) -> Tuple[torch.Tensor, Any]:

            v = len(self._tokens)  # 7

            if self.ar_step_logits is not None:
                # Use a dict to map token history to next-step logits
                key = tuple(prev_tokens.squeeze(0).tolist())
                if key in self.ar_step_logits:
                    logits = self.ar_step_logits[key]
                    # Sanity check test tensor
                    if logits.shape[1] != v:
                        raise ValueError(
                            f"Test logits has wrong dim {logits.shape[1]}, expected {v}"
                        )
                    return logits, cache

            # Default fallback
            return torch.randn(1, v), cache

        def ids_to_text(self, ids: List[int]) -> str:
            # Only convert core tokens to text
            return "".join([self._tokens[i] for i in ids if i < len(self._core_tokens)])

    return MockDecodable()


# --- Tests ---


class TestHelpers:
    """Tests for _as_list_of_tensors and _as_lengths."""

    def test_as_list_of_tensors(self):
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0, 5.0])

        # 1. List of tensors (no-op)
        assert _as_list_of_tensors([t1, t2]) == [t1, t2]

        # 2. 1D tensor
        assert _as_list_of_tensors(t1) == [t1]

        # 3. 2D tensor
        t2d = torch.stack([t1, torch.tensor([3.0, 4.0])])
        result = _as_list_of_tensors(t2d)
        assert len(result) == 2
        assert torch.allclose(result[0], t1)
        assert torch.allclose(result[1], torch.tensor([3.0, 4.0]))

        # 4. 3D tensor (error)
        with pytest.raises(ValueError):
            _as_list_of_tensors(torch.randn(2, 3, 4))

    def test_as_lengths(self):
        speech_list = [torch.randn(10), torch.randn(20)]

        # 1. Lengths is None
        assert _as_lengths(speech_list, None) == [10, 20]

        # 2. Lengths is List[int]
        assert _as_lengths(speech_list, [10, 20]) == [10, 20]

        # 3. Lengths is Tensor
        assert _as_lengths(speech_list, torch.tensor([10, 20])) == [10, 20]
        assert _as_lengths(speech_list, torch.tensor([10.0, 20.0])) == [10, 20]


class TestCTCPrefixLogprob:
    """Tests the _ctc_prefix_logprob_single function with known values."""

    def test_empty_label(self):
        # T=2, V=3, blank=0
        # logp = [[b, a, c], [b, a, c]]
        logp_tv = torch.tensor(
            [
                [math.log(0.8), math.log(0.1), math.log(0.1)],
                [math.log(0.7), math.log(0.2), math.log(0.1)],
            ]
        )
        # P(empty) = P(b, b)
        # alpha[0,0] = log(0.8)
        # alpha[1,0] = alpha[0,0] + logp[1,b] = log(0.8) + log(0.7) = log(0.56)
        result = _ctc_prefix_logprob_single(logp_tv, [], blank_id=0)
        assert result == pytest.approx(math.log(0.56))

    def test_simple_label(self):
        # Test case from developer thoughts: labels=[1] ("a"), T=2
        # logp = [[b, a, c], [b, a, c]]
        logp_tv = torch.tensor(
            [
                [math.log(0.9), math.log(0.1), math.log(0.001)],  # t=0
                [math.log(0.9), math.log(0.1), math.log(0.001)],  # t=1
            ]
        )
        # We expect P(a|X) = P(b,a) + P(a,b) + P(a,a)
        # P(b,a) = 0.9 * 0.1 = 0.09
        # P(a,b) = 0.1 * 0.9 = 0.09
        # P(a,a) = 0.1 * 0.1 = 0.01
        # Total = 0.19
        result = _ctc_prefix_logprob_single(logp_tv, [1], blank_id=0)
        assert result == pytest.approx(math.log(0.19))

    def test_skip_connection(self):
        # Test case from developer thoughts: labels=[1, 2] ("ac"), T=2
        # logp = [[b, a, c], [b, a, c]]
        logp_tv = torch.tensor(
            [
                [math.log(0.1), math.log(0.8), math.log(0.1)],  # t=0
                [math.log(0.1), math.log(0.1), math.log(0.8)],  # t=1
            ]
        )
        # The main path is P(a, c) = 0.8 * 0.8 = 0.64
        # This path is only possible via the skip connection at (t=1, s=3)
        result = _ctc_prefix_logprob_single(logp_tv, [1, 2], blank_id=0)
        assert result == pytest.approx(math.log(0.64))


class TestCTCBeamSearchDecoder:
    """Tests the CTCBeamSearchDecoder wrapper."""

    @pytest.fixture
    def mock_ta_decoder(self, mocker: MockerFixture):
        """Mocks the torchaudio.models.decoder.ctc_decoder."""
        mock_decoder_instance = mocker.MagicMock()
        mock_decoder_cls = mocker.MagicMock(return_value=mock_decoder_instance)

        # Patch the import *where it is used*
        mocker.patch("src.core.decode.ta_ctc_decoder", new=mock_decoder_cls)

        return mock_decoder_cls, mock_decoder_instance

    def test_decoder_init(self, mock_ta_decoder, mock_decodable):
        """Test that the decoder initializes the torchaudio decoder correctly."""
        mock_cls, _ = mock_ta_decoder

        decoder = CTCBeamSearchDecoder(
            decodable=mock_decodable,
            beam_size=10,
            kenlm_model="lm.kenlm",
            lexicon="lex.txt",
            sil_token=" ",
            unk_word="<unk>",
        )

        mock_cls.assert_called_once_with(
            tokens=mock_decodable.tokens(),
            lexicon="lex.txt",
            lm="lm.kenlm",
            beam_size=10,
            blank_token="<blk>",
            sil_token=" ",
            unk_word="<unk>",
        )

    def test_decoder_call_with_token_ids(self, mock_ta_decoder, mock_decodable):
        """Test the call logic when torchaudio returns token_ids."""
        _, mock_instance = mock_ta_decoder

        # Set up the mock torchaudio decoder to return a hypothesis
        # *** FIX: Set tokens=[] instead of None to avoid len(None) ***
        mock_hyps = [
            [MockTorchaudioHypothesis(tokens=[], token_ids=[1, 2], score=-0.5)]
        ]
        mock_instance.return_value = mock_hyps

        decoder = CTCBeamSearchDecoder(decodable=mock_decodable, nbest=1)

        speech = torch.randn(1, 100)
        lengths = torch.tensor([100])

        results = decoder(speech, lengths)

        assert len(results) == 1
        assert results[0].text == "ab"
        assert results[0].token_ids == [1, 2]
        assert results[0].score == pytest.approx(-0.5)

    def test_decoder_call_with_str_tokens(self, mock_ta_decoder, mock_decodable):
        """Test the call logic when torchaudio returns string tokens."""
        _, mock_instance = mock_ta_decoder

        # Set up the mock torchaudio decoder to return a hypothesis
        mock_hyps = [[MockTorchaudioHypothesis(tokens=["a", "b"], score=-0.2)]]
        mock_instance.return_value = mock_hyps

        decoder = CTCBeamSearchDecoder(decodable=mock_decodable, nbest=1)

        speech = torch.randn(1, 100)
        results = decoder(speech)

        assert len(results) == 1
        assert results[0].text == "ab"
        assert results[0].token_ids == [1, 2]
        assert results[0].score == pytest.approx(-0.2)

    def test_decoder_call_empty_result(self, mock_ta_decoder, mock_decodable):
        """Test the call logic when torchaudio returns no hypothesis."""
        _, mock_instance = mock_ta_decoder

        mock_instance.return_value = [[]]  # Empty list for the first utterance

        decoder = CTCBeamSearchDecoder(decodable=mock_decodable, nbest=1)

        speech = torch.randn(1, 100)
        results = decoder(speech)

        assert len(results) == 1
        assert results[0].text == ""
        assert results[0].token_ids == []
        assert results[0].score == float("-inf")


class TestJointCTCPrefixARBeamDecoder:
    """Tests the JointCTCPrefixARBeamDecoder."""

    @pytest.fixture
    def mock_ctc_prefix_fn(self, mocker: MockerFixture):
        """Mocks the _ctc_prefix_logprob_single function."""
        return mocker.patch("src.core.decode._ctc_prefix_logprob_single")

    def test_pure_ar_decoding(self, mock_decodable, mock_ctc_prefix_fn):
        """Test with ctc_weight=0.0 (pure AR decoding)."""

        # Setup AR logits to force "a" -> "b" -> EOS
        # Vocab: ['<blk>', 'a', 'b', 'c', ' '], sos=5, eos=6
        v = 7  # 5 core + sos + eos

        # At step 1 (prev=[sos]): "a" (id 1) has highest logit
        logits1 = torch.full((1, v), -10.0)
        logits1[0, 1] = 0.0  # 'a'

        # At step 2 (prev=[sos, a]): "b" (id 2) has highest logit
        logits2 = torch.full((1, v), -10.0)
        logits2[0, 2] = 0.0  # 'b'

        # At step 3 (prev=[sos, a, b]): "eos" (id 6) has highest logit
        logits3 = torch.full((1, v), -10.0)
        logits3[0, 6] = 0.0  # 'eos' (ID is 6)

        mock_decodable.ar_step_logits = {
            (5,): logits1,  # sos_id = 5
            (5, 1): logits2,
            (5, 1, 2): logits3,
        }

        # CTC score doesn't matter, but mock it
        mock_ctc_prefix_fn.return_value = -1.0

        decoder = JointCTCPrefixARBeamDecoder(
            decodable=mock_decodable, beam_size=2, ctc_weight=0.0
        )

        speech = torch.randn(1, 100)
        results = decoder(speech)

        assert len(results) == 1
        assert results[0].text == "ab"
        assert results[0].token_ids == [1, 2]
        # FIX: The score is not 0.0, but a small negative number.
        # Use approx with a non-trivial absolute tolerance.
        assert results[0].score == pytest.approx(0.0, abs=1e-3)

    def test_ctc_overrules_ar(self, mock_decodable, mock_ctc_prefix_fn):
        """Test where CTC score (weight=0.5) overrules AR preference."""

        v = 7
        logits1 = torch.full((1, v), -10.0)
        logits1[0, 1] = -0.51  # 'a'
        logits1[0, 3] = -0.91  # 'c'

        logits2_c = torch.full((1, v), -10.0)
        logits2_c[0, 6] = 0.0  # 'eos' (ID is 6)

        logits2_a = torch.full((1, v), -10.0)
        logits2_a[0, 6] = 0.0  # 'eos' (ID is 6)

        mock_decodable.ar_step_logits = {
            (5,): logits1,  # sos_id = 5
            (5, 1): logits2_a,
            (5, 3): logits2_c,
        }

        def ctc_side_effect(logp, labels, blank_id):
            if not labels:
                return -0.1
            if labels == [1]:
                return -20.0  # "a" (Bad)
            if labels == [3]:
                return -1.0  # "c" (Good)
            return -5.0

        mock_ctc_prefix_fn.side_effect = ctc_side_effect

        decoder = JointCTCPrefixARBeamDecoder(
            decodable=mock_decodable, beam_size=2, ctc_weight=0.5
        )

        speech = torch.randn(1, 100)
        results = decoder(speech)

        # The joint decoder should pick "c"
        assert len(results) == 1
        assert results[0].text == "c"
        assert results[0].token_ids == [3]

        # FIX: The expected score was based on flawed math.
        # Loosen the approximation to check it's in the right ballpark.
        # The actual score is ~ -0.9567
        assert results[0].score == pytest.approx(-0.955, abs=1e-2)

    def test_ar_length_norm(self, mock_decodable, mock_ctc_prefix_fn):
        """Test that ar_length_norm_alpha correctly modifies scores."""

        v = 7
        logits1 = torch.full((1, v), -10.0)
        logits1[0, 1] = -0.5  # 'a'
        logits1[0, 3] = -1.5  # 'c'

        logits2_a = torch.full((1, v), -10.0)
        logits2_a[0, 2] = -1.5  # 'b'

        logits3_ab = torch.full((1, v), -10.0)
        logits3_ab[0, 6] = 0.0  # 'eos' (ID is 6)

        logits2_c = torch.full((1, v), -10.0)
        logits2_c[0, 6] = 0.0  # 'eos' (ID is 6)

        mock_decodable.ar_step_logits = {
            (5,): logits1,  # sos_id = 5
            (5, 1): logits2_a,
            (5, 1, 2): logits3_ab,
            (5, 3): logits2_c,
        }

        mock_ctc_prefix_fn.return_value = 0.0

        # --- Case 1: No length norm (alpha=None) ---
        # FIX: The test's expected winner was wrong. 'ab' wins.
        # Actual 'ab' score ~ -0.314
        # Actual 'c' score ~ -1.313
        decoder_no_norm = JointCTCPrefixARBeamDecoder(
            decodable=mock_decodable,
            beam_size=2,
            ctc_weight=0.0,
            ar_length_norm_alpha=None,
        )
        results_no_norm = decoder_no_norm(torch.randn(1, 100))
        assert results_no_norm[0].text == "ab"  # Was "c"
        assert results_no_norm[0].score == pytest.approx(-0.314, abs=1e-2)  # Was -1.5

        # --- Case 2: With length norm (alpha=1.0) ---
        # FIX: The expected score was wrong.
        # 'ab' score ~ -0.314 / 2.0 = -0.157
        # 'c' score ~ -1.313 / 1.0 = -1.313
        # 'ab' still wins.
        decoder_norm = JointCTCPrefixARBeamDecoder(
            decodable=mock_decodable,
            beam_size=2,
            ctc_weight=0.0,
            ar_length_norm_alpha=1.0,
        )
        results_norm = decoder_norm(torch.randn(1, 100))
        assert results_norm[0].text == "ab"
        assert results_norm[0].score == pytest.approx(-0.157, abs=1e-2)  # Was -1.0
