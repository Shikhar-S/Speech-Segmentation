"""Interfaces and decoders for end-to-end speech recognition models.

This module defines the `Decodable` abstract base class, which specifies
the interface required for models to be used with CTC and autoregressive
decoders.

It provides two decoder implementations:
  - `CTCBeamSearchDecoder`: A wrapper for the torchaudio (Flashlight)
    beam search decoder.
  - `JointCTCPrefixARBeamDecoder`: A single-pass joint CTC/AR beam search
    decoder that uses CTC prefixes to constrain the AR search.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

_HAS_TORCHAUDIO_DECODER = None
try:
    from torchaudio.models.decoder import ctc_decoder as ta_ctc_decoder
except Exception as e:
    _HAS_TORCHAUDIO_DECODER = e


class Decodable(ABC, torch.nn.Module):
    """Abstract interface for decodable models.

    Models implementing this interface can be used with the provided decoders.

    BATCHING:
      - speech: List[1D Tensor] or a 2D Tensor (B, T) of mono waveforms.
      - speech_lengths: List[int] or 1D Tensor of original sample lengths.

    Required for CTC decoding:
      - encode(speech, speech_lengths) -> (enc, enc_lens)
      - ctc_log_probs(enc, enc_lens) -> (B, T, V)
      - tokens() -> List[str]
      - blank_id() -> int

    Required for AR decoding (if supports_autoregressive() == True):
      - sos_id(), eos_id() -> int
      - ar_step(prev_tokens, enc_one, enc_len_one, cache) -> (logits, cache)
            logits: (1, V) tensor of *next token* logits.
      - supports_autoregressive() -> bool

    Text conversion (used for final output):
      - ids_to_text(ids: List[int]) -> str
    """

    # ---- CTC hooks ----
    @abstractmethod
    def encode(
        self,
        speech: Union[List[torch.Tensor], torch.Tensor],
        speech_lengths: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes speech inputs.

        Args:
            speech: A 2D (B, T) tensor or list of 1D tensors.
            speech_lengths: A 1D tensor or list of ints.

        Returns:
            A tuple of (enc, enc_lens), where `enc` is the encoded
            feature tensor (e.g., (B, T', D)) and `enc_lens` is a
            1D tensor of encoded feature lengths.
        """
        ...

    @abstractmethod
    def ctc_log_probs(self, enc: torch.Tensor, enc_lens: torch.Tensor) -> torch.Tensor:
        """Computes CTC log-probabilities from encoded features.

        Args:
            enc: Encoded feature tensor (B, T, D).
            enc_lens: Lengths of encoded features (B,).

        Returns:
            A (B, T, V) tensor of framewise log-probabilities over the
            vocabulary.
        """
        ...

    @abstractmethod
    def tokens(self) -> List[str]:
        """Returns the vocabulary token list."""
        ...

    @abstractmethod
    def blank_id(self) -> int:
        """Returns the ID of the CTC blank token."""
        ...

    # ---- AR hooks ----
    @abstractmethod
    def supports_autoregressive(self) -> bool:
        """Returns True if the model supports autoregressive decoding."""
        ...

    @abstractmethod
    def sos_id(self) -> int:
        """Returns the ID of the Start-of-Sequence (SOS) token."""
        ...

    @abstractmethod
    def eos_id(self) -> int:
        """Returns the ID of the End-of-Sequence (EOS) token."""
        ...

    @abstractmethod
    def ar_step(
        self,
        prev_tokens: torch.Tensor,  # (1, L)
        enc_one: torch.Tensor,  # (1, T, D) or (T, D)
        enc_len_one: torch.Tensor,  # (1,) or scalar tensor
        cache: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """Performs a single autoregressive decoding step.

        Args:
            prev_tokens: (1, L) tensor of token ids (including SOS).
            enc_one: Encoded features for a single utterance.
            enc_len_one: Length of the encoded features.
            cache: Any model-specific state from the *previous* step.

        Returns:
            A tuple of (logits, new_cache):
                logits (torch.Tensor): (1, V) tensor of next-token logits.
                new_cache (Any): Updated state for the *next* step.
        """
        ...

    # ---- Text conversion ----
    @abstractmethod
    def ids_to_text(self, ids: List[int]) -> str:
        """Converts a list of token IDs to a string."""
        ...


def _as_list_of_tensors(
    speech: Union[List[torch.Tensor], torch.Tensor],
) -> List[torch.Tensor]:
    """Normalizes speech input to a list of 1D tensors."""
    if isinstance(speech, torch.Tensor):
        if speech.ndim == 1:
            return [speech]
        elif speech.ndim == 2:
            return [speech[i] for i in range(speech.size(0))]
        else:
            raise ValueError("speech tensor must be 1D or 2D")
    return speech


def _as_lengths(
    speech: List[torch.Tensor],
    speech_lengths: Optional[Union[List[int], torch.Tensor]],
) -> List[int]:
    """Normalizes speech lengths to a list of ints."""
    if speech_lengths is None:
        return [int(x.numel()) for x in speech]
    if isinstance(speech_lengths, torch.Tensor):
        return [int(x) for x in speech_lengths.tolist()]
    return [int(x) for x in speech_lengths]


@dataclass
class DecodeHypothesis:
    """A single decoding hypothesis."""

    text: str
    token_ids: List[int]
    score: float  # total log score


class CTCBeamSearchDecoder:
    """
    Torchaudio CTC beam-search (Flashlight-based). No LM by default.
    """

    def __init__(
        self,
        decodable: Decodable,
        beam_size: int = 5,
        nbest: int = 1,
        lexicon: Optional[str] = None,
        kenlm_model: Optional[str] = None,
        blank_token_override: Optional[str] = None,
        sil_token: Optional[str] = None,
        unk_word: Optional[str] = None,
    ):
        """
        Args:
            decodable: An instance of a decodable model.
            beam_size: Beam size for CTC beam search.
            nbest: Number of best hypotheses to return per utterance.
            lexicon: Optional path to a lexicon file.
            kenlm_model: Optional path to a KenLM model file.
            blank_token_override: Optional token to use as the blank token.
            sil_token: Optional token to use as the silence token.
            unk_word: Optional token to use as the unknown word.
        """
        if _HAS_TORCHAUDIO_DECODER is not None:
            raise RuntimeError(
                f"Error: {_HAS_TORCHAUDIO_DECODER}"
                f"torchaudio CTC decoder can not be imported. "
                f"Make sure you have flashlight-text and torchaudio installed. "
            )

        self.decodable = decodable
        self.beam_size = beam_size
        self.nbest = nbest

        toks = decodable.tokens()
        if not isinstance(toks, list) or not all(isinstance(t, str) for t in toks):
            raise ValueError("Decodable.tokens() must return List[str] ordered by ID")

        blank_tok = blank_token_override or toks[decodable.blank_id()]
        self._token_to_id = {t: i for i, t in enumerate(toks)}

        # Build TA decoder
        self._ta_decoder = ta_ctc_decoder(
            tokens=toks,
            lexicon=lexicon,
            lm=kenlm_model,
            beam_size=beam_size,
            blank_token=blank_tok,
            sil_token=sil_token,
            unk_word=unk_word,
        )

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[List[torch.Tensor], torch.Tensor],
        speech_lengths: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> List[DecodeHypothesis]:
        """Decodes the input speech.

        Args:
            speech: A 2D (B, T) tensor or list of 1D tensors.
            speech_lengths: A 1D tensor or list of ints.

        Returns:
            A list of `DecodeHypothesis` objects, one for each utterance
            in the batch (representing the 1-best hypothesis).
        """
        speech_list = _as_list_of_tensors(speech)
        lengths = _as_lengths(speech_list, speech_lengths)

        # 1) Encode (model decides what "encode" means)
        enc, enc_lens = self.decodable.encode(speech_list, lengths)

        # 2) Get CTC log-probs
        logp = self.decodable.ctc_log_probs(enc, enc_lens).cpu()  # (B, T, V)
        T_mask = enc_lens.cpu().to(dtype=torch.long)

        # 3) Decode with torchaudio
        nbest_results = self._ta_decoder(logp, T_mask)  # List[List[Hypothesis]]

        hyps: List[DecodeHypothesis] = []
        for one in nbest_results:
            # take top-nbest per utterance
            best = one[: self.nbest]

            # We return 1-best by default.
            if not best:
                hyps.append(
                    DecodeHypothesis(text="", token_ids=[], score=float("-inf"))
                )
                continue

            hyp = best[0]

            # Tokens may be strings or ints depending on decoder path.
            # Normalize to IDs.
            raw_tokens = getattr(hyp, "tokens", getattr(hyp, "words", []))
            if len(raw_tokens) == 0 and hasattr(hyp, "token_ids"):
                # some builds expose token_ids directly
                token_ids = list(getattr(hyp, "token_ids"))
            else:
                token_ids = []
                for t in raw_tokens:
                    if isinstance(t, int):
                        token_ids.append(t)
                    elif isinstance(t, str):
                        token_ids.append(self._token_to_id.get(t, -1))
                    else:
                        raise TypeError(
                            "Unexpected token type from torchaudio decoder."
                        )
                # drop unknowns that mapped to -1
                token_ids = [i for i in token_ids if i >= 0]

            text = self.decodable.ids_to_text(token_ids)
            score = float(getattr(hyp, "score", 0.0))
            hyps.append(DecodeHypothesis(text=text, token_ids=token_ids, score=score))

        return hyps


def _ctc_prefix_logprob_single(
    logp_TV: torch.Tensor,
    labels: List[int],
    blank_id: int,
) -> float:
    """Exact CTC forward DP for a *fixed* label sequence.

    Calculates log P_ctc(labels | X) where `labels` are the
    *non-blank* target symbols. Uses blank interleaving (size S = 2*L+1).

    Args:
        logp_TV: (T, V) log-softmax over vocabulary for each frame.
        labels: List[int] target symbols (no blanks inside).
        blank_id: The ID of the blank token.

    Returns:
        The CTC log-probability as a float.
    """
    T, V = logp_TV.shape
    L = len(labels)
    S = 2 * L + 1

    # Build extended target sequence with blanks: e.g., [b, l1, b, l2, ..., b]
    ext = [blank_id] * S
    for i, c in enumerate(labels):
        ext[2 * i + 1] = c

    # DP table alpha[t, s] in log domain
    alpha = logp_TV.new_full((T, S), float("-inf"))

    # init (t=0)
    alpha[0, 0] = logp_TV[0, blank_id]
    if S > 1:
        alpha[0, 1] = logp_TV[0, ext[1]]

    # forward
    for t in range(1, T):
        # s=0 (must come from s=0)
        alpha[t, 0] = alpha[t - 1, 0] + logp_TV[t, ext[0]]
        for s in range(1, S):
            # from same state (stay) or from previous state (advance)
            stay = alpha[t - 1, s]
            prev = alpha[t - 1, s - 1]
            acc = torch.logaddexp(stay, prev)

            # If at a non-blank symbol state (s % 2 == 1) and
            # symbol differs from two steps back, we can skip over blank.
            if s % 2 == 1 and s > 1 and ext[s] != ext[s - 2]:
                acc = torch.logaddexp(acc, alpha[t - 1, s - 2])

            alpha[t, s] = acc + logp_TV[t, ext[s]]

    # termination: sum of the last two states (blank or last symbol)
    if S == 1:
        return float(alpha[T - 1, 0].item())
    return float(torch.logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2]).item())


@dataclass
class _Beam:
    """Internal state for JointCTCPrefixARBeamDecoder."""

    tokens: List[int]  # Full sequence incl. SOS, potentially EOS
    ar_score: float  # Accumulated raw (un-normalized) AR log-prob
    cache: Any  # AR decoder state
    ctc_score: float  # Absolute CTC log-prob of core tokens (no SOS/EOS)


class JointCTCPrefixARBeamDecoder:
    """
    Single-pass joint decoder with CTC prefix-constrained beam search.

    At each step:
      - Expand each beam with AR top-K candidates (k = beam_size).
      - For each candidate 'c', compute exact CTC prefix score
        log P_ctc(prefix ⊕ c | X) via a trellis DP.
      - Combine:
            joint = (1 - ctc_weight) * AR_logprob_accum
                  + (      ctc_weight) * CTC_logprob(prefix_plus_c)
      - Prune to 'beam_size'.

    Notes:
      * Batch size = 1 for simplicity.
    """

    def __init__(
        self,
        decodable: Decodable,
        beam_size: int = 8,
        max_len: int = 256,
        ctc_weight: float = 0.5,
        ar_length_norm_alpha: Optional[float] = None,
    ):
        """
        Args:
            decodable: An instance of a decodable model.
            beam_size: Beam size for joint search.
            max_len: Maximum length of generated hypothesis.
            ctc_weight: Weight for CTC score (0.0 to 1.0).
                0.0 = Pure AR decoding.
                1.0 = Pure CTC-constrained decoding (AR drives candidates).
            ar_length_norm_alpha: Optional alpha for AR length normalization
                (score / len**alpha).
        """
        if not decodable.supports_autoregressive():
            raise ValueError("Model must support AR decoding.")
        if not (0.0 <= ctc_weight <= 1.0):
            raise ValueError("ctc_weight must be in [0, 1].")

        self.decodable = decodable
        self.beam_size = int(beam_size)
        self.max_len = int(max_len)
        self.ctc_weight = float(ctc_weight)
        self.ar_length_norm_alpha = ar_length_norm_alpha

    def _get_joint_score(self, beam: _Beam) -> float:
        """Computes the final joint score for a beam."""
        core = [
            t
            for t in beam.tokens
            if t not in (self.decodable.sos_id(), self.decodable.eos_id())
        ]
        if self.ar_length_norm_alpha is not None:
            L = max(1, len(core))
            ar_norm = beam.ar_score / (L ** max(1e-6, self.ar_length_norm_alpha))
        else:
            ar_norm = beam.ar_score

        return (1.0 - self.ctc_weight) * ar_norm + (self.ctc_weight * beam.ctc_score)

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[List[torch.Tensor], torch.Tensor],
        speech_lengths: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> List[DecodeHypothesis]:
        """Decodes the input speech (batch size 1 only).

        Args:
            speech: A 2D (1, T) tensor or list of one 1D tensor.
            speech_lengths: A 1D tensor or list of one int.

        Returns:
            A list containing the single best `DecodeHypothesis`.
        """
        speech_list = _as_list_of_tensors(speech)
        lengths = _as_lengths(speech_list, speech_lengths)
        if len(speech_list) != 1:
            raise ValueError("This implementation currently supports batch=1.")

        # Encode once
        enc, enc_lens = self.decodable.encode(speech_list, lengths)
        enc_one = enc[:1]
        len_one = enc_lens[:1]
        device = enc_one.device

        # Precompute framewise CTC log-probs once: (1, T, V) -> (T, V)
        logp_TV = self.decodable.ctc_log_probs(enc_one, len_one)[0]
        V = logp_TV.size(1)

        sos = self.decodable.sos_id()
        eos = self.decodable.eos_id()
        blank = self.decodable.blank_id()

        # Beam stores state:
        beams: List[_Beam] = [
            _Beam(
                tokens=[sos],
                ar_score=0.0,
                cache=None,
                ctc_score=_ctc_prefix_logprob_single(logp_TV, [], blank),
            )
        ]
        finished: List[_Beam] = []

        for _ in range(self.max_len):
            new_beams: List[_Beam] = []
            all_ended = True

            for beam in beams:
                if beam.tokens and beam.tokens[-1] == eos:
                    finished.append(beam)
                    continue

                all_ended = False

                # AR step # (1, L)
                prev = torch.tensor(
                    beam.tokens, dtype=torch.long, device=device
                ).unsqueeze(0)

                logits, new_cache = self.decodable.ar_step(
                    prev, enc_one, len_one, beam.cache
                )
                ar_logp_vec = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)

                # Candidate pruning by AR top-K
                k = min(self.beam_size, V)
                topk = torch.topk(ar_logp_vec, k=k)
                cand_ids = topk.indices.tolist()
                cand_ar_scores = topk.values.tolist()

                # For each candidate, compute new CTC prefix score exactly
                for idx, ar_step_lp in zip(cand_ids, cand_ar_scores):
                    if idx == sos:  # avoid re-emitting SOS
                        continue

                    new_raw_ar_score = beam.ar_score + float(ar_step_lp)

                    if idx == eos:
                        # For EOS, CTC score remains same as prefix before EOS.
                        new_beams.append(
                            _Beam(
                                tokens=beam.tokens + [eos],
                                ar_score=new_raw_ar_score,
                                cache=new_cache,
                                ctc_score=beam.ctc_score,
                            )
                        )
                        continue

                    # Extend prefix for CTC with the symbol itself
                    core = [t for t in beam.tokens if t not in (sos, eos)]
                    new_core = core + [idx]
                    ctc_lp_new = _ctc_prefix_logprob_single(logp_TV, new_core, blank)

                    new_beams.append(
                        _Beam(
                            tokens=beam.tokens + [idx],
                            ar_score=new_raw_ar_score,
                            cache=new_cache,
                            ctc_score=ctc_lp_new,
                        )
                    )

            if all_ended:
                break

            # Prune to top-K by JOINT score
            new_beams.sort(key=self._get_joint_score, reverse=True)
            beams = new_beams[: self.beam_size]

            # Early stop if all active beams ended with EOS
            if beams and all(b.tokens[-1] == eos for b in beams):
                finished.extend(beams)
                break

        cands = finished if finished else beams

        # Select best by the same joint criterion
        if not cands:
            return [DecodeHypothesis(text="", token_ids=[], score=float("-inf"))]

        best_beam = max(cands, key=self._get_joint_score)

        core = [t for t in best_beam.tokens if t not in (sos, eos)]
        text = self.decodable.ids_to_text(core)
        return [
            DecodeHypothesis(
                text=text,
                token_ids=core,
                score=float(self._get_joint_score(best_beam)),
            )
        ]
