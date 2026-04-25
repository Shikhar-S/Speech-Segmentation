import warnings
import joblib
import numpy as np
import pandas as pd
import panphon
import torch
import torchaudio.compliance.kaldi as kaldi
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine as cos_dist

warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")

SR = 16000
FRAME_SHIFT = 320  # WavLM hop in samples
FRAME_SHIFT_SEC = FRAME_SHIFT / SR
MEL_FRAME_SHIFT_MS = 10
assert FRAME_SHIFT_SEC == 0.02, "Frame shift should be 20ms for WavLM features."


class SilenceHandler:
    def __init__(self, detector_path="logistic_regression_silence_detector.joblib"):
        self.model = joblib.load(detector_path)

    def _fill_gaps(self, raw_mask):
        neighbors_silent = np.logical_and(
            np.concatenate(([False], raw_mask[:-1])), np.concatenate((raw_mask[1:], [False]))
        )
        return np.logical_or(raw_mask, neighbors_silent)

    def predict_silence_mask(self, feats, *, frame_shift=FRAME_SHIFT):
        raw_silence_mask = self.model.predict(feats)
        filled_silence_mask = self._fill_gaps(raw_silence_mask)
        if frame_shift != FRAME_SHIFT:
            assert FRAME_SHIFT % frame_shift == 0, "FRAME_SHIFT must be a multiple of frame_shift"
            factor = FRAME_SHIFT // frame_shift
            filled_silence_mask = np.repeat(filled_silence_mask, factor)
        return filled_silence_mask

    def handle_silence(self, preds, silence_mask, snap_tolerance=1):
        """Suppress peaks inside silence, snap nearby peaks to silence edges, and inject silence-boundary frames."""
        n_frames = len(silence_mask)

        # Find contiguous silence spans [start, end)
        spans = []
        start = None
        for i, v in enumerate(silence_mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                spans.append((start, i))
                start = None
        if start is not None:
            spans.append((start, len(silence_mask)))

        snapped = set()
        silence_boundaries = []

        for s, e in spans:
            # Left edge (silence onset)
            if s > 0:
                nearby = preds[(preds >= s - snap_tolerance) & (preds <= s + snap_tolerance)]
                if len(nearby) > 0:
                    outside = nearby[nearby <= s]
                    silence_boundaries.append(outside.min() if len(outside) > 0 else nearby.min())
                    snapped.update(nearby.tolist())
                else:
                    silence_boundaries.append(s)

            # Right edge (silence offset)
            if e < n_frames:
                nearby = preds[(preds >= e - snap_tolerance) & (preds <= e + snap_tolerance)]
                if len(nearby) > 0:
                    outside = nearby[nearby >= e]
                    silence_boundaries.append(outside.max() if len(outside) > 0 else nearby.max())
                    snapped.update(nearby.tolist())
                else:
                    silence_boundaries.append(e)

        # Suppress peaks inside silence or already snapped
        keep = ~silence_mask[preds]
        for i, p in enumerate(preds):
            if p in snapped:
                keep[i] = False

        out = np.unique(np.concatenate([preds[keep], np.array(silence_boundaries, dtype=int)]))
        return out


class PhonologicalVectors:
    @staticmethod
    def prep_featmap(vocab, ft):
        names = (
            ["speech+"]
            + [f"{n}+" for n in ft.fts("a").names]
            + [f"{n}-" for n in ft.fts("a").names]
        )
        featmap = {}
        for v in vocab:
            if v == "_":
                featmap[v] = [1] + ([0] * (len(names) - 1))
            elif ft.seg_known(v):
                feats = ft.fts(v).numeric()
                featmap[v] = (
                    [0]
                    + [1 if n == 1 else 0 for n in feats]
                    + [1 if n == -1 else 0 for n in feats]
                )
        return names, featmap

    def split_phns(self, featname):
        index = self.featnames.index(featname)
        pos_phns = {p for p, v in self.featmap.items() if v[index] == 1}
        zero_phns = {p for p, v in self.featmap.items() if v[index] == 0}
        return pos_phns, zero_phns

    def calc_phnvectors(self, df, group_col):
        pos_vecs, zero_vecs, scales, biases = [], [], [], []
        self.in_dim = len(df[~df.feat.isna()].iloc[0].feat)

        for featname in self.featnames:
            pos_phns, zero_phns = self.split_phns(featname)
            if len(pos_phns) > 0 and len(zero_phns) > 0:
                pos_samples = np.stack(df[df[group_col].isin(pos_phns)].feat.tolist())
                zero_samples = np.stack(df[df[group_col].isin(zero_phns)].feat.tolist())
                pos_vec = pos_samples.mean(0)
                zero_vec = zero_samples.mean(0)
                w = pos_vec - zero_vec
                pos_center = (pos_samples @ w.T).mean(0)
                zero_center = (zero_samples @ w.T).mean(0)
                bias = -(zero_center + pos_center) / 2.0
                scale = 4.0 / (pos_center - zero_center)
            else:
                pos_vec = np.zeros(self.in_dim)
                zero_vec = np.zeros(self.in_dim)
                bias = 0.0
                scale = 1.0

            pos_vecs.append(pos_vec)
            zero_vecs.append(zero_vec)
            scales.append(scale)
            biases.append(bias)

        self.pos_vecs = np.stack(pos_vecs)
        self.zero_vecs = np.stack(zero_vecs)
        self.scales = np.stack(scales)
        self.biases = np.stack(biases)

    def _filter_features(self):
        """Remove dead and degenerate features.

        Dead: no phones in the + or 0 class.
        Degenerate: +/0 partition identical to speech+ (just a silence detector).
        """
        speech_pos, speech_zero = self.split_phns("speech+")
        keep = []
        for i, name in enumerate(self.featnames):
            pos_phns, zero_phns = self.split_phns(name)
            if len(pos_phns) == 0 or len(zero_phns) == 0:
                continue
            if name != "speech+" and pos_phns == speech_pos and zero_phns == speech_zero:
                continue
            keep.append(i)

        keep = np.array(keep)
        self.featnames = [self.featnames[i] for i in keep]
        self.pos_vecs = self.pos_vecs[keep]
        self.zero_vecs = self.zero_vecs[keep]
        self.scales = self.scales[keep]
        self.biases = self.biases[keep]

        # Update featmap to match reduced indexing
        self.featmap = {phone: [vals[i] for i in keep] for phone, vals in self.featmap.items()}

    def __init__(self, df, vocab, group_col="ipa", filter_features=True):
        ft = panphon.FeatureTable()
        self.featnames, self.featmap = self.prep_featmap(vocab, ft)
        self.calc_phnvectors(df, group_col)
        if filter_features:
            self._filter_features()

    def project_raw(self, feats):
        W = self.pos_vecs - self.zero_vecs
        raw = feats @ W.T + self.biases[None, :]
        raw = raw * self.scales[None, :]
        return raw

    def project(self, feats):
        raw = self.project_raw(feats)
        return 1.0 / (1.0 + np.exp(-raw))


# Signals
def _melspec_kaldi(y, sr=16000, n_mels=40):
    waveform = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    feats = kaldi.fbank(
        waveform,
        sample_frequency=float(sr),
        frame_length=25.0,
        frame_shift=MEL_FRAME_SHIFT_MS,
        num_mel_bins=n_mels,
        use_power=True,
        use_energy=False,
        dither=0.0,
        snip_edges=False,
    )
    return feats.cpu().numpy()


def _mel_svf(mel_frames, left, right):
    mel_frames = np.asarray(mel_frames, dtype=float)
    n = mel_frames.shape[0]
    signal = np.full(n, np.nan)
    norms = np.linalg.norm(mel_frames, axis=1)
    for t in range(left, n - right):
        denom = norms[t - left] * norms[t + right]
        if denom > 0:
            signal[t] = 1.0 - np.dot(mel_frames[t - left], mel_frames[t + right]) / denom
    finite = np.isfinite(signal)
    if finite.any():
        lo, hi = np.nanmin(signal), np.nanmax(signal)
        if hi > lo:
            signal[finite] = (signal[finite] - lo) / (hi - lo)
    return signal


def _mel_svf_signal(audio: np.ndarray, left: int, right: int, target_len: int) -> np.ndarray:
    mel = _melspec_kaldi(audio)
    sig = _mel_svf(mel, left=left, right=right)
    if len(sig) == 0 or target_len == 0:
        return np.full(target_len, np.nan, dtype=np.float32)
    indices = np.round(np.linspace(0, len(sig) - 1, target_len)).astype(int)
    return sig[indices].astype(np.float32)


def _delta(proj, offset):
    """Compare phonological projections at t and t+offset using cosine distance."""
    T = proj.shape[0]
    delta = np.full(T, np.nan)
    for t in range(T - offset):
        delta[t] = cos_dist(proj[t], proj[t + offset])
    return delta


def _fwd_contrast(proj_ipa, proj_r1, W_r1_to_ipa, lookahead):
    """cos(fwd_pred[t], ipa[t+lookahead]) - cos(fwd_pred[t], ipa[t]).

    Peaks at boundaries where the forward prediction matches the upcoming
    phone better than the current one.
    """
    T = proj_ipa.shape[0]
    fwd_proj = proj_r1 @ W_r1_to_ipa
    contrast = np.full(T, np.nan)
    for t in range(T - lookahead):
        contrast[t] = cos_dist(fwd_proj[t], proj_ipa[t]) - cos_dist(
            fwd_proj[t], proj_ipa[t + lookahead]
        )
    return contrast


def _bwd_contrast(proj_ipa, proj_l1, W_l1_to_ipa, lookbehind):
    """cos(bwd_pred[t], ipa[t-lookbehind]) - cos(bwd_pred[t], ipa[t]).

    Peaks at boundaries where the backward prediction matches the preceding
    phone better than the current one.
    """
    T = proj_ipa.shape[0]
    bwd_proj = proj_l1 @ W_l1_to_ipa
    contrast = np.full(T, np.nan)
    for t in range(lookbehind, T):
        contrast[t] = cos_dist(bwd_proj[t], proj_ipa[t]) - cos_dist(
            bwd_proj[t], proj_ipa[t - lookbehind]
        )
    return contrast


def _shift_signal(signal, shift_frames):
    """Shift a 1D signal in frame units, padding exposed positions with NaN."""
    if shift_frames == 0:
        return signal.copy()

    shifted = np.full(signal.shape, np.nan)
    if abs(shift_frames) >= len(signal):
        return shifted

    if shift_frames > 0:
        shifted[shift_frames:] = signal[:-shift_frames]
    else:
        shifted[:shift_frames] = signal[-shift_frames:]
    return shifted


class Segmenter:
    COMBINED_SIGNALS = (
        "frame_delta",
        "fwd_delta",
        "bwd_delta",
        "fwd_contrast",
        "bwd_contrast",
        "mel_svf",
    )
    COMBINED_SIGNAL_KWARGS = {
        "frame_delta": {"offset": 1},
        "fwd_delta": {"offset": 1},
        "bwd_delta": {"offset": 2},
        "fwd_contrast": {"lookahead": 2},
        "bwd_contrast": {"lookbehind": 2},
        "mel_svf": {"left": 1, "right": 1},
    }
    COMBINED_SIGNAL_SHIFTS = {
        "frame_delta": 0,
        "fwd_delta": 0,
        "bwd_delta": 1,
        "fwd_contrast": 1,
        "bwd_contrast": -2,
        "mel_svf": 0,
    }
    COMBINED_DROP_K = 2
    COMBINED_PROMINENCE = 0.001

    def __init__(
        self, timit_train_df, silence_detector_path="logistic_regression_silence_detector.joblib"
    ):
        """Build segmenter from per-phone training data. (docstring by Claude)

        timit_train_df: DataFrame with one row per phone, requiring columns:
            feat       - WavLM feature vector (np.ndarray) at phone center frame
            ipa        - IPA label for this phone (NaN for silence)
            l_1        - IPA label of the preceding phone
            r_1        - IPA label of the following phone
            audio_path - path to the source audio file
            min        - phone onset time (for sorting phones within utterances)

        Generated by prepare_datasets.py; cached as
        feats/timit-wavlm-large-24-center-featslice.pkl.
        """
        self.silence_handler = SilenceHandler(silence_detector_path)

        pv_train_df = timit_train_df[~timit_train_df.ipa.isna()]
        vocab = pv_train_df.ipa.unique().tolist()

        self.pv_ipa = PhonologicalVectors(pv_train_df, vocab, group_col="ipa")  # current phone
        self.pv_l1 = PhonologicalVectors(pv_train_df, vocab, group_col="l_1")  # previous phone
        self.pv_r1 = PhonologicalVectors(pv_train_df, vocab, group_col="r_1")  # next phone

        df_sorted = timit_train_df[~timit_train_df.ipa.isna()].sort_values(["audio_path", "min"])
        same_utt = df_sorted.audio_path.values[:-1] == df_sorted.audio_path.values[1:]
        prev_feats = np.stack(
            df_sorted.feat.values[:-1][same_utt]
        )  # previous phone center feature
        curr_feats = np.stack(df_sorted.feat.values[1:][same_utt])  # next phone center features

        proj_ipa_curr = self.pv_ipa.project_raw(
            curr_feats
        )  # what the current phone thinks it should look like
        proj_ipa_prev = self.pv_ipa.project_raw(
            prev_feats
        )  # what the previous phone thinks it should look like
        proj_r1_prev = self.pv_r1.project_raw(
            prev_feats
        )  # what the previous phone thinks the current phone should look like
        proj_l1_next = self.pv_l1.project_raw(
            curr_feats
        )  # what the current phone thinks the previous phone should look like

        self.W_r1_to_ipa = np.linalg.lstsq(proj_r1_prev, proj_ipa_curr, rcond=None)[0]
        self.W_l1_to_ipa = np.linalg.lstsq(proj_l1_next, proj_ipa_prev, rcond=None)[0]

    def _signal(
        self,
        name: str,
        proj_ipa: np.ndarray,
        proj_r1: np.ndarray,
        proj_l1: np.ndarray,
        waveform_np: np.ndarray,
    ) -> np.ndarray:
        kw = self.COMBINED_SIGNAL_KWARGS[name]
        if name == "frame_delta":
            return _delta(proj_ipa, kw["offset"])
        if name == "fwd_delta":
            return _delta(proj_r1, kw["offset"])
        if name == "bwd_delta":
            return _delta(proj_l1, kw["offset"])
        if name == "fwd_contrast":
            return _fwd_contrast(proj_ipa, proj_r1, self.W_r1_to_ipa, kw["lookahead"])
        if name == "bwd_contrast":
            return _bwd_contrast(proj_ipa, proj_l1, self.W_l1_to_ipa, kw["lookbehind"])
        if name == "mel_svf":
            return _mel_svf_signal(
                waveform_np, left=kw["left"], right=kw["right"], target_len=proj_ipa.shape[0]
            )
        raise ValueError(f"Unknown signal: {name}")

    def _combined_signal(self, proj_ipa, proj_r1, proj_l1, waveform_np):
        components = []
        for signal_name in self.COMBINED_SIGNALS:
            sig = self._signal(signal_name, proj_ipa, proj_r1, proj_l1, waveform_np)
            shifted = _shift_signal(sig, self.COMBINED_SIGNAL_SHIFTS[signal_name])
            ranked = shifted - np.nanmin(shifted)
            components.append(ranked)

        stacked = np.stack(components, axis=0)  # (n_signals, T)
        if self.COMBINED_DROP_K > 0:
            stacked = np.sort(stacked, axis=0)[self.COMBINED_DROP_K :]

        return np.prod(stacked, axis=0)

    def segment(self, wavlm_feats, waveform_np, use_combined=True, snap_silence=True):
        proj_ipa = self.pv_ipa.project_raw(wavlm_feats)
        proj_r1 = self.pv_r1.project_raw(wavlm_feats)
        proj_l1 = self.pv_l1.project_raw(wavlm_feats)

        if use_combined:
            signal = self._combined_signal(proj_ipa, proj_r1, proj_l1, waveform_np)
            prominence = self.COMBINED_PROMINENCE
        else:
            # Careful! This is tuned on a subset of VoxAngeles
            signal = _shift_signal(
                _fwd_contrast(proj_ipa, proj_r1, self.W_r1_to_ipa, lookahead=1), shift_frames=1
            )
            prominence = 0.1

        preds = find_peaks(signal, prominence=prominence)[0]
        silence_mask = self.silence_handler.predict_silence_mask(wavlm_feats)
        if snap_silence:
            preds = self.silence_handler.handle_silence(
                preds=preds,
                silence_mask=silence_mask,
                snap_tolerance=2,
            )

        return preds