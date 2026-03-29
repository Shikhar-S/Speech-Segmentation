"""Visualization tool for segmentation boundary analysis.

Generates interactive Plotly figures showing mel spectrograms with predicted
and ground-truth phone boundaries overlaid.  Exports to a single self-contained
HTML file (base64-embedded audio) that can be shared without a Python env,
or to HTML that streams audio from a remote URL (e.g. S3/AWS).

Typical notebook usage::

    from src.recipe.segmentation.local.viz_segmentation import (
        HFAudioSource, SegViz)
    import glob

    audio = HFAudioSource("changelinglab/timit-segment", split="test")
    viz = SegViz(audio_src=audio).from_shards(
        glob.glob("exp/runs/.../seg_*.*.jsonl"))
    viz.show("dr1-mcpm0/sx384")     # inline figure in notebook
    viz.export("report.html")       # shareable HTML
"""

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import datasets
import numpy as np
import plotly.graph_objects as go
import torch
import torchaudio
from tqdm import tqdm

from src.data.segmentation.segmentation_dataset import SegmentationDataset
from src.metrics.segmentation_evaluator import SegmentationEvaluator, SegmentationUnit
from src.recipe.segmentation.local.eval_segmentation import (
    parse_groundtruth,
    parse_predictions,
)

_HF_CACHE = os.getenv(
    "HF_HOME",
    "/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/cache/hf",
)
_HOP_LENGTH = 64
_N_FFT = 1024
_N_MELS = 256


# ─── Data structures ──────────────────────────────────────────────────────


@dataclass
class UtteranceVizData:
    """Parsed data for one utterance, ready for visualization."""

    utt_id: str
    predictions: list[SegmentationUnit]
    ground_truth: list[SegmentationUnit]
    phones: list[str]
    duration: float


# ─── JSONL loading ────────────────────────────────────────────────────────


def load_seg_shards(
    shard_files: list,
    split_filter: str | None = "test",
) -> dict:
    """Load segmentation JSONL shards into a dict keyed by utt_id.

    Args:
        shard_files: List of JSONL shard file paths (str or Path).
        split_filter: Keep only entries whose passthrough["split"] matches.
            Pass None to keep all entries.

    Returns:
        Dict mapping utt_id to UtteranceVizData.
    """
    result = {}
    for filepath in tqdm(shard_files, desc="Loading shards"):
        with open(filepath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for idx, data in json.loads(line).items():
                    if idx == "__error__":
                        continue
                    passthrough = data.get("passthrough", {})
                    if (
                        split_filter is not None
                        and passthrough.get("split") != split_filter
                    ):
                        continue
                    utt_id = passthrough.get("utt_id", str(idx))
                    phones = passthrough.get("phones", [])
                    preds = parse_predictions(data["pred"])
                    gt = parse_groundtruth(passthrough, forced=bool(phones))
                    duration = gt[-1].end if gt else 0.0
                    result[utt_id] = UtteranceVizData(
                        utt_id=utt_id,
                        predictions=preds,
                        ground_truth=gt,
                        phones=phones,
                        duration=duration,
                    )
    return result


# ─── Audio sourcing ───────────────────────────────────────────────────────


class AudioSource(Protocol):
    """Protocol for audio retrieval by utterance ID."""

    def get_audio(self, utt_id: str) -> tuple:
        """Return (waveform_float32_array, sample_rate)."""
        ...


class _DummyTokenizer:
    """Minimal tokenizer stub for audio-only use of SegmentationDataset."""

    def tokens2ids(self, phones: list) -> list:
        return list(range(len(phones)))


class HFAudioSource:
    """Load audio via SegmentationDataset, indexed by utt_id.

    Uses SegmentationDataset for consistent preprocessing (resampling,
    ARPAbet→IPA conversion).  Builds a utt_id → row-index lookup from
    the raw HF column so individual lookups are O(1).

    Args:
        hf_repo: HuggingFace dataset repo (e.g. "changelinglab/timit-segment").
        split: Dataset split to use.
        cache_dir: Local HF cache directory.  Defaults to $HF_HOME if set,
            otherwise the project-standard exp/cache/hf absolute path.
        target_sr: Target sample rate (default 16000).
    """

    def __init__(
        self,
        hf_repo: str,
        split: str,
        cache_dir: str = _HF_CACHE,
        target_sr: int = 16000,
    ):
        hf_ds = datasets.load_dataset(hf_repo, split=split, cache_dir=cache_dir)
        self._seg_dataset = SegmentationDataset(
            hf_ds, _DummyTokenizer(), target_sr=target_sr
        )
        self._index = {uid: i for i, uid in enumerate(hf_ds["utt_id"])}
        self._target_sr = target_sr

    def _get_item(self, utt_id: str) -> dict:
        return self._seg_dataset[self._index[utt_id]]

    def get_audio(self, utt_id: str) -> tuple:
        """Return waveform array and sample rate for the given utt_id.

        Args:
            utt_id: Utterance identifier.

        Returns:
            Tuple of (float32 waveform ndarray, sample rate int).

        Raises:
            KeyError: If utt_id is not found in the dataset.
        """
        return self._get_item(utt_id)["speech"].numpy(), self._target_sr

    def get_text(self, utt_id: str) -> str:
        """Return the orthographic transcript for the given utt_id.

        Args:
            utt_id: Utterance identifier.

        Returns:
            Orthographic transcript string, or empty string if not found.
        """
        return self._get_item(utt_id).get("text", "")


# ─── Audio embedding strategies ───────────────────────────────────────────


class AudioEmbedder(Protocol):
    """Protocol for converting audio to an HTML <audio> src value."""

    def get_src(
        self, utt_id: str, waveform: np.ndarray | None, sr: int
    ) -> str:
        """Return value for the HTML audio element's src attribute."""
        ...


class Base64Embedder:
    """Encode audio as a base64 WAV data URI for self-contained HTML."""

    def get_src(
        self, utt_id: str, waveform: np.ndarray | None, sr: int
    ) -> str:
        """Encode waveform as a base64 WAV data URI.

        Args:
            utt_id: Utterance identifier (unused; present for protocol compat).
            waveform: Float32 audio array.
            sr: Sample rate.

        Returns:
            data:audio/wav;base64,... string, or empty string if no waveform.
        """
        if waveform is None:
            return ""
        buf = io.BytesIO()
        torchaudio.save(buf, torch.from_numpy(waveform).float().unsqueeze(0),
                        sr, format="wav")
        buf.seek(0)
        return "data:audio/wav;base64," + base64.b64encode(buf.read()).decode("ascii")


class UrlEmbedder:
    """Reference audio by remote URL (e.g. S3/AWS).

    Audio is NOT embedded in the HTML; the browser fetches it from the URL.
    The waveform argument to get_src is ignored, so an audio_src is not
    required when using this embedder.

    Args:
        base_url: Base URL prefix (e.g. "https://bucket.s3.amazonaws.com/audio").
        ext: Audio file extension (default "wav").
    """

    def __init__(self, base_url: str, ext: str = "wav"):
        self._base_url = base_url.rstrip("/")
        self._ext = ext

    def get_src(
        self, utt_id: str, waveform: np.ndarray | None, sr: int
    ) -> str:
        """Return remote URL for the given utterance.

        Args:
            utt_id: Utterance identifier; appended to base_url.
            waveform: Ignored (audio is served remotely).
            sr: Ignored.

        Returns:
            Remote URL string.
        """
        return f"{self._base_url}/{utt_id}.{self._ext}"


# ─── Spectrogram ──────────────────────────────────────────────────────────


def compute_mel_spectrogram(
    waveform: np.ndarray, sr: int = 16000
) -> np.ndarray:
    """Compute log-mel spectrogram matching the model frontend parameters.

    Uses n_fft=1024, hop_length=64 (~4ms hop), n_mels=256 for
    high-resolution visualization.

    Args:
        waveform: Float32 audio array of shape (T,).
        sr: Sample rate.

    Returns:
        Log-mel spectrogram of shape (n_mels, n_frames) in dB.
    """
    wav = torch.from_numpy(waveform).float().unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=_N_FFT, hop_length=_HOP_LENGTH, n_mels=_N_MELS,
    )(wav).squeeze(0)
    return torchaudio.transforms.AmplitudeToDB(top_db=80)(mel).numpy()


# ─── Figure helpers ───────────────────────────────────────────────────────


def _prepare_mel(
    utt: UtteranceVizData,
    waveform: np.ndarray | None,
    sr: int,
) -> np.ndarray:
    """Return mel spectrogram, falling back to a blank array if no audio."""
    if waveform is not None:
        return compute_mel_spectrogram(waveform, sr)
    n_frames = max(1, int(utt.duration * sr / _HOP_LENGTH) + 1)
    return np.full((_N_MELS, n_frames), -80.0)


def _boundary_hits(
    evaluator: SegmentationEvaluator,
    predictions: list[SegmentationUnit],
    ground_truth: list[SegmentationUnit],
) -> list[bool]:
    """Return per-prediction hit flags using evaluator boundary extraction.

    A predicted boundary start is a hit if it falls within evaluator.tolerance_sec
    of any ground-truth boundary time.
    """
    gt_times = evaluator._extract_boundary_times(ground_truth)
    return [
        np.abs(gt_times - p.start).min() <= evaluator.tolerance_sec
        for p in predictions
    ]


def _pbe_from_evaluator(
    evaluator: SegmentationEvaluator,
    predictions: list[SegmentationUnit],
    ground_truth: list[SegmentationUnit],
) -> list[float] | None:
    """Return per-phone PBE (ms) via evaluator._compute_metrics, or None.

    Returns None when prediction and GT counts differ (free-mode inference),
    preserving the free-mode uniform-color path in the caller.
    """
    if len(predictions) != len(ground_truth):
        return None
    return [
        evaluator._compute_metrics(p.start, p.end, g.start, g.end)[2] * 1000
        for p, g in zip(predictions, ground_truth)
    ]


def _pbe_color(pbe_ms: float, alpha: float = 1.0, max_ms: float = 40.0) -> str:
    """Map PBE in ms to an rgba color (green = accurate, red = large error).

    Args:
        pbe_ms: Phone Boundary Error in milliseconds.
        alpha: Opacity for the rgba color.
        max_ms: PBE value mapped to pure red.

    Returns:
        CSS rgba color string.
    """
    t = min(pbe_ms / max_ms, 1.0)
    r, g = int(255 * t), int(255 * (1.0 - t))
    return f"rgba({r},{g},0,{alpha})"


def _build_id2phone(tokenizer) -> dict:
    """Build an int→phone reverse lookup from a tokenizer instance.

    Supports XeusPRTokenizer (has .vocab dict) and TokenIDConverter
    (has .token_list).

    Args:
        tokenizer: Any tokenizer with a .vocab dict or .token_list attribute.

    Returns:
        Dict mapping integer token ID to phone string.

    Raises:
        TypeError: If the tokenizer type is not recognised.
    """
    if hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
        return {v: k for k, v in tokenizer.vocab.items()}
    if hasattr(tokenizer, "token_list"):
        return dict(enumerate(tokenizer.token_list))
    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)!r}")


def _resolve_label(
    unit: SegmentationUnit,
    id2phone: dict | None = None,
) -> str:
    """Return string label for a SegmentationUnit.

    Priority: string label → id2phone vocab decode → integer as string.
    """
    label=unit.label
    if id2phone is not None:
        label=id2phone.get(int(unit.label), str(unit.label))
    
    if label=='<unk>':
        label='?'
    return label


def _gt_shapes_annotations(
    ground_truth: list[SegmentationUnit],
    phones: list[str],
    n_mels: int,
) -> tuple[list, list]:
    """Build ground-truth boundary shapes and phone label annotations.

    Args:
        ground_truth: GT segmentation units.
        phones: GT phone label strings aligned to ground_truth.
        n_mels: Number of mel bins (sets line/label y extent).

    Returns:
        Tuple of (shapes, annotations) lists for Plotly layout.
    """
    shapes, annotations = [], []
    for i, gt in enumerate(ground_truth):
        shapes.append(dict(
            type="line",
            x0=gt.start, x1=gt.start, y0=0, y1=n_mels,
            line=dict(color="rgba(0,220,80,0.65)", width=1, dash="dash"),
            layer="above",
        ))
        label = _resolve_label(gt)
        if label:
            annotations.append(dict(
                x=(gt.start + gt.end) * 0.5,
                y=n_mels * 0.07,
                text=label,
                showarrow=False,
                font=dict(size=8, color="white"),
                bgcolor="rgba(0,160,60,0.65)",
                xanchor="center",
            ))
    return shapes, annotations


def _pred_shapes_annotations(
    predictions: list[SegmentationUnit],
    phones: list[str],
    pbe_list: list[float] | None,
    n_mels: int,
    id2phone: dict | None = None,
) -> tuple[list, list]:
    """Build predicted boundary shapes and phone label annotations.

    When pbe_list is provided (forced mode), line and label colors are
    mapped from PBE magnitude.  When None (free mode), uniform red is used.

    Args:
        predictions: Predicted segmentation units.
        phones: GT phone label strings used as positional fallback.
        pbe_list: Per-phone PBE in ms, or None for free-mode output.
        n_mels: Number of mel bins (sets line/label y extent).
        id2phone: Optional int→phone vocab for decoding integer labels
            that fall outside the GT phones range (free-mode overflow).

    Returns:
        Tuple of (shapes, annotations) lists for Plotly layout.
    """
    shapes, annotations = [], []
    for i, pred in enumerate(predictions):
        line_color = (
            _pbe_color(pbe_list[i], alpha=0.9)
            if pbe_list else "rgba(255,60,60,0.85)"
        )
        shapes.append(dict(
            type="line",
            x0=pred.start, x1=pred.start, y0=0, y1=n_mels,
            line=dict(color=line_color, width=1.5),
            layer="above",
        ))
        label = _resolve_label(pred, id2phone)
        if label:
            annotations.append(dict(
                x=(pred.start + pred.end) * 0.5,
                y=n_mels * 0.93,
                text=label,
                showarrow=False,
                font=dict(size=8, color="white"),
                bgcolor=_pbe_color(pbe_list[i], alpha=0.65) if pbe_list
                    else "rgba(200,50,50,0.65)",
                xanchor="center",
            ))
    return shapes, annotations


# ─── Figure builder ───────────────────────────────────────────────────────


def plot_utterance(
    utt: UtteranceVizData,
    waveform: np.ndarray | None = None,
    sr: int = 16000,
    text: str = "",
    tolerance_ms: int = 20,
    id2phone: dict | None = None,
) -> go.Figure:
    """Build an interactive Plotly figure for one utterance.

    Displays a mel spectrogram with predicted boundaries (solid, PBE
    color-coded in forced mode), ground-truth boundaries (green dashed),
    hit/miss tick markers, and phone labels.

    Args:
        utt: Parsed utterance data.
        waveform: Optional float32 audio array; blank spectrogram if None.
        sr: Sample rate.
        text: Orthographic transcript shown as figure subtitle.
        tolerance_ms: Boundary tolerance in ms for hit/miss classification,
            passed to SegmentationEvaluator.
        id2phone: Optional int→phone vocab for decoding integer predicted
            labels in free-mode (when pred count exceeds GT phone count).
            Build from a tokenizer with _build_id2phone(tokenizer).

    Returns:
        Plotly Figure.
    """
    mel = _prepare_mel(utt, waveform, sr)
    n_mels, n_frames = mel.shape
    frame_times = np.arange(n_frames) * _HOP_LENGTH / sr

    evaluator = SegmentationEvaluator(tolerance_ms=tolerance_ms)
    pbe_list = _pbe_from_evaluator(evaluator, utt.predictions, utt.ground_truth)
    hits = _boundary_hits(evaluator, utt.predictions, utt.ground_truth)

    gt_shapes, gt_ann = _gt_shapes_annotations(
        utt.ground_truth, utt.phones, n_mels
    )
    pred_shapes, pred_ann = _pred_shapes_annotations(
        utt.predictions, utt.phones, pbe_list, n_mels, id2phone
    )

    hit_x = [p.start for p, h in zip(utt.predictions, hits) if h]
    miss_x = [p.start for p, h in zip(utt.predictions, hits) if not h]
    tick_y = n_mels - 0.5

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=mel, x=frame_times, y=list(range(n_mels)),
        colorscale="Viridis", showscale=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="rgba(255,60,60,0.9)", width=2), name="Predicted",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="rgba(0,220,80,0.9)", width=1.5, dash="dash"),
        name="Ground Truth",
    ))
    fig.add_trace(go.Scatter(
        x=hit_x, y=[tick_y] * len(hit_x), mode="markers",
        marker=dict(symbol="triangle-down", size=9,
                    color="rgba(0,255,100,0.95)"),
        name=f"Hit (≤{tolerance_ms}ms)",
    ))
    fig.add_trace(go.Scatter(
        x=miss_x, y=[tick_y] * len(miss_x), mode="markers",
        marker=dict(symbol="x", size=8, color="rgba(255,80,80,0.95)",
                    line=dict(width=2)),
        name="Miss",
    ))

    pbe_mean_str = f"  |  PBE {np.mean(pbe_list):.1f}ms" if pbe_list else ""
    text_str = f"<br><sub>{text}</sub>" if text else ""
    fig.update_layout(
        shapes=gt_shapes + pred_shapes,
        annotations=gt_ann + pred_ann,
        title=dict(
            text=(
                f"{utt.utt_id}"
                f"  |  {len(utt.ground_truth)} phones"
                f"  |  {utt.duration:.2f}s"
                f"{pbe_mean_str}"
                f"{text_str}"
            ),
            font=dict(size=11),
        ),
        xaxis=dict(title="Time (s)", range=[0, utt.duration], showgrid=False),
        yaxis=dict(title="Mel bin", showgrid=False),
        height=380,
        margin=dict(l=55, r=20, t=55, b=50),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


# ─── HTML export ──────────────────────────────────────────────────────────

# Placeholders replaced via str.replace to avoid escaping all JS braces.
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Segmentation Visualization</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {
      font-family: monospace;
      background: #0d1117;
      color: #e6edf3;
      margin: 0;
      padding: 16px;
    }
    #nav {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }
    button {
      background: #21262d;
      border: 1px solid #30363d;
      color: #e6edf3;
      padding: 5px 14px;
      cursor: pointer;
      border-radius: 4px;
      font-size: 13px;
    }
    button:hover { background: #30363d; }
    select {
      background: #21262d;
      border: 1px solid #30363d;
      color: #e6edf3;
      padding: 5px 8px;
      max-width: 480px;
      font-family: monospace;
      font-size: 13px;
    }
    #counter { color: #8b949e; font-size: 13px; }
    #audio-player { width: 100%; margin-top: 6px; }
    .legend-hint {
      font-size: 11px;
      color: #8b949e;
      margin-top: 4px;
    }
  </style>
</head>
<body>
  <div id="nav">
    <button id="btn-prev">&#9664; Prev</button>
    <select id="utt-select"></select>
    <button id="btn-next">Next &#9654;</button>
    <span id="counter"></span>
  </div>
  <div id="plot"></div>
  <audio id="audio-player" controls></audio>
  <div class="legend-hint">
    &mdash; Predicted (solid, green=accurate &rarr; red=large PBE)
    &nbsp;&nbsp;
    &mdash; Ground Truth (green dashed)
    &nbsp;&nbsp;
    &larr; &rarr; keyboard navigation
  </div>

  <script>
  const FIGURES = __FIGURES_JSON__;
  const AUDIO_SRCS = __AUDIO_SRCS_JSON__;
  const UTT_IDS = __UTT_IDS_JSON__;
  let current = 0;

  const select = document.getElementById('utt-select');
  UTT_IDS.forEach(function(id, i) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.text = id;
    select.appendChild(opt);
  });

  function show(idx) {
    current = ((idx % UTT_IDS.length) + UTT_IDS.length) % UTT_IDS.length;
    Plotly.react('plot', FIGURES[current].data, FIGURES[current].layout,
                 {responsive: true});
    const ap = document.getElementById('audio-player');
    if (AUDIO_SRCS[current]) {
      ap.src = AUDIO_SRCS[current];
      ap.style.display = 'block';
    } else {
      ap.style.display = 'none';
    }
    select.value = current;
    document.getElementById('counter').textContent =
      (current + 1) + ' / ' + UTT_IDS.length;
  }

  document.getElementById('btn-prev').onclick = function() { show(current - 1); };
  document.getElementById('btn-next').onclick = function() { show(current + 1); };
  select.onchange = function() { show(parseInt(select.value)); };
  document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowLeft') show(current - 1);
    if (e.key === 'ArrowRight') show(current + 1);
  });

  show(0);
  </script>
</body>
</html>
"""


def _load_audio_and_text(
    utt: UtteranceVizData, audio_src: AudioSource | None
) -> tuple[np.ndarray | None, int, str]:
    """Load waveform and text for one utterance from audio_src.

    Returns (waveform, sr, text).  Any missing field falls back to its zero
    value so callers need no special-casing.
    """
    if audio_src is None:
        return None, 16000, ""
    try:
        waveform, sr = audio_src.get_audio(utt.utt_id)
        text = audio_src.get_text(utt.utt_id) if hasattr(audio_src, "get_text") else ""
        return waveform, sr, text
    except KeyError:
        return None, 16000, ""


def export_html(
    utterances: dict,
    output_path,
    audio_src: AudioSource | None = None,
    embedder: AudioEmbedder | None = None,
    max_utterances: int | None = None,
    tolerance_ms: int = 20,
    id2phone: dict | None = None,
    save_n=20,
) -> None:
    """Export utterances to a self-contained interactive HTML file.

    The output can be opened in any browser without a Python environment.
    With Base64Embedder (default), audio is embedded in the file; with
    UrlEmbedder, the browser fetches audio from the provided remote URL.

    Args:
        utterances: Dict of UtteranceVizData keyed by utt_id.
        output_path: Path for the output HTML file (str or Path).
        audio_src: Optional AudioSource used to load audio for the spectrogram
            and (when using Base64Embedder) for embedding.
        embedder: AudioEmbedder strategy.  Defaults to Base64Embedder when
            audio_src is provided.  Use UrlEmbedder to reference remote audio
            without loading it locally.
        max_utterances: If set, only include the first N utterances.
        tolerance_ms: Boundary tolerance in ms for hit/miss tick markers.
    """
    if embedder is None and audio_src is not None:
        embedder = Base64Embedder()

    utt_list = list(utterances.values())
    utt_list = utt_list[:save_n]  # quick limit for faster export during development
    if max_utterances is not None:
        utt_list = utt_list[:max_utterances]

    figures_data, audio_srcs = [], []
    for utt in tqdm(utt_list, desc="Building figures"):
        waveform, sr, text = _load_audio_and_text(utt, audio_src)
        fig = plot_utterance(utt, waveform, sr, text=text,
                             tolerance_ms=tolerance_ms, id2phone=id2phone)
        figures_data.append(json.loads(fig.to_json()))
        audio_srcs.append(
            embedder.get_src(utt.utt_id, waveform, sr) if embedder else None
        )

    html = (
        _HTML_TEMPLATE
        .replace("__FIGURES_JSON__", json.dumps(figures_data))
        .replace("__AUDIO_SRCS_JSON__", json.dumps(audio_srcs))
        .replace("__UTT_IDS_JSON__", json.dumps([u.utt_id for u in utt_list]))
    )
    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved {len(utt_list)} utterances → {output_path}")


# ─── Convenience class ────────────────────────────────────────────────────


class SegViz:
    """Convenience wrapper for segmentation visualization in notebooks.

    Example::

        viz = SegViz(audio_src=HFAudioSource("changelinglab/timit-segment",
                                             split="test"))
        viz.from_shards(glob.glob("exp/runs/.../seg_*.*.jsonl"))
        viz.show("dr1-mcpm0/sx384")
        viz.export("report.html")

    Args:
        audio_src: Optional AudioSource for loading audio.
        embedder: AudioEmbedder for HTML export.  Defaults to Base64Embedder
            when audio_src is provided.
        tolerance_ms: Boundary tolerance in ms for hit/miss tick markers.
        tokenizer: Optional tokenizer for decoding integer predicted labels
            in free-mode output.  Accepts XeusPRTokenizer (has .vocab) or
            TokenIDConverter (has .token_list).  Use _build_id2phone directly
            if you have a pre-built dict.
    """

    def __init__(
        self,
        audio_src: AudioSource | None = None,
        embedder: AudioEmbedder | None = None,
        tolerance_ms: int = 20,
        tokenizer=None,
    ):
        self.audio_src = audio_src
        self.embedder = embedder
        self.tolerance_ms = tolerance_ms
        self._id2phone = _build_id2phone(tokenizer) if tokenizer else None
        self._utterances: dict = {}

    def set_tolerance(self, tolerance_ms: int) -> "SegViz":
        """Set the boundary hit tolerance used for tick markers.

        Args:
            tolerance_ms: Tolerance in milliseconds.

        Returns:
            self (for method chaining).
        """
        self.tolerance_ms = tolerance_ms
        return self

    def from_shards(
        self,
        shard_files: list,
        split_filter: str | None = "test",
    ) -> "SegViz":
        """Load utterances from JSONL shard files.

        Args:
            shard_files: List of JSONL shard paths.
            split_filter: Only keep entries matching this split label.

        Returns:
            self (for method chaining).
        """
        self._utterances = load_seg_shards(shard_files, split_filter)
        print(f"Loaded {len(self._utterances)} utterances from "
              f"{len(shard_files)} shards")
        return self

    def show(self, utt_id: str) -> go.Figure:
        """Display a figure for one utterance in a Jupyter notebook.

        Args:
            utt_id: Utterance identifier to visualize.

        Returns:
            Plotly Figure (auto-displayed by Jupyter via _repr_html_).
        """
        utt = self._utterances[utt_id]
        waveform, sr, text = _load_audio_and_text(utt, self.audio_src)
        return plot_utterance(
            utt, waveform, sr, text=text, tolerance_ms=self.tolerance_ms,
            id2phone=self._id2phone,
        )

    def export(
        self,
        output_path,
        max_utterances: int | None = None,
    ) -> None:
        """Export all loaded utterances to a shareable HTML file.

        Args:
            output_path: Path for the output HTML file.
            max_utterances: If set, limit to the first N utterances.
        """
        export_html(
            self._utterances,
            output_path,
            audio_src=self.audio_src,
            embedder=self.embedder,
            max_utterances=max_utterances,
            tolerance_ms=self.tolerance_ms,
            id2phone=self._id2phone,
        )
