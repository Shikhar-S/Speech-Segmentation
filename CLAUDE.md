# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PhoneBench** is a phonetic model benchmarking framework built on PyTorch Lightning + Hydra. It evaluates phone recognition and segmentation across diverse datasets and phonetic representations (IPA, ARPAbet, etc.). The main model family is **PowSM** (Phoneme-oriented Weighted Speech Model), a CTC-Attention hybrid encoder-decoder.

## Environment Setup

**Important:** Python commands (training, inference, tests) require a GPU node and an active environment.

**Step 1 — Get an interactive GPU node** (max 30 min on all clusters):

| Cluster | Account | Partition |
|---|---|---|
| Delta | `bbjs-delta-gpu` | `gpuA100x4-interactive` or `gpuA40x4-interactive` |
| Delta-AI | `bbjs-dtai-gh` | `ghx4-interactive` |
| Babel | *(none)* | `debug` |

```bash
# Example for Delta (adjust -A and -p per cluster/partition):
srun -A bbjs-delta-gpu -p gpuA100x4-interactive --gpus-per-node=1 -c 16 --mem=32GB --time=0:30:00 --pty bash
```

**Step 2 — Install and activate environment:**

```bash
make install        # auto-detects x86 vs aarch64
make install-x86    # force Delta / Babel
make install-dai    # force Delta-AI

# Activate on subsequent sessions (make cannot activate your shell):
source .venv/bin/activate
```

> **Delta-AI only:** `flash_attn_3` (Hopper build) is installed automatically from the
> pre-built egg at `/work/nvme/bbjs/sbharadwaj/powsm/flash-attention/hopper/dist/`.
> If the egg is missing, rebuild with:
> `cd /work/nvme/bbjs/sbharadwaj/powsm/flash-attention/hopper && python setup.py bdist_egg`

## Sanity Check Before Large Runs

Before submitting SLURM jobs, always run the recipe module directly to catch import/data errors early:

```bash
python -m src.recipe.segmentation.model_module
python -m src.recipe.phone_recognition.model_module
```

## General Instruction on Coding Style
Prefer to make new files or functions with minimal changes to original code. Do not bloat the code with unnecessary try catch. Be biased towards simplicity, but if there are any major decisions be proactive to ask the user.

- **Extract helpers with descriptive names.** Non-trivial logic embedded in orchestration methods should be pulled into named helpers. Names should document intent. Helpers should hide complexity, not rename simplicity or introduce unnecessary layers — extract when the logic is non-trivial, not just to give a readable name to something already readable inline.
- **Guard clause first.** Handle the no-op or trivial case with an early return at the top; keep the main logic unindented.
- **Preserve caller uniformity.** Side effects that affect only part of a shared structure should be encapsulated in helpers so the caller can treat all cases identically, without special-casing.
- **Minimal lines, no redundant intermediates.** Prefer dense, direct expressions over named temporaries and explanatory comments for obvious steps.

## Architecture

### Configuration System (Hydra)

All config lives in `configs/`. The entry point is `configs/main.yaml`, which is composed with experiment overrides. Experiment configs in `configs/experiment/` are organized as:
- `train/` – training runs
- `inference/` – inference-only runs

### Execution Flow

`src/main.py` → `src/core/task.py` → PyTorch Lightning `Trainer`

`task.py` orchestrates train/test/predict phases, checkpoint loading, and distributed inference. It uses `src/utils/instantiators.py` to dynamically instantiate datamodules, models, callbacks, and loggers from Hydra config.

### Model Stack (`src/model/powsm/`)

**`powsm_model.py`** – Main Lightning module. Components (all optional/swappable via config):
- `frontend.py` – Audio feature extraction (Fbank, etc.)
- `specaug.py` – SpecAugment data augmentation
- `e_branchformer.py` – Default encoder (EBranchformer)
- `transformer_decoder.py` – Attention decoder
- `ctc.py` – CTC loss module

### Data Layer (`src/data/`)

Datasets load from Kaldi-style ark/scp files (`kaldi_dataset.py`) or JSON. Key datasets: TIMIT, Buckeye, CMU L2Arctic, EDACC, FLEURS, Vaani, UltraSuite. The TIMIT datamodule supports mixing Epitran (IPA) labels with original ARPAbet labels via `epitran_mix_ratio`.

### Recipe Modules (`src/recipe/`)

Task-specific Lightning modules (model + data glue):
- `phone_recognition/` – Phone recognition models and error analysis
- `segmentation/` – Segmentation models (`SegmentationModel`, `SegmentationLoss`, `SegmentationInference`), evaluation

### Distributed Inference (`src/core/distributed_inference.py`)

Inference experiments use a custom distributed system instead of Lightning's built-in predict loop. It is triggered when `distributed_predict: True` is set in an experiment config, which causes `task.py` to call `run_distributed_inference_()` instead of `trainer.predict()`.

**Two-level parallelism:**
1. **SLURM array jobs** – The dataset is sharded across `SLURM_ARRAY_TASK_COUNT` jobs. Each job processes `N / num_slurm_tasks` items. Set via `--array=0-N` in the sbatch script.
2. **`num_workers` per job** – Each SLURM task spawns this many Python processes (via `multiprocessing.spawn`), each pinned to a GPU (`cuda:0`, `cuda:1`, …, cycling if more workers than GPUs).

**Data flow:** Each worker independently instantiates the full inference object (via `hydra.utils.instantiate(inference_config, device=device)`) and iterates over its chunk of the dataset. Results are written to a per-job JSONL file: `<out_file_base>.<SLURM_TASK_ID>.jsonl`. If `cache_path` is set on the inference runner, it is also sharded by job and worker.

**Output format** (one JSON object per line):
```json
{"<dataset_idx>": {"pred": "<model output>", "passthrough": {"utt_id": "...", "phones": [...]}}}
```

**Inference experiment config skeleton** (`configs/experiment/inference/`):
```yaml
# @package _global_
defaults:
  - override /data: your_datamodule
  - override /logger: csv

distributed_predict: True   # triggers distributed inference path

inference:
  num_workers: 15            # workers per SLURM task
  passthrough_keys: ["utt_id", "phones"]   # dataset keys copied to output
  out_file: ${paths.output_dir}/preds.json # .jsonl extension used automatically
  inference_runner:
    _target_: src.model.powsm.powsm_inference.build_powsm_inference
    device: cuda             # auto | cpu | cuda
    beam_size: 5
    ctc_weight: 0.3
  inference_call_args:       # static args merged with (and overridden by) dataset item keys
    task_sym: <pr>
    lang_sym: <eng>
```

**DataModule contract for inference:** Must implement `predict_dataloader()` returning a dataset whose `__getitem__` yields a dict. The dict must include `speech` (raw waveform tensor). Any key in `inference_call_args` can be overridden per-sample by including it in the dataset item dict.

**Merging sharded outputs:** After all SLURM tasks finish, collect `*.0.jsonl`, `*.1.jsonl`, … and merge.

### Metrics (`src/metrics/`)

- `phone_recognition.py` – CER / PER. Key class: `PhoneRecognitionEvaluator(normalize_ipa=True)`.
  - `evaluator.evaluate(utt_data, compute_inventory=False, tqdm_enabled=False)` → `(PhoneRecognitionSummary, instance_metrics)`
  - `utt_data` format: `{utt_id: {"prediction": str, "transcription": str}}`
  - `instance_metrics` format: `{utt_id: {"pfer": float, "fer": float, "fed": float, "per": float}}`
  - `PhoneRecognitionSummary` fields: `N`, `phones`, `PER`, `FER`, `FED`, `PFER`, `SUB`, `INS`, `DEL`
  - Reference phone count: `len(evaluator.dst.fm.ipa_segs(evaluator._prepare(ref)))`
- `segmentation_evaluator.py` – Alignment-based metrics

## Code Conventions

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
The full quick-reference is in **`docs/style.md`** — consult it for any design decision.

Key rules:
- **Line length:** 80 (Black + Google standard)
- **Imports:** 4 groups (future → stdlib → third-party → local); no relative imports
- **Naming:** `lower_with_under` for functions/vars, `CapWords` for classes, `CAPS` for constants
- **Docstrings:** Google format (`Args:` / `Returns:` / `Raises:` / `Yields:` sections)
- **Type annotations:** required on all public functions and complex logic
- **Defaults:** never mutable (`list`, `dict`) — use `None` sentinel
- **Exceptions:** specific built-in types; no bare `except:`
- **No `staticmethod`** — write module-level functions instead
- **Strings in loops:** `"".join(...)`, never `+=`
- **Checks:** `if x is None:`, `if items:`, `if flag:` — never `== None` or `== True`
- **Files:** always use `with` context manager
- **Docstring coverage:** 80% minimum (interrogate)
- Config files use YAML with `_target_` for Hydra `instantiate()` calls

## Cluster / Job Submission

SLURM batch scripts: `scripts/daixpr.batch`, `scripts/deltaxpr.batch`

## Analysis Notebooks & Error Analysis Utils

### `src/recipe/phone_recognition/local/error_analysis_utils.py`

Pure-function utility module for phonetic error analysis.

**Seven sections (all public, no main/argparse):**

| Section | Key functions / constants |
|---|---|
| 1 — IPA/Phonetics | `is_diacritic_char`, `strip_diacritics`, `count_diacritics`, `segment_ipa`, `parse_predicted_transcript`, `lang_name`, `clean_ipa` |
| 2 — Data Loading | `load_jsonl`, `load_jsonl_shards`, `load_dataset_predictions`, `load_train_langs`; constants `IPAPACK_YAML`, `RUNS_DIR`, `DATASETS`, `VA_AUDIO_PATTERN`, `TUSOM_AUDIO_PATTERN`, `DEFAULT_TRAIN_SPLITS`, `LANG_FN` |
| 3 — Alignment | `align_phones(ref_segs, hyp_segs) → [(op, ref, hyp)]` ops: C/S/D/I, pure-Python DP; `normalized_edit_distance`; `phone_confusion_matrix(df, evaluator, ...)` — evaluator is explicit |
| 4 — Metrics | `compute_metrics(dataset_name, preds) → pd.DataFrame` — creates own evaluator internally |
| 5 — Per-utt DataFrame | `audio_duration(utt_id, pattern)`, `build_utt_dataframe(preds, evaluator, audio_fn)`, `df_to_entries(df)` |
| 6 — Error Analysis | `analyze_accent_deafness`, `analyze_diacritic_gap`, `analyze_substitutions`, `analyze_feature_errors`, `analyze_annotation_consistency` |
| 7 — Report | `format_report(entries, accent, diacritics, substitutions, features, consistency) → str` |

**`entries` format** (used by all Section 6 functions):
```python
{"utt_id": str, "lang": str, "split": str, "ref_str": str, "pred_str": str,
 "ref_phones": list[str], "pred_phones": list[str]}
```
Use `df_to_entries(df)` to convert a `build_utt_dataframe` result to this format.

### `src/recipe/phone_recognition/local/rq4.ipynb`

Per-utterance analysis notebook for VoxAngeles, DoReCo, and TUSOM datasets. Imports everything from `error_analysis_utils` via `from ... import *`. Session-specific constants (MODEL, PLOT_LANG, K, N_TOP, etc.) and `evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)` are defined in Cell 1.

**Audio paths (read-only — never write to these locations):**
- VoxAngeles: `/work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_voxangeles/recording/{langcode}/{utt_id}.wav`
- TUSOM: `/work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_tusom2021/data/wav/{utt_id}.wav`

**Decode outputs:** `exp/runs/decodedv3.<dataset>/<model>/transcription.*.jsonl`
- JSONL format (one object per line): `{idx: {"pred": [{"processed_transcript": str, ...}], "passthrough": {"utt_id": str, "target": str, "lang_sym": str, ...}}}`
- VoxAngeles language key: `utt_id.split("-")[0]` (ISO 639-3 code)
- DoReCo language key: `passthrough["lang_sym"]`

**iso639-lang usage:** `from iso639 import Lang; Lang("hrv").name` → `"Croatian"`
