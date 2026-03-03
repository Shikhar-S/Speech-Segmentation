# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PhoneBench** is a phonetic model benchmarking framework built on PyTorch Lightning + Hydra. It evaluates phone recognition across diverse datasets and phonetic representations (IPA, ARPAbet, etc.). The main model family is **PowSM** (Phoneme-oriented Weighted Speech Model), a CTC-Attention hybrid encoder-decoder.

## Environment Setup

```bash
envinit   # activate project environment (alias for: source setup_uv.sh)
```

**Important:** Commands that run Python (training, inference, smoke tests, etc.) require a GPU node
and an active environment. Always:
1. Get an interactive GPU node: `now --time=2:00:00`
2. Activate the environment: `envinit`

`now` is a shell function defined in `~/.bashrc` that calls `srun` with defaults from
`~/.slurm_defaults` (`-A bbjs-delta-gpu -c 16 -p gpuA100x4-interactive --mem=32GB --gpus-per-node=1`).
Pass `--time=2:00:00` (or any duration) as an argument.

**Phone recognition pipeline (scripts/):**
```bash
bash scripts/pr_timit_train.sh      # train on TIMIT with Epitran mixing
bash scripts/pr_decode.sh           # decode/inference on datasets
bash scripts/pr_eval.sh             # evaluate decoded outputs
python scripts/pr_parse_results.py  # aggregate results
python scripts/parse_wandb.py       # extract WandB metrics
```

## Architecture

### Configuration System (Hydra)

All config lives in `configs/`. The entry point is `configs/main.yaml`, which is composed with experiment overrides. Experiment configs in `configs/experiment/` are organized as:
- `train/` – training runs
- `probing/` – probing experiments (format: `task_dataset_model`)
- `inference/` – inference-only runs
- `cascade/` – cascade pipeline experiments

### Execution Flow

`src/main.py` → `src/core/task.py` → PyTorch Lightning `Trainer`

`task.py` orchestrates train/test/predict phases, checkpoint loading, and distributed inference. It uses `src/utils/instantiators.py` to dynamically instantiate datamodules, models, callbacks, and loggers from Hydra config.

### Model Stack (`src/model/powsm/`)

**`powsm_model.py`** – Main Lightning module. Components (all optional/swappable via config):
- `frontend.py` – Audio feature extraction (Fbank, etc.)
- `specaug.py` – SpecAugment data augmentation
- `e_branchformer.py` – Default encoder (EBranchformer)
- `transformer_decoder.py` – Attention decoder
- `ctc.py` – CTC loss module (imports from variant files below)

**CTC variants in `ctc.py`** (the active variant is imported at top of `ctc.py`):
- Standard builtin CTC
- `fixed_articulatory_ctc.py` – Articulatory CTC with fixed phonetic feature distances (currently active)
- Scheduled articulatory CTC – annealed loss weighting during training
- Vectorized articulatory CTC – batch-efficient version
- BRCTC (Balanced Risk CTC)

Articulatory CTC variants use **panphon** phonetic feature distances to weight substitution errors in the CTC loss, encouraging phonetically similar confusions over arbitrary ones.

### Data Layer (`src/data/`)

Datasets load from Kaldi-style ark/scp files (`kaldi_dataset.py`) or JSON. Key datasets: TIMIT, Buckeye, CMU L2Arctic, EDACC, FLEURS, Vaani, UltraSuite. The TIMIT datamodule supports mixing Epitran (IPA) labels with original ARPAbet labels via `epitran_mix_ratio`.

### Recipe Modules (`src/recipe/`)

Task-specific Lightning modules (model + data glue) organized by task: `phone_recognition/`, `forced_alignment/`, `geolocation/`, `l1_classification/`, `l2_assessment/`, `langid/`, `tonal_phone_recognition/`.

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

**Merging sharded outputs:** After all SLURM tasks finish, collect `*.0.jsonl`, `*.1.jsonl`, … and merge. `scripts/pr_parse_results.py` handles downstream aggregation.

### Metrics (`src/metrics/`)

- `phone_recognition.py` – CER / PER. Key class: `PhoneRecognitionEvaluator(normalize_ipa=True)`.
  - `evaluator.evaluate(utt_data, compute_inventory=False, tqdm_enabled=False)` → `(PhoneRecognitionSummary, instance_metrics)`
  - `utt_data` format: `{utt_id: {"prediction": str, "transcription": str}}`
  - `instance_metrics` format: `{utt_id: {"pfer": float, "fer": float, "fed": float, "per": float}}`
  - `PhoneRecognitionSummary` fields: `N`, `phones`, `PER`, `FER`, `FED`, `PFER`, `SUB`, `INS`, `DEL`
  - Reference phone count: `len(evaluator.dst.fm.ipa_segs(evaluator._prepare(ref)))`
- `forced_alignment.py` – Alignment-based metrics
- `zeroshot_eval.py` – Zero-shot phonetic evaluation

## Code Conventions

- **Line length:** 99 (Black)
- **Imports:** isort-sorted
- **Docstring coverage:** 80% minimum (interrogate)
- Config files use YAML with `_target_` for Hydra `instantiate()` calls
- Tags follow `[dataset, model, task]` convention for probing experiments

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
