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

- `phone_recognition.py` – CER / PER
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
