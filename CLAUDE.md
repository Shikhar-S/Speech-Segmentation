# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A phonetic modeling research framework built on PyTorch Lightning + Hydra. Covers phone recognition, forced alignment, and segmentation across diverse datasets and phonetic representations (IPA, ARPAbet, etc.). Primary encoders: **XEUS** (cross-lingual speech encoder) and **PhoneticXEUS** (XEUS with interCTC self-conditioning at layers 4, 8, 12). Legacy encoder: **PowSM** (EBranchformer CTC-Attention hybrid). Forced alignment via **MFA** (batch and single-utterance).

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
make install          # auto-detects x86 (Delta/Babel) vs aarch64 (Delta-AI)
source .venv/bin/activate
```

> **Delta-AI only:** `flash_attn_3` (Hopper build) is installed automatically from the
> pre-built egg at `/work/nvme/bbjs/sbharadwaj/powsm/flash-attention/hopper/dist/`.
> If missing, rebuild: `cd /work/nvme/bbjs/sbharadwaj/powsm/flash-attention/hopper && python setup.py bdist_egg`

## Sanity Check Before Large Runs

Before submitting SLURM jobs, always run the recipe module directly to catch import/data errors early:

```bash
python -m src.recipe.segmentation.model_module
python -m src.recipe.phone_recognition.model_module
python -m src.recipe.segment_recognize.model_module
```

## General Instruction on Coding Style
Prefer to make new files or functions with minimal changes to original code. Do not bloat the code with unnecessary try catch. Be biased towards simplicity, but if there are any major decisions be proactive to ask the user.

- **Data model first.** Define the data structure before the algorithm. Eliminate special cases by fixing the shape of the data, not by adding conditionals. If the structure is wrong, the algorithm is irrelevant.
- **Simplicity over generality.** Write the dumbest code that is obviously right. No speculative abstractions, no flexibility nobody asked for, no cleverness for its own sake. Every extra line is a liability — if 50 lines solve it, 500 lines is a confession.
- **Surgical changes.** Touch only what the request requires. No drive-by refactors, no unrelated edits, no vanity cleanup. Every changed line must have a direct reason to exist. Mention unrelated problems; do not start a second project.
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

### Model Stack (`src/model/`)

**Encoders** (swappable via `configs/model/net/`):
- `xeusphoneme/` – **XEUS** (`xeuspr.yaml`) and **PhoneticXEUS** (`phoneticxeus.yaml`, interCTC at layers 4, 8, 12). Primary encoders for current experiments.
- `powsm/` – **PowSM** (EBranchformer CTC-Attention hybrid). Legacy encoder.
- `wav2vec2phoneme/`, `wavlm/`, `whisper/` – Other encoder backends.

**Shared encoder interface** (all nets expose):
- `encode(speech, lengths)` → `(features, feature_lens)`
- `encoder_output_size()` → `int`
- `points_by_frames()` → `int` (audio samples per encoder frame, typically 640)
- `_calc_ctc_loss(features, lens, text, text_lens)` → `(loss, stats)`
- `ctc.ctc_lo` — linear projection to vocab logits

### Data Layer (`src/data/`)

Datasets load from Kaldi-style ark/scp files (`kaldi_dataset.py`) or HuggingFace Hub. Key datasets: TIMIT, Buckeye, VoxAngeles, DoReCo, FLEURS. The TIMIT datamodule supports mixing Epitran (IPA) labels with original ARPAbet labels via `epitran_mix_ratio`.

For joint training, `JointPRSegDataModule` (`joint_prseg_dataset.py`) combines a PR datamodule and a segmentation datamodule with weighted sampling. Batches are `{"segmentation": sub_batch | None, "recognition": sub_batch | None}`.

### Recipe Modules (`src/recipe/`)

Task-specific Lightning modules (model + data glue):
- `phone_recognition/` – Phone recognition models and error analysis
- `segmentation/` – Standalone segmentation (`SegmentationModel`, `SegmentationLoss`, `SegmentationInference`)
- `joint/` – `JointPRSegModel`: monolithic joint PR + segmentation with inline losses
- `segment_recognize/` – `SegmentRecognizeModel`: composable joint PR + segmentation using modular `TaskHead` classes in `heads/`. Preferred for new experiments. See below.

### Composable Loss System (`src/recipe/segment_recognize/`)

`SegmentRecognizeModel` replaces `JointPRSegModel` with a composable architecture. Task heads are self-contained `TaskHead(nn.Module)` classes that own their parameters, criteria, metric trackers, and eval logic.

**`heads/base.py` — `TaskHead`**: Base class. Subclasses implement `forward()` → `{"loss": tensor, ...}` and optionally `eval_metrics()`. Each head has `train_loss`/`val_loss` MeanMetric trackers and a `log_output()` helper.

**Concrete heads** (`heads/`):
- `BCEBoundaryHead` — owns `boundary_head: nn.Linear(D, 1)`, `BoundaryLoss` criterion, `SegmentationEvaluator` for P/R/F1/R-value
- `FASegmentationHead` — uses `net.ctc.ctc_lo` (via `**ctx`) for frame-level forced-alignment loss
- `CTCRecognitionHead` — delegates to `net._calc_ctc_loss` (via `**ctx`)

**Model composition**: Heads registered in `nn.ModuleDict` (`seg_losses`, `pr_losses`). Training step iterates over each group. Heads are instantiated from Hydra config with runtime-injected `encoder_dim`, `effective_pbf`, `audio_sr`.

**Adding a new head**: Write a `TaskHead` subclass, add to config under `seg_losses` or `pr_losses`. No model code changes.

**Configs**: `configs/model/segment_recognize.yaml`, `configs/experiment/train/sr_*.yaml`. Reuses `configs/data/joint_prseg.yaml` (same batch format).

### Distributed Inference (`src/core/distributed_inference.py`)

Triggered by `distributed_predict: True` in experiment config. Two-level parallelism: SLURM array jobs shard the dataset, each job spawns `num_workers` processes pinned to GPUs. Results written to per-job JSONL files (`<out_file_base>.<SLURM_TASK_ID>.jsonl`). See `configs/experiment/inference/` for config examples.

### Metrics (`src/metrics/`)

- `phone_recognition.py` – `PhoneRecognitionEvaluator` for CER/PER/FER/FED
- `segmentation_evaluator.py` – `SegmentationEvaluator` for boundary P/R/F1/R-value (20ms tolerance)

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

## Experiment Log

Experiment results, run paths, and evaluation tables: [`exp/experiment_log.md`](exp/experiment_log.md)

Aggregated evaluation metrics (all experiments x datasets): [`exp/all_eval_results.json`](exp/all_eval_results.json)

## Experimentation Workflow

When running experiments (training, inference, evaluation):

1. **Use `/loop` to monitor SLURM jobs.** After submitting jobs, set up a recurring check (e.g. `/loop 5m check job status`) that:
   - Polls `sacct -j <JOB_ID>` for completion/failure
   - On failure: reads SLURM logs (`exp/slurm_logs/<jobid>_*.out`), diagnoses the error, fixes the code, resubmits
   - On success: proceeds to the next step (inference after training, evaluation after inference)
   - Stops looping once the pipeline is complete

2. **Maintain `exp/experiment_log.md`.** After every experiment round:
   - Record run directories, SLURM job IDs, configs used, and key hyperparameters
   - Record evaluation results with the full comparison table
   - Note any code changes or bug fixes applied during the run
   - Commit the updated log

3. **Pipeline order:** train → monitor → inference (all datasets) → evaluate → log results → compare with prior rounds

## Cluster / Job Submission

SLURM batch scripts: `scripts/daixpr.batch` (Delta-AI), `scripts/deltaxpr.batch` (Delta), `scripts/babel.batch` (Babel), `scripts/daixpr_inference.batch` (inference)

