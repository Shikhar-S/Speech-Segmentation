# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reference implementation and evaluation benchmark for the paper
**"Phone Segmentation and Recognition through Phonological Activation Mapping"**
(Bharadwaj et al., 2026, [arXiv:2607.09020](https://arxiv.org/abs/2607.09020)).
The method is **SPAM** (S3M-based Phonological Activation Mapping): a
gradient-descent-free phone segmenter + recognizer built on frozen
self-supervised speech features (WavLM-large layer 24), evaluated on
8 segmentation datasets and the 6 PRiSM phone-recognition sets.

The repo is built on PyTorch Lightning + Hydra and contains three families of
systems, all sharing one entrypoint (`src/main.py`) and one scoring stack:

| Family | What | Code |
|---|---|---|
| **SPAM (ours)** | Inference-only wrapper around the external `phonological-posteriogram` library and the released `juice500/*-phonemodel` HF checkpoints | `src/model/phonvec/` |
| **Supervised baselines** | Fully fine-tuned WavLM-large with CTC / FCE / BCE heads, trained on TIMIT | `src/model/wavlm/`, `src/recipe/segment_recognize/` |
| **Toplines** | Montreal Forced Aligner on ground-truth phones, and MFA cascaded behind KoelLabs-XLSR or PhoneticXEUS recognizers | `src/model/mfa/`, `src/model/koel/`, `src/model/xeusphoneme/` + `src/model/powsm/` |

## Paper results → reproduction scripts

Every row below is a self-contained SLURM script (`sbatch scripts/<name>.sh`)
that runs distributed inference **and** scoring. Detailed usage, dataset lists
and output paths are in `README.md`; recorded numbers and job IDs are in
`SPAM_experiment_log.md`.

| Paper result | Scripts |
|---|---|
| Table I, SPAM row (R-value) | `run_spam_segment.sh` |
| Table I, CTC / FCE / BCE rows | `run_{bce,fce}_train.sh`, `run_wavlm_ctc_segment_train.sh` → `run_{bce,fce}_segment.sh`, `run_wavlm_ctc_segment.sh` |
| Table I, GT + MFA topline | `run_mfa_topline.sh` |
| Table I, KoelLabs-XLSR + MFA | `run_koel_mfa_baseline.sh` |
| Table I, PhoneticXeus + MFA (English) | `run_pxeus_recognize.sh` → `run_mfa_align_dump.sh` |
| Table I, PhoneticXeus + MFA (SSNCE Tamil, MFA 2.x) | `run_mfa2_tamil.sh` |
| Table II, SPAM row (PFER, PRiSM) | `run_spam_recognition.sh` (panphon-unrestricted decoding) |
| Table II, CTC / FCE rows | `run_wavlm_ctc_recognition_train.sh` → `run_wavlm_ctc_recognition.sh`, `run_wavlm_fce_recognition.sh` |
| Table III, boundary-signal ablation | `run_arch_ablation.sh` |
| Figure 3, data efficiency | `run_efficiency_{segmentation,recognition}_ablation.sh` |
| Figure 4, S3M × layer | `run_sslwlayer_ablation.sh` |
| Text: oracle-boundary recognition, seen/unseen PFER | `run_phonvec_oracle.sh ... panphon`, then `scripts/eval_seen_unseen.py` |
| Per-language vocab routing (feature, not a paper table) | `run_spam_recognition_novocab.sh`, `scripts/build_phonvec_vocab*.py` |

Table II toplines (KoelLabs-XLSR, PhoneticXeus) are not produced by this repo.

## Environment Setup

Dependencies are declared once in `pyproject.toml` (extras `x86` / `dai` hold the
ESPnet runtime deps); `setup.sh` installs them with `uv sync`. Run scripts source
`scripts/env.sh` to activate `.venv`; they never install anything.


Python commands (training, inference, tests) require a GPU node and an active
environment. MFA toplines additionally need the external micromamba envs
`mfa310` (MFA 3.3.9) and `mfa2` (MFA 2.2.17).

**Step 1 — Get an interactive GPU node** (max 30 min on all clusters):

| Cluster | Account | Partition |
|---|---|---|
| Delta | `bbjs-delta-gpu` | `gpuA100x4-interactive` or `gpuA40x4-interactive` |
| Delta-AI | `bbjs-dtai-gh` | `ghx4-interactive` |
| Babel | *(none)* | `debug` |

```bash
srun -A bbjs-delta-gpu -p gpuA100x4-interactive --gpus-per-node=1 -c 16 --mem=32GB --time=0:30:00 --pty bash
```

**Step 2 — Install and activate environment:**

```bash
make install          # full install; `bash setup.sh core` skips ESPnet (SPAM, MFA, Koel only)
source scripts/env.sh # activates .venv and puts the repo root on PYTHONPATH
```

The `juice500/*-phonemodel` models are public. The `changelinglab/*-segment`
and several `changelinglab/*-pr` datasets are private: run `huggingface-cli login`
once, or export `HF_TOKEN`.

> **Delta-AI only:** `flash_attn_3` (Hopper build) is installed from the egg
> named by `FLASH_ATTN3_EGG`; skipped when unset.

## Sanity Check Before Large Runs

Before submitting SLURM jobs, compose and import the target experiment on the
GPU node to catch config/import/data errors early, e.g.:

```bash
python src/main.py experiment=segment/phonvec data.hf_repo=changelinglab/timit-segment \
    data.predict_split=test paths.output_dir=tmp/sanity inference.num_workers=1
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

All config lives in `configs/`. The entry point is `configs/main.yaml`, composed
with one experiment override. Experiment configs in `configs/experiment/`:
- `segment/` – segmentation inference (phonvec, wavlm_{bce,fce,ctc}, mfa_topline*, mfa_baseline)
- `recognize/` – recognition inference (phonvec*, wavlm_{ctc,fce}, xeuspr*)
- `inference/` – second-stage runners (mfa_align_dump)
- `train/` – the WavLM baseline training runs

Non-Hydra assets under `configs/inference/`: per-language vocab JSONs
(`phonvec_vocab/`), MFA inventory masks (`mfa_mask_vocab/`), and the
segmentation→PRiSM IPA maps used by `scripts/postprocess_for_prism.py`.

### Execution Flow

`src/main.py` → `src/core/task.py` → either a Lightning `Trainer`
(`train/*` experiments) or `src/core/distributed_inference.py`
(everything else, `distributed_predict: True`). On the inference path only
`cfg.data` and `cfg.inference.inference_runner` are instantiated; trainer,
logger and callbacks are composed but unused.

### Model Stack (`src/model/`)

- `phonvec/inference.py` – **SPAM.** `build_phonvec_inference` loads a
  `juice500/*-phonemodel`, extracts S3M features, segments via the model's
  boundary-signal stack, and recognizes each segment against either the fitted
  IPA featmap, a per-language panphon vocab, or the full PanPhon inventory
  (`panphon_unrestricted`). Supports `oracle=true` (GT boundaries).
- `wavlm/` – WavLM-large encoder (`encode`, `forced_align`, `ctc.ctc_lo`) used
  by the supervised baselines.
- `mfa/` – MFA integration: `inference_baseline.py` (per-utterance
  `mfa align_one`, topline or cascade), `align_from_dump.py` (align a
  previously dumped recognizer transcript), `mfa2_align.py` (corpus-level MFA
  2.x for Tamil), `utils.py` (shared helpers).
- `koel/` – KoelLabs XLSR CTC recognizer adapter (cascade front-end).
- `xeusphoneme/` + `powsm/` – PhoneticXEUS recognizer (E-Branchformer + CTC,
  vendored ESPnet-style layers). Used only for the PhoneticXEUS + MFA topline.
  `xeusphoneme/builders.py` also defines `XeusPRTokenizer`, the vocab-file
  tokenizer used by all WavLM configs, and `resources/` holds the IPA vocab
  JSONs referenced by configs.

**Shared encoder interface** (nets used by `segment_recognize`):
`encode(speech, lengths)`, `encoder_output_size()`, `points_by_frames()`,
`_calc_ctc_loss(...)`, `ctc.ctc_lo`.

### Data Layer (`src/data/`)

Dataset preparation and Hub upload scripts (LDC corpora → `*-segment` schema)
live in `scripts/data_prep/`; see docs/data_preparation.md. The transform
registry in `dataset_processing_transforms.py` is keyed by dataset name and
resolves user repo ids through the `SEG_REPO_*` env vars.

Datasets load from the HuggingFace Hub (`changelinglab/*-segment`) or from
Kaldi-style PRiSM dumps (`recognition/prism_preval.py`).
`segmentation/segmentation_dataset.py` provides `SegmentationDataModule`,
`build_segmentation_dataset` and `DummyTokenizer`;
`segmentation/dataset_processing_transforms.py` holds per-dataset label
transforms (`HF_REPO_TRANSFORMS`). `mixed_prseg_dataset.py`
(`build_prseg_datamodule`) combines PR + segmentation sets with weighted
sampling for the WavLM training runs.

### Supervised Baseline Recipe (`src/recipe/segment_recognize/`)

`SegmentRecognizeModel` composes `TaskHead` subclasses from `heads/`
(`BCEBoundaryHead`, `FCESegmentationHead`, `CTCRecognitionHead`) registered
in `seg_losses` / `pr_losses`. Heads own their parameters, criteria and
metrics; `inference.py` wraps a trained checkpoint for distributed inference.
`src/recipe/common/` holds greedy-CTC / forced-alignment decode strategies,
boundary utilities and the error calculator.

### Scoring (`src/metrics/`, `scripts/`)

`src/metrics/types.py` defines `SegmentationUnit`. Scoring uses the external
`phone_metrics` package via `scripts/eval_segmentation.py` (boundary
P/R/F1/R-value, 20 ms tolerance, strict mode, `--strip-outer-silences`,
optional bootstrap CI) and `scripts/eval_recognition.py` (PER / PFER after
`scripts/postprocess_for_prism.py`).

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

Experiment results, run paths, and evaluation tables: [`SPAM_experiment_log.md`](SPAM_experiment_log.md)

## Experimentation Workflow

When running experiments (training, inference, evaluation):

1. **Use `/loop` to monitor SLURM jobs.** After submitting jobs, set up a recurring check (e.g. `/loop 5m check job status`) that:
   - Polls `sacct -j <JOB_ID>` for completion/failure
   - On failure: reads SLURM logs (`exp/slurm_logs/<jobid>_*.out`), diagnoses the error, fixes the code, resubmits
   - On success: proceeds to the next step (inference after training, evaluation after inference)
   - Stops looping once the pipeline is complete

2. **Maintain `SPAM_experiment_log.md`.** After every experiment round:
   - Record run directories, SLURM job IDs, configs used, and key hyperparameters
   - Record evaluation results with the full comparison table
   - Note any code changes or bug fixes applied during the run
   - Commit the updated log

3. **Pipeline order:** train → monitor → inference (all datasets) → evaluate → log results → compare with prior rounds

## Cluster / Job Submission

Do not create new scripts unnecessarily. Each task has a self-contained
`scripts/run_*.sh` SLURM script (own `#SBATCH` header + `python src/main.py
experiment=...` invocation). Submit with `sbatch`, and override defaults on the
CLI:
```bash
sbatch <sbatcharg1> <sbatcharg2> scripts/run_<task>.sh <scriptarg1> <scriptarg2>
```
here `<sbatcharg>` overrides the SLURM header arguments, `<scriptarg>` overrides
the Hydra command-line arguments (last value wins). Each script documents its own
positional args in a header comment.

## Environment Variables

All paths are relative to the repo root; nothing is hardcoded to a cluster.
Every `scripts/run_*.sh` does `cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"`,
so submit with `sbatch` from the repo root or export `PROJECT_ROOT`. `sbatch`
forwards the submitting shell's environment. Every variable has a default except
`XEUSPR_CKPT`; set only the ones the script you run needs:

| Variable | Default | Needed by |
|---|---|---|
| `PROJECT_ROOT` | `$SLURM_SUBMIT_DIR` (set by `.project-root` for `src/main.py`) | everything |
| `HF_HOME` | `exp/cache/hf` | everything (HF datasets + models) |
| `HF_TOKEN` | token saved by `huggingface-cli login` | private `changelinglab/*` datasets |
| `SEG_REPO_<DATASET>` | `changelinglab/<dataset>-segment` | Hub repo id of each segmentation set you prepared (`SEG_REPO_TIMIT`, `SEG_REPO_BUCKEYE`, `SEG_REPO_GTIMIT_L2SIMPLE`, `SEG_REPO_GTIMIT_L2TBNK`, `SEG_REPO_TORGO`, `SEG_REPO_SSNCE`, `SEG_REPO_VOXANGELES`, `SEG_REPO_GTIMIT_THA`) |
| `PRISM_DUMP_DIR` | `exp/data/prism` | Table II recognition (Kaldi `test_*/wav.scp` dumps) |
| `XEUSPR_CKPT` | *(required)* | PhoneticXEUS topline: `run_pxeus_recognize.sh`, `run_mfa2_tamil.sh` |
| `MFA_ROOT_DIR` | `exp/cache/mfa` | MFA toplines |
| `MAMBA_ROOT_PREFIX`, `MAMBA_BIN` | `~/micromamba`, `micromamba` on PATH | micromamba envs for MFA (`mfa310`, `mfa2`); creation recipe in docs/reproduction.md |
| `FLASH_ATTN3_EGG` | unset (skipped) | Delta-AI Hopper build only |

SLURM account/partition in each script header target NCSA Delta; override on
the CLI, e.g. `sbatch -A <acct> -p <partition> scripts/run_spam_segment.sh`.

## Do not pollute my code!
* PROTECTED CODE: Do not change code inside src/core without requiring permission from the user explicitly.
* Put all temporary scripts in tmp/

## DO NOT WASTE RESOURCES
* When making new slurm scripts for launching jobs, use 1 GPU 16 cpu and 32GB memory.
* Use a time limit for new slurm scripts.
