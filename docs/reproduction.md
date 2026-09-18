# Reproducing the paper

## Systems

- **SPAM** is the paper's method: a phone segmenter and recognizer built on
  frozen self-supervised speech features, released as `juice500/*-phonemodel`.
- **CTC, FCE, BCE** are supervised baselines: WavLM-large fine-tuned on TIMIT
  with a CTC, frame-wise cross-entropy, or boundary-classification head.
- **MFA** is the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/).
  The toplines align either the ground-truth transcript or the output of a
  phone recognizer: **KoelLabs-XLSR** (`KoelLabs/xlsr-english-01`, a public
  CTC model) or **PhoneticXEUS** (an XEUS-based recognizer; its checkpoint is
  supplied through `XEUSPR_CKPT`).
- **PRiSM** is the phone-recognition benchmark used for Table II. Its test sets
  are read as Kaldi-style directories (`wav.scp`, `text`) under `PRISM_DUMP_DIR`.

Each script runs distributed inference and scoring over all of its datasets,
with one SLURM array task per dataset, and writes `summary.csv` under
`exp/runs/<name>/<jobid>/`. A single shard can be re-run with
`sbatch --array=<index>`; the index map is given in each script's header. Any
SPAM script accepts an alternative checkpoint as its first argument, for
example `juice500/wavlm-12-phonemodel`.

Recorded results and job ids for all runs are listed in
[`SPAM_experiment_log.md`](../SPAM_experiment_log.md).

## Montreal Forced Aligner setup

MFA is installed with conda, following the
[MFA documentation](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html).
The toplines expect two micromamba environments:

- `mfa310`: the latest MFA 3 plus the project dependencies, because the run
  scripts execute `src/main.py` inside it.

  ```bash
  micromamba create -n mfa310 -c conda-forge python=3.10 montreal-forced-aligner
  uv pip install --python "$(micromamba run -n mfa310 python -c 'import sys; print(sys.executable)')" \
      -r pyproject.toml --extra x86
  ```

- `mfa2`: MFA 2, used only for SSNCE. Its Tamil acoustic model `tamil_cv` was
  built with MFA 2 and does not align correctly under MFA 3.

  ```bash
  micromamba create -n mfa2 -c conda-forge python=3.10 "montreal-forced-aligner=2.*"
  ```

Acoustic models and dictionaries are downloaded on first use into
`MFA_ROOT_DIR`. If micromamba is not on `PATH` or its root is not
`~/micromamba`, set `MAMBA_BIN` and `MAMBA_ROOT_PREFIX`.

## Table I: phone segmentation (R-value, 20 ms tolerance)

| System | Command |
|---|---|
| SPAM | `sbatch scripts/run_spam_segment.sh` |
| CTC, FCE, BCE | see [Supervised baselines](#supervised-baselines) |
| Ground-truth transcript + MFA | `sbatch scripts/run_mfa_topline.sh` |
| KoelLabs-XLSR + MFA | `sbatch scripts/run_koel_mfa_baseline.sh` |
| PhoneticXEUS + MFA | `sbatch scripts/run_pxeus_recognize.sh <dataset>`, then `sbatch scripts/run_mfa_align_dump.sh <dataset> <dump glob>`. For SSNCE: `sbatch scripts/run_mfa2_tamil.sh` |

## Table II: phone recognition (PFER on PRiSM)

| System | Command |
|---|---|
| SPAM | `sbatch scripts/run_spam_recognition.sh` |
| CTC, FCE | see [Supervised baselines](#supervised-baselines) |

Two further scripts support the recognition analysis in the text.
`run_spam_recognition_novocab.sh` runs SPAM without per-language vocabulary
routing. `run_phonvec_oracle.sh` evaluates recognition on ground-truth
boundaries; `scripts/eval_seen_unseen.py` then splits its output into phones
seen and unseen during training.

## Table III and figures

| Result | Command |
|---|---|
| Table III: boundary-signal ablation | `sbatch scripts/run_arch_ablation.sh` |
| Figure 3: training-data efficiency | `sbatch scripts/run_efficiency_segmentation_ablation.sh` and `sbatch scripts/run_efficiency_recognition_ablation.sh` |
| Figure 4: encoder and layer ablation | `sbatch scripts/run_sslwlayer_ablation.sh` |

## Supervised baselines

The baselines fine-tune WavLM-large on TIMIT with one of three heads. Train
first, then pass the selected checkpoint to the corresponding evaluation script.

| Head | Training | Evaluation |
|---|---|---|
| BCE (boundary classifier) | `run_bce_train.sh` | `run_bce_segment.sh <ckpt>` |
| FCE (frame-wise cross-entropy) | `run_fce_train.sh` | `run_fce_segment.sh <ckpt>`, `run_wavlm_fce_recognition.sh <ckpt>` |
| CTC, checkpoint selected by R-value (Table I) | `run_wavlm_ctc_segment_train.sh` | `run_wavlm_ctc_segment.sh <ckpt>` |
| CTC, checkpoint selected by PER (Table II) | `run_wavlm_ctc_recognition_train.sh` | `run_wavlm_ctc_recognition.sh <ckpt>` |

Training scripts accept Hydra overrides, for example
`sbatch scripts/run_bce_train.sh trainer.max_steps=30000`.
