<div align="center">

<a href="https://arxiv.org/abs/2607.09020"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2607.09020-b31b1b?logo=arxiv&logoColor=white"></a>
<a href="https://github.com/juice500ml/phonespam"><img alt="phonespam" src="https://img.shields.io/badge/model-phonespam-blue"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

</div>

# Speech-Segmentation

Evaluation code for **Phone Segmentation and Recognition through Phonological
Activation Mapping** (SLT 2026). The repository reproduces every table and
figure in the paper: the SPAM segmenter and recognizer, the supervised WavLM
baselines (CTC, FCE, BCE), and the Montreal Forced Aligner toplines. All systems
are scored with [`phone-metrics`](https://github.com/stephenmac7/phone-metrics).

The SPAM model is developed in [`phonespam`](https://github.com/juice500ml/phonespam).
This repository wraps it in `src/model/phonvec/` and evaluates the released
checkpoint `juice500/wavlm-24-phonemodel`.

## Installation

```bash
git clone git@github.com:Shikhar-S/Speech-Segmentation.git
cd Speech-Segmentation
make install          # full install; detects x86_64 vs aarch64
source scripts/env.sh # activates .venv and puts the repo root on PYTHONPATH
```

The full install includes ESPnet, which only the WavLM baselines and the
PhoneticXEUS topline need. `bash setup.sh core` installs everything else
(SPAM, MFA and Koel toplines, scoring, data preparation) without it.

A GPU is required. Each `scripts/run_*.sh` is a self-contained SLURM job.
Submit it from the repository root and override the account and partition for
your cluster:

```bash
sbatch -A <account> -p <partition> scripts/run_spam_segment.sh
```

## Configuration

No path in the repository is specific to a cluster. Site-specific settings are
supplied through environment variables, which `sbatch` forwards to the job.
Every variable has a default, so set only those used by the scripts you run.
The one exception is `XEUSPR_CKPT`, which has no default and is read only by
the PhoneticXEUS topline.

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | token saved by `huggingface-cli login` | access to private Hugging Face datasets |
| `SEG_REPO_<DATASET>` | `changelinglab/<dataset>-segment` | Hugging Face repository of each segmentation dataset (see [Data preparation](#data-preparation)) |
| `HF_HOME` | `exp/cache/hf` | Hugging Face cache directory |
| `PROJECT_ROOT` | directory from which `sbatch` was run | repository root, when submitting from elsewhere |
| `PRISM_DUMP_DIR` | `exp/data/prism` | Kaldi-style directories of the PRiSM phone-recognition test sets (Table II) |
| `MFA_ROOT_DIR`, `MAMBA_ROOT_PREFIX` | `exp/cache/mfa`, `~/micromamba` | Montreal Forced Aligner toplines; requires an MFA installation (see [docs/reproduction.md](docs/reproduction.md)) |
| `XEUSPR_CKPT` | none (required) | checkpoint of the PhoneticXEUS phone recognizer, used by one topline |

## Data preparation

The segmentation datasets are built from licensed corpora that cannot be
redistributed. The scripts in `scripts/data_prep/` convert the original
releases into Hugging Face datasets and upload them to a private repository of
your choosing; the `SEG_REPO_<DATASET>` variables then point the code at those
repositories. Sources, commands, and the correspondence with the paper's splits
are described in [docs/data_preparation.md](docs/data_preparation.md).

## Reproducing the paper

Each result maps to one SLURM script that runs inference and scoring and writes
`summary.csv` under `exp/runs/`.

| Result | Script |
|---|---|
| Table I, SPAM | `run_spam_segment.sh` |
| Table I, MFA toplines | `run_mfa_topline.sh`, `run_koel_mfa_baseline.sh`, `run_pxeus_recognize.sh` + `run_mfa_align_dump.sh` |
| Table II, SPAM | `run_spam_recognition.sh` |
| Tables I and II, WavLM baselines | `run_{bce,fce,wavlm_ctc_*}_train.sh`, then the matching `*_segment.sh` / `*_recognition.sh` |
| Table III | `run_arch_ablation.sh` |
| Figure 3 | `run_efficiency_{segmentation,recognition}_ablation.sh` |
| Figure 4 | `run_sslwlayer_ablation.sh` |

Per-system commands, checkpoint selection, and the analysis scripts are in
[docs/reproduction.md](docs/reproduction.md). Recorded results and job ids are
in [SPAM_experiment_log.md](SPAM_experiment_log.md); the code layout is in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@inproceedings{spam2026,
  title     = {Phone Segmentation and Recognition through Phonological Activation Mapping},
  author    = {Bharadwaj, Shikhar and Choi, Kwanghee and McIntosh, Stephen and Li, Chin-Jou and
               Yeo, Eunjung and Saito, Daisuke and Minematsu, Nobuaki and Watanabe, Shinji and
               Zhu, Jian and Harwath, David and Mortensen, David R.},
  booktitle = {Proc. IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026}
}
```

## Acknowledgement

The repository structure is based on the
[Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).
