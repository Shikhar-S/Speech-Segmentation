______________________________________________________________________

<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

A codebase for building phone segmentation and recognition models.

## 🚀 Quickstart

```bash
# clone project
git clone git@github.com:Shikhar-S/Speech-Segmentation.git
cd Speech-Segmentation

# install (auto-detects x86_64 for Delta/Babel, aarch64 for Delta-AI)
make install

# activate environment (once per session)
source .venv/bin/activate
```

On Delta-AI, force the `dai` variant explicitly:
```bash
make install-dai
source .venv/bin/activate
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/main.py trainer=cpu

# train on GPU
python src/main.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
# For phone recognition training
python src/main.py experiment=train/timit_powsmpr

# For inference experiments
python src/main.py experiment=inference/timit_powsmpr
```

You can override any parameter from command line like this

```bash
python src/main.py trainer.max_epochs=20 data.batch_size=64
```

## Phone Recognition Pipeline

```bash
bash scripts/pr_timit_train.sh      # train on TIMIT with Epitran mixing
bash scripts/pr_decode.sh           # decode/inference on datasets
bash scripts/pr_eval.sh             # evaluate decoded outputs
python scripts/pr_parse_results.py  # aggregate results
python scripts/parse_wandb.py       # extract WandB metrics
```

## More Documentation

- **[Features & Capabilities](docs/features.md)** - Look at this to train on multi-gpu, run hyper-param searches etc.
- **[Running Inference](docs/running_inference.md)** - Guide for running phone recognition inference with pre-trained models
- **[Tokenization Workflow](docs/tokenization.md)** - How to build vocabularies and use tokenizers for IPA transcripts
- **[Contributing Guide](CONTRIBUTING.md)** - Project structure, workflow, and best practices for contributors

## ❤️ Acknowledgement

This repository structure is based on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).
