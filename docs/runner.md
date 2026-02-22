# Combined Experiment Runner

## Quick Start

```bash
# Run all models on LID task
./run.sh --recipe lif

# Run specific models with full fine-tuning
./run.sh --model powsm,ctag --recipe lif --fft

# Use different cluster
./run.sh --model all --recipe fab,fat --cluster delta

# Inference setup
./run.sh --model ctag --recipe lif --setup inference

# Dry run
./run.sh --model powsm --recipe lif --dry_run true
```

## Recipe Codes

| Code | Task |
|------|------|
| `fab` | fa_buckeye |
| `fat` | fa_timit |
| `gsw` | geolocation_swissgerman |
| `gva` | geolocation_vaani |
| `l1c` | l1cls_cmul2arctic |
| `l2a` | l2as_speechocean |
| `lif` | lid_fleurs |

## Models

`logmel`, `powsm`, `powsmvr`, `ctag`, `lv60`, `xlsr53`, `zipactc`, `zipactc_ns`

## Options

```bash
--model LIST        Comma-separated or "all" (default: all)
--recipe LIST       Recipe codes or "all" (default: all)
--cluster NAME      dai or delta (default: dai)
--setup NAME        probing or inference (default: probing)
--fft               Enable full fine-tuning
--parallel          Run in parallel
--dry_run           Print commands only
--run_name STR      Custom run name
--sbatch_args STR   Extra sbatch arguments
--extra_args STR    Extra training arguments
```

## How It Works

### Recipe Mapping
Short codes → full task_dataset names
```bash
lif → lid_fleurs
fab → fa_buckeye
```

### Cluster Mapping
Cluster name → batch script
```bash
dai → scripts/dai.batch
delta → scripts/delta.batch
```

### Model Config
Model variant → base_model|hf_repo
```bash
ctag → w2v2ph|ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns
powsm → powsm|
```

### Setup Mode
- `probing`: `model.net.hf_repo=<repo>`
- `inference`: `inference.inference_runner.hf_repo=<repo>`

### Config Path
`configs/experiment/{setup}/{task_dataset}_{base_model}.yaml`

## Command Example

```bash
./run.sh --model ctag --recipe lif --cluster delta --setup probing --fft
```

Generates:
```bash
sbatch scripts/delta.batch experiment=probing/lid_fleurs_w2v2ph \
  model.net.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns \
  model.freeze_encoder=false \
  tags+=["fft"] \
  data.batch_size=24
```

## Examples

```bash
# Basic
./run.sh --model powsm --recipe lif
./run.sh --model ctag,lv60,xlsr53 --recipe lif
./run.sh --model powsm --recipe fab,fat,lif

# Cluster
./run.sh --model zipactc --cluster delta
./run.sh --model ctag --cluster dai --sbatch_args "--partition=gpu-a100"

# Setup
./run.sh --model ctag --recipe lif --setup inference

# Full fine-tuning
./run.sh --model all --recipe lif --fft

# Parallel
./run.sh --model powsm,ctag,zipactc --parallel
```

## Output

```
exp/multirun/<run_name>/summary.log
```

Example:
```
=== Run: 20241209143022 ===
Started: Mon Dec  9 14:30:22 UTC 2024
Setup: probing | Cluster: dai | FFT: false

RUN: powsm × lif
CMD: sbatch scripts/dai.batch experiment=probing/lid_fleurs_powsm
SUCCESS: powsm × lif

SKIP: logmel × lif

=== Summary ===
Total: 8 | Success: 6 | Failed: 1 | Skipped: 1
```

## Extending

```bash
# Add model
model_configs["mynew"]="mybase|myorg/my-checkpoint"

# Add recipe
recipe_configs["mrc"]="mytask_mydataset"

# Add cluster
cluster_configs["newcluster"]="scripts/newcluster.batch"
```