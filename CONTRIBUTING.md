# Contributing

## Repository layout

```
├── configs/
│   ├── experiment/
│   │   ├── segment/       <- segmentation inference (phonvec, wavlm_*, mfa_*)
│   │   ├── recognize/     <- recognition inference (phonvec*, wavlm_*, xeuspr*)
│   │   └── train/         <- WavLM baseline training
│   ├── data/              <- datamodules
│   └── inference/         <- vocab JSONs, MFA masks, PRiSM IPA maps
├── scripts/
│   ├── run_*.sh           <- one SLURM job per paper result: inference + scoring
│   ├── eval_*.py          <- scoring (wraps phone-metrics)
│   └── data_prep/         <- licensed corpus -> HF dataset -> your Hub repo
├── src/
│   ├── core/              <- entrypoint + distributed inference (ask before editing)
│   ├── data/              <- datamodules and per-dataset label transforms
│   ├── model/
│   │   ├── phonvec/       <- SPAM wrapper (the paper's method)
│   │   ├── wavlm/         <- baseline backbone
│   │   └── mfa/ koel/ xeusphoneme/ powsm/   <- toplines and their recognizers
│   ├── recipe/            <- baseline heads (CTC / FCE / BCE)
│   └── metrics/           <- SegmentationUnit + phone-metrics adapters
└── tests/
```

## Execution flow

1. `sbatch scripts/run_x.sh` calls `python src/main.py experiment=…`.
2. Inference experiments shard the dataset over SLURM array tasks and GPUs and
   write `*.jsonl`. Training experiments run a Lightning `Trainer` instead.
3. The script scores the JSONL with `scripts/eval_segmentation.py` or
   `scripts/eval_recognition.py` and writes `summary.csv`.

Cluster paths and dataset repo ids come from environment variables only (see
the README). In shell: `${SEG_REPO_TIMIT:-changelinglab/timit-segment}`. In
YAML: `${oc.env:SEG_REPO_TIMIT,changelinglab/timit-segment}`.

## Extending the benchmark

**Adding an evaluation dataset.** Write a prep script in `scripts/data_prep/`
that emits the segmentation schema (see the README), register its label
cleanup in `dataset_processing_transforms.py` under `"<name>-segment"`, add it
to the `REPOS` map of the segmentation run scripts, and add a README row.
Datasets built from licensed corpora stay private on the Hub.

**Comparing a new segmenter or recognizer against SPAM.** Write a builder that
returns a callable mapping a batch to per-utterance predictions, following
`build_phonvec_inference` in `src/model/phonvec/inference.py`. Point
`inference.inference_runner._target_` at it in a new experiment config and
copy the closest `run_*.sh` (1 GPU, 16 CPUs, 32 GB, time limit) so it is scored
with the same `eval_*.py` as the paper's numbers.

**Changing SPAM itself.** The model lives upstream in
[`phonespam`](https://github.com/juice500ml/phonespam); this repo only wraps
it. Fix the model there, then bump the pin in `pyproject.toml` and re-run the
affected scripts.

## Checklist

- `grep -rnE '/work/|/home/|/scratch/' scripts configs src` returns nothing.
- Style follows `docs/style.md`. Scratch files live in `tmp/`.
- Any re-run of a paper result is recorded in `SPAM_experiment_log.md`
  (job ids, config, numbers) next to the previous ones.
- `pytest` passes (needs the project env).
