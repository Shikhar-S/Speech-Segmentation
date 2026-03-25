# BCE Segmentation Experiment Log

## Overview

BCE boundary detection for phone segmentation using XEUS and PhoneticXEUS
backbones, trained on TIMIT, evaluated on TIMIT, Buckeye, and VoxAngeles.

## Training Runs

### Round 1 — resolution=1 (20ms frames)

| Run | Model | Config | Run Dir | SLURM |
|-----|-------|--------|---------|-------|
| r1-xeus | XEUS | `experiment=train/segmentation_xeus_bce` | `exp/runs/seg_bce_xeus/20260323_194645/` | 16943945 |
| r1-pxeus | pXEUS | `experiment=train/segmentation_pxeus_bce` | `exp/runs/seg_bce_pxeus/20260323_194652/` | 16943946 |

**Hyperparameters:** bce_weight=1.0, pos_weight=1.0, resolution=1, lr=1e-4,
batch_size=32, max_steps=5000, val_check_interval=125, data=timit-segment

### Round 2 — resolution=2 (10ms frames)

| Run | Model | Config | Run Dir | SLURM |
|-----|-------|--------|---------|-------|
| r2-xeus | XEUS | `experiment=train/segmentation_xeus_bce model.resolution=2` | `exp/runs/seg_bce_xeus_r2/20260323_233545/` | 16947275 |
| r2-pxeus | pXEUS | `experiment=train/segmentation_pxeus_bce model.resolution=2` | `exp/runs/seg_bce_pxeus_r2/20260323_233545/` | 16947276 |

**Hyperparameters:** same as r1 except resolution=2 (Linear upsample
D→D*2, reshape to double temporal frames, effective_pbf=160 = 10ms)

## Inference Runs

All inference uses `build_boundary_inference` with threshold=0.5.
Config: `experiment=inference/segmentation_{xeus,pxeus}_bce`

### Round 1

| Model | Dataset | Output JSONL | Lines |
|-------|---------|-------------|-------|
| XEUS | TIMIT | `exp/runs/inf_bce_xeus_timit/inf_bce_xeus_timit/seg_xeus_bce.0.jsonl` | 1680 |
| XEUS | Buckeye | `exp/runs/inf_bce_xeus_buckeye/inf_bce_xeus_buckeye/seg_xeus_bce.0.jsonl` | 10477 |
| XEUS | VoxAngeles | `exp/runs/inf_bce_xeus_voxangeles/inf_bce_xeus_voxangeles/seg_xeus_bce.0.jsonl` | 5445 |
| pXEUS | TIMIT | `exp/runs/inf_bce_pxeus_timit/inf_bce_pxeus_timit/seg_pxeus_bce.0.jsonl` | 1680 |
| pXEUS | Buckeye | `exp/runs/inf_bce_pxeus_buckeye/inf_bce_pxeus_buckeye/seg_pxeus_bce.0.jsonl` | 10477 |
| pXEUS | VoxAngeles | `exp/runs/inf_bce_pxeus_voxangeles/inf_bce_pxeus_voxangeles/seg_pxeus_bce.0.jsonl` | 5445 |

### Round 2

| Model | Dataset | Output JSONL | Lines |
|-------|---------|-------------|-------|
| XEUS | TIMIT | `exp/runs/inf_bce_xeus_r2_timit/inf_bce_xeus_r2_timit/seg_xeus_bce.0.jsonl` | 1680 |
| XEUS | Buckeye | `exp/runs/inf_bce_xeus_r2_buckeye/inf_bce_xeus_r2_buckeye/seg_xeus_bce.0.jsonl` | 10477 |
| XEUS | VoxAngeles | `exp/runs/inf_bce_xeus_r2_voxangeles/inf_bce_xeus_r2_voxangeles/seg_xeus_bce.0.jsonl` | 5445 |
| pXEUS | TIMIT | `exp/runs/inf_bce_pxeus_r2_timit/inf_bce_pxeus_r2_timit/seg_pxeus_bce.0.jsonl` | 1680 |
| pXEUS | Buckeye | `exp/runs/inf_bce_pxeus_r2_buckeye/inf_bce_pxeus_r2_buckeye/seg_pxeus_bce.0.jsonl` | 10477 |
| pXEUS | VoxAngeles | `exp/runs/inf_bce_pxeus_r2_voxangeles/inf_bce_pxeus_r2_voxangeles/seg_pxeus_bce.0.jsonl` | 5445 |

### Round 3 — 32kHz input upsampling (10ms frames via CNN stride)

| Run | Model | Config | Run Dir | SLURM | Steps |
|-----|-------|--------|---------|-------|-------|
| 32k-xeus | XEUS | `experiment=train/segmentation_xeus_bce_32k` | `exp/runs/seg_bce_xeus_32k/20260324_165829/` | 16983986 | 2458/5000 (timeout) |
| 32k-pxeus | pXEUS | `experiment=train/segmentation_pxeus_bce_32k` | `exp/runs/seg_bce_pxeus_32k/20260324_165829/` | 16983987 | 1479/5000 (early stop) |

**Hyperparameters:** bce_weight=1.0, resolution=1, audio_sr=32000, target_sr=32000,
lr=1e-4, batch_size=8, accumulate_grad_batches=16, max_speech_length=8s,
max_steps=5000, 1×A40 GPU. Effective batch=128.

**Note:** XEUS training timed out at 8h (step 2458). pXEUS converged via early
stopping (patience=100) at step 1479. Results may improve with full XEUS training.

### Round 3 Inference

| Model | Dataset | Output JSONL | Lines |
|-------|---------|-------------|-------|
| XEUS | TIMIT | `exp/runs/inf_bce_xeus_32k_timit/inf_bce_xeus_32k_timit/seg_xeus_bce_32k.0.jsonl` | 6300 |
| XEUS | Buckeye | `exp/runs/inf_bce_xeus_32k_buckeye/inf_bce_xeus_32k_buckeye_v2/seg_xeus_bce_32k.0.jsonl` | 10477 |
| XEUS | VoxAngeles | `exp/runs/inf_bce_xeus_32k_voxangeles/inf_bce_xeus_32k_voxangeles_v2/seg_xeus_bce_32k.0.jsonl` | 5445 |
| pXEUS | TIMIT | `exp/runs/inf_bce_pxeus_32k_timit/inf_bce_pxeus_32k_timit/seg_pxeus_bce_32k.0.jsonl` | 6300 |
| pXEUS | Buckeye | `exp/runs/inf_bce_pxeus_32k_buckeye/inf_bce_pxeus_32k_buckeye/seg_pxeus_bce_32k.0.jsonl` | 10477 |
| pXEUS | VoxAngeles | `exp/runs/inf_bce_pxeus_32k_voxangeles/inf_bce_pxeus_32k_voxangeles/seg_pxeus_bce_32k.0.jsonl` | 5445 |

## Evaluation Results (20ms tolerance)

| Dataset | Model | Round | F1 | Precision | Recall | R-value |
|---------|-------|-------|------|-----------|--------|---------|
| TIMIT | XEUS | r1 | 0.736 | 0.928 | 0.610 | 0.723 |
| TIMIT | XEUS | r2 | 0.600 | 0.981 | 0.432 | 0.599 |
| TIMIT | XEUS | 32k | 0.890 | 0.924 | 0.858 | 0.895 |
| TIMIT | pXEUS | r1 | 0.860 | 0.908 | 0.818 | 0.867 |
| TIMIT | pXEUS | r2 | 0.753 | 0.982 | 0.610 | 0.724 |
| TIMIT | pXEUS | **32k** | **0.906** | 0.938 | **0.876** | **0.910** |
| Buckeye | XEUS | r1 | 0.621 | 0.809 | 0.504 | 0.647 |
| Buckeye | XEUS | r2 | 0.269 | 0.874 | 0.159 | 0.405 |
| Buckeye | XEUS | 32k | 0.777 | 0.787 | 0.767 | 0.809 |
| Buckeye | pXEUS | r1 | 0.767 | 0.790 | 0.744 | 0.799 |
| Buckeye | pXEUS | r2 | 0.641 | 0.904 | 0.497 | 0.644 |
| Buckeye | pXEUS | **32k** | **0.781** | 0.801 | **0.762** | **0.812** |
| VoxAngeles | XEUS | r1 | 0.367 | 0.391 | 0.346 | 0.478 |
| VoxAngeles | XEUS | r2 | 0.190 | 0.266 | 0.148 | 0.375 |
| VoxAngeles | XEUS | 32k | 0.516 | 0.457 | 0.592 | 0.500 |
| VoxAngeles | pXEUS | r1 | 0.438 | 0.436 | 0.441 | 0.519 |
| VoxAngeles | pXEUS | r2 | 0.381 | 0.439 | 0.337 | 0.497 |
| VoxAngeles | pXEUS | **32k** | **0.528** | 0.473 | **0.597** | **0.524** |

## Key Findings

1. **32kHz input upsampling is the clear winner** — beats both r1 and r2 on every
   dataset/model combination, often by large margins.
2. **Recall is the key driver** — 32kHz dramatically improves recall while
   maintaining or slightly reducing precision, resulting in much higher F1/R-value.
3. **XEUS now competitive with pXEUS** — at 32kHz the gap narrows significantly
   (e.g. Buckeye R-val: XEUS 0.809 vs pXEUS 0.812).
4. **Best overall: pXEUS 32k** — TIMIT R-val=0.910, Buckeye R-val=0.812.
5. **resolution=2 (learned upsample) underperforms** — 32kHz input upsampling
   achieves the same 10ms frame rate but via the CNN frontend, which works much
   better than the learned Linear upsample layer.
6. **Partial training** — XEUS reached only step 2458/5000 (timeout). Results
   may improve further with full training.

## Code Changes (branch: phoneticxeus)

1. `5e7ebb0f1` — BoundaryLoss in segmentation_loss.py
2. `503a2bf18` — BCE integration + multi-mode test eval in model_module.py
3. `0a255dc84` — Integration tests (test_model_module.py, test_segmentation_loss.py)
4. `b878576d9` — BCE train/inference configs
5. `eaea80bbc` — SLURM scripts (scripts/bce_loss/)
6. `0b467023c` — Fix boundary metrics (evaluator-based, not frame-level)
7. `f77630d24` — Temporal upsampling (resolution parameter)
8. `78ce08716` — Checkpoint monitors val/rval (max)
9. `bdf0cb048` — audio_sr param for correct timestamps at non-16kHz sample rates
10. `75a0d16bf` — 32kHz experiment configs and SLURM scripts
11. `8e615bb1b` — CTC-based rval metrics for FA mode, fix 32kHz OOM configs
