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

## Evaluation Results (20ms tolerance)

| Dataset | Model | Res | F1 | Precision | Recall | R-value |
|---------|-------|-----|------|-----------|--------|---------|
| TIMIT | XEUS | r1 | 0.736 | 0.928 | 0.610 | 0.723 |
| TIMIT | XEUS | r2 | 0.600 | 0.981 | 0.432 | 0.599 |
| TIMIT | pXEUS | r1 | **0.860** | 0.908 | **0.818** | **0.867** |
| TIMIT | pXEUS | r2 | 0.753 | 0.982 | 0.610 | 0.724 |
| Buckeye | XEUS | r1 | 0.621 | 0.809 | 0.504 | 0.647 |
| Buckeye | XEUS | r2 | 0.269 | 0.874 | 0.159 | 0.405 |
| Buckeye | pXEUS | r1 | **0.767** | 0.790 | **0.744** | **0.799** |
| Buckeye | pXEUS | r2 | 0.641 | 0.904 | 0.497 | 0.644 |
| VoxAngeles | XEUS | r1 | 0.367 | 0.391 | 0.346 | 0.478 |
| VoxAngeles | XEUS | r2 | 0.190 | 0.266 | 0.148 | 0.375 |
| VoxAngeles | pXEUS | r1 | **0.438** | 0.436 | **0.441** | **0.519** |
| VoxAngeles | pXEUS | r2 | 0.381 | 0.439 | 0.337 | 0.497 |

## Key Findings

1. **pXEUS consistently outperforms XEUS** across all datasets and resolutions.
2. **resolution=1 (20ms) outperforms resolution=2 (10ms)** everywhere.
   Resolution=2 increases precision but severely hurts recall, lowering F1/R-value.
3. **Best overall: pXEUS r1** — TIMIT R-val=0.867, Buckeye R-val=0.799.

## Code Changes (branch: phoneticxeus)

1. `5e7ebb0f1` — BoundaryLoss in segmentation_loss.py
2. `503a2bf18` — BCE integration + multi-mode test eval in model_module.py
3. `0a255dc84` — Integration tests (test_model_module.py, test_segmentation_loss.py)
4. `b878576d9` — BCE train/inference configs
5. `eaea80bbc` — SLURM scripts (scripts/bce_loss/)
6. `0b467023c` — Fix boundary metrics (evaluator-based, not frame-level)
7. `f77630d24` — Temporal upsampling (resolution parameter)
8. `78ce08716` — Checkpoint monitors val/rval (max)
