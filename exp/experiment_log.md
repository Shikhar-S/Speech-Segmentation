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

**Checkpoint selection:** `val/loss` (min). Inference used best-loss checkpoint
(`step_000562.ckpt`) for both models.

### Round 2 — resolution=2 (10ms frames)

| Run | Model | Config | Run Dir | SLURM |
|-----|-------|--------|---------|-------|
| r2-xeus | XEUS | `experiment=train/segmentation_xeus_bce model.resolution=2` | `exp/runs/seg_bce_xeus_r2/20260323_233545/` | 16947275 |
| r2-pxeus | pXEUS | `experiment=train/segmentation_pxeus_bce model.resolution=2` | `exp/runs/seg_bce_pxeus_r2/20260323_233545/` | 16947276 |

**Hyperparameters:** same as r1 except resolution=2 (Linear upsample
D→D*2, reshape to double temporal frames, effective_pbf=160 = 10ms)

**Checkpoint selection:** `val/loss` (min). Inference used `last.ckpt` (not
best-loss checkpoint). XEUS best was `step_000562`, pXEUS best was `step_000593`.

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

**Checkpoint selection:** `val/rval` (max). Inference used best-rval checkpoint
(`step_002458` for XEUS, `step_001479` for pXEUS).

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

## Comparability Notes

Checkpoint selection is **not consistent** across rounds:

| Round | Ckpt monitor | Ckpt used for inference |
|-------|-------------|------------------------|
| r1 | `val/loss` (min) | best val/loss (`step_000562`) |
| r2 | `val/loss` (min) | `last.ckpt` (not best) |
| 32k | `val/rval` (max) | best val/rval (`step_002458`/`step_001479`) |

### Wandb curve analysis: does checkpoint selection matter?

We pulled val/loss and val/rval curves from wandb to quantify the impact.
val/loss overfits early (best at step ~31) while val/rval keeps improving
and plateaus later.

| Run | Best-loss step | Rval @ best-loss | Best-rval step | Rval @ best-rval | Gap |
|-----|---------------|------------------|----------------|------------------|-----|
| r1 XEUS | 31 | 0.960 | 53 | 0.978 | +0.018 |
| r1 pXEUS | 31 | 0.973 | 157 | 0.978 | +0.006 |
| r2 XEUS | 31 | 0.811 | 196 | 0.910 | +0.099 |
| r2 pXEUS | 32 | 0.822 | 187 | 0.909 | +0.086 |

**Conclusion: checkpoint selection is NOT a major confound.**

- **r1:** Gap is tiny (0.6–1.8% rval). The best-loss checkpoint already has
  near-optimal rval. Rerunning with val/rval won't meaningfully change results.
- **r2:** Gap looks large (8–10%), but r2 inference used `last.ckpt` (rval
  ~0.90), which is within 0.5% of the best-rval checkpoint (0.91). The real
  bottleneck is the approach (learned upsample), not checkpoint selection.
- **32k** already uses val/rval — no issue.
- The 32kHz gains over r1 (4–16% rval) far exceed any checkpoint effect.

Reruns with val/rval (jobs 16996720–16996723) will provide definitive
confirmation but are not expected to change conclusions.

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

---

## Round 4 — WavLM-large + segment_recognize recipe (16kHz)

Switch from XEUS/pXEUS to **WavLM-large** as the encoder, paired with the
composable `segment_recognize` recipe (`heads/`: BCEBoundaryHead, CountCTCHead).
All Round 4 runs are 16 kHz, max_steps=20000, batch=8 on 4× GH200, lr=1e-4,
checkpoint monitor `val_seg/rval` (max), early-stop patience=100. Inference
goes through `build_segment_recognize_inference`; `tied_weights=true` is set
for any run with the `TieBCECountCTCProj` callback.

### Round 4a — Single-loss baselines

| Run | Loss | Train data | Run dir |
|---|---|---|---|
| timit_wavlm_bce | BCE | TIMIT | `exp/runs/timit_wavlm_bce/20260505_095300/` |
| timit_wavlm_count_ctc | CountCTC | TIMIT | `exp/runs/timit_wavlm_count_ctc/20260506_021729/` (v2; v1 at `20260505_095631_beststop_on_rval/`) |
| buckeye_wavlm_bce | BCE | Buckeye | `exp/runs/buckeye_wavlm_bce/20260509_035946/` |
| buckeye_wavlm_count_ctc | CountCTC | Buckeye | `exp/runs/buckeye_wavlm_count_ctc/20260509_035946/` |

### Round 4b — Tied joint configs (BCE+CountCTC sharing one projection)

`TieBCECountCTCProj` callback rewires `bce.boundary_head` to read class-1−class-0
log-odds out of `count_ctc.proj` (`Linear(D,2)`); BCE consumes one corpus,
CountCTC consumes another, both heads share the projection.

| Run | BCE on | CC on | Run dir |
|---|---|---|---|
| buckeye_timit_wavlm_count_ctc_bce | TIMIT | Buckeye | `exp/runs/buckeye_timit_wavlm_count_ctc_bce/20260509_040018/` |
| timit_buckeye_wavlm_count_ctc_bce | Buckeye | TIMIT | `exp/runs/timit_buckeye_wavlm_count_ctc_bce/20260509_042107/` |

### Round 4c — Combined-corpus runs (TIMIT+Buckeye for both losses)

| Run | Loss | Train data | Run dir |
|---|---|---|---|
| timit_buckeye_wavlm_bce | BCE | TIMIT+Buckeye | `exp/runs/timit_buckeye_wavlm_bce/20260509_051351/` |
| timit_buckeye_wavlm_count_ctc | CountCTC | TIMIT+Buckeye | `exp/runs/timit_buckeye_wavlm_count_ctc/20260509_051529/` |

### Round 4 — Eval on TIMIT/Buckeye test, lenient @20ms (full table in `tmp/eval_results_tol20ms_lenient.csv`; strict in `…_strict.csv`)

| Train | Decoding head | Eval | F1 | Precision | Recall | R-value |
|---|---|---|---|---|---|---|
| TIMIT | BCE | TIMIT | 0.850 | 0.864 | 0.837 | 0.870 |
| TIMIT | BCE | Buckeye | 0.786 | 0.783 | 0.788 | 0.817 |
| TIMIT | CC (up_switch) | TIMIT | 0.639 | 0.634 | 0.644 | 0.690 |
| TIMIT | CC (up_switch) | Buckeye | 0.645 | 0.630 | 0.661 | 0.692 |
| Buckeye | BCE | TIMIT | 0.774 | 0.810 | 0.742 | 0.803 |
| Buckeye | BCE | Buckeye | 0.817 | 0.822 | 0.812 | 0.844 |
| Buckeye | CC (up_switch) | TIMIT | 0.674 | 0.695 | 0.655 | 0.723 |
| Buckeye | CC (up_switch) | Buckeye | 0.650 | 0.643 | 0.657 | 0.699 |
| Tied: BCE-Tm + CC-Bk | BCE | TIMIT | 0.854 | 0.868 | 0.840 | 0.873 |
| Tied: BCE-Tm + CC-Bk | CC (up_switch) | TIMIT | 0.802 | 0.853 | 0.757 | 0.820 |
| Tied: BCE-Tm + CC-Bk | BCE | Buckeye | 0.764 | 0.665 | 0.898 | 0.657 |
| Tied: BCE-Tm + CC-Bk | CC (up_switch) | Buckeye | 0.625 | 0.615 | 0.636 | 0.676 |
| Tied: BCE-Bk + CC-Tm | BCE | TIMIT | 0.712 | 0.622 | 0.833 | 0.633 |
| Tied: BCE-Bk + CC-Tm | CC (up_switch) | TIMIT | 0.589 | 0.576 | 0.603 | 0.643 |
| Tied: BCE-Bk + CC-Tm | BCE | Buckeye | 0.822 | 0.831 | 0.814 | 0.848 |
| Tied: BCE-Bk + CC-Tm | CC (up_switch) | Buckeye | 0.727 | 0.796 | 0.668 | 0.755 |
| TIMIT+Buckeye | BCE | TIMIT | 0.850 | 0.857 | 0.843 | 0.871 |
| TIMIT+Buckeye | BCE | Buckeye | 0.818 | 0.813 | 0.824 | 0.845 |
| TIMIT+Buckeye | CC (up_switch) | TIMIT | 0.686 | 0.686 | 0.686 | 0.732 |
| TIMIT+Buckeye | CC (up_switch) | Buckeye | 0.679 | 0.678 | 0.679 | 0.726 |

### Round 4d — Decoding aggregation modes for CountCTC head

CTC's width-invariance means CC emits *plateaus* of class-1 logits at boundary
locations. Three aggregation rules tested on the two **tied** ckpts (where BCE
and CC share a projection so per-frame masks are identical — the only thing
that varies is which frames inside a plateau emit a boundary event):

| Mode | Definition | Lenient F1 vs up_switch | Strict F1 vs up_switch |
|---|---|---|---|
| `up_switch` (default) | One event at the leftmost frame of each plateau | baseline | baseline |
| `mid_switch` | One event at the plateau midpoint | +0.005 to +0.030 | +0.005 to +0.024 |
| `any_switch` | One event per rising AND falling edge | +0.080 to +0.115 | −0.110 to −0.130 |

Full table in `tmp/eval_boundary_modes_tol20_{lenient,strict}.csv`.

**Findings:**
- `any_switch` is a textbook lenient-mode loophole: emits 2× events per plateau
  → recall ~0.92 lenient, but strict matching (greedy 1-to-1 align) collapses
  it (R-value drops to 0.16 on TIMIT, 0.42 on Buckeye).
- `mid_switch` closes the BCE-vs-CC gap on tied checkpoints from ~5 F1 down to
  ~1 F1 in-domain — the right default *for tied models*.
- `mid_switch` HURTS pure-CC checkpoints (Round 4e) because their plateaus are
  narrow; offset detection just adds noise. Pure CC stays on `up_switch`.

### Round 4e — Pure CountCTC-on-TIMIT generalization across 10 datasets

Two TIMIT-only CC checkpoints inferenced on TIMIT, Buckeye, VoxAngeles, Torgo,
ssnce, and the four GlobalTIMIT splits + Thai. Outputs at
`exp/runs/cc_timit_mid_inference/{v1,v2}_{ds}/preds.0.jsonl`.

`up_switch` (the default) eval at lenient @20ms, v2 only (full file
`tmp/eval_cc_timit_v2_upswitch.csv`):

| Dataset | F1 | P | R | R-value |
|---|---|---|---|---|
| TIMIT | 0.639 | 0.634 | 0.644 | 0.690 |
| Buckeye | 0.645 | 0.630 | 0.661 | 0.692 |
| VoxAngeles | 0.245 | 0.231 | 0.260 | 0.351 |
| Torgo | 0.487 | 0.481 | 0.494 | 0.553 |
| ssnce | 0.503 | 0.503 | 0.504 | 0.567 |
| gtimit-l1simple | 0.629 | 0.640 | 0.620 | 0.689 |
| gtimit-l1tbnk | 0.637 | 0.659 | 0.617 | 0.690 |
| gtimit-l2simple | 0.487 | 0.504 | 0.471 | 0.561 |
| gtimit-l2tbnk | 0.524 | 0.541 | 0.508 | 0.591 |
| gtimit-tha | 0.495 | 0.529 | 0.464 | 0.572 |

`mid_switch` eval at lenient @20ms (full file `tmp/eval_cc_timit_mid_10ds.csv`):
v2 mid_switch loses 6–19 F1 vs up_switch on every dataset (e.g. TIMIT
0.528 vs 0.639; gtimit-l1tbnk 0.589 vs 0.637). v1 (older
`20260505_095631_beststop_on_rval` ckpt) is uniformly 9–21 F1 worse than v2 across
all 10 datasets — first round v1 was ever evaluated.

**Findings:**
- **mid_switch is a tied-only optimization.** Pure CC plateaus are narrow;
  midpoint emission adds rather than removes noise. Default pure-CC stays
  on `up_switch`.
- **v2 strictly dominates v1** by 9–21 F1 across all 10 datasets — all prior
  published CC-on-TIMIT inferences used v2/last.ckpt.
- **CC-on-TIMIT WavLM generalizes to English variants but degrades on
  unrelated phonologies**: best on TIMIT and English GlobalTIMIT splits
  (~0.63), drops to 0.49–0.50 on Torgo/ssnce, and collapses on VoxAngeles
  (0.25 — multilingual phonology too far from TIMIT).

### Round 4f — From-scratch (random-init) WavLM ablation

Mirror of all 8 Round 4a–c configs but `+model.net.pretrained=false` so the
encoder is randomly initialized while keeping the same `microsoft/wavlm-large`
architecture (24 layers, hidden 1024, conv stride 320). Implementation: 1
new arg threaded through `src/model/wavlm/{wavlm_model,builders}.py` (no
config edits — Hydra `+` prefix). Submitted via `tmp/submit_scratch_runs.sh`,
SLURM jobs 2266829–2266836, all completed cleanly (~2-3h each).

Run dirs: `exp/runs/<task>_scratch/20260510_07*/checkpoints/last.ckpt` for all
8 configs. Inference jobs 2267258–2267273 + 2267284–2267285 (2 interconnect
resubmits). Eval CSVs: `tmp/eval_results_scratch_tol20ms_{lenient,strict}.csv`.

**Pretrained-vs-scratch ΔF1 at lenient @20ms** (negative = scratch worse):

| Train | Head | Eval | Pretrained F1 | Scratch F1 | ΔF1 |
|---|---|---|---|---|---|
| TIMIT | BCE | TIMIT | 0.850 | 0.741 | −0.109 |
| TIMIT | BCE | Buckeye | 0.786 | 0.516 | −0.270 |
| TIMIT | CC | TIMIT | 0.639 | 0.286 | −0.353 |
| TIMIT | CC | Buckeye | 0.645 | 0.199 | −0.446 |
| Buckeye | BCE | TIMIT | 0.774 | 0.631 | −0.143 |
| Buckeye | BCE | Buckeye | 0.817 | 0.711 | −0.106 |
| Buckeye | CC | TIMIT | 0.674 | 0.360 | −0.314 |
| Buckeye | CC | Buckeye | 0.650 | 0.535 | −0.115 |
| Tied BCE-Tm+CC-Bk | BCE | TIMIT | 0.854 | 0.601 | −0.253 |
| Tied BCE-Tm+CC-Bk | CC | TIMIT | 0.802 | 0.106 | −0.696 |
| Tied BCE-Tm+CC-Bk | BCE | Buckeye | 0.764 | 0.590 | −0.174 |
| Tied BCE-Tm+CC-Bk | CC | Buckeye | 0.625 | 0.030 | −0.595 |
| Tied BCE-Bk+CC-Tm | BCE | TIMIT | 0.712 | 0.390 | −0.322 |
| Tied BCE-Bk+CC-Tm | CC | TIMIT | 0.589 | 0.128 | −0.461 |
| Tied BCE-Bk+CC-Tm | BCE | Buckeye | 0.822 | 0.225 | −0.597 |
| Tied BCE-Bk+CC-Tm | CC | Buckeye | 0.727 | 0.037 | −0.690 |
| TIMIT+Buckeye | BCE | TIMIT | 0.850 | 0.768 | −0.082 |
| TIMIT+Buckeye | BCE | Buckeye | 0.818 | 0.725 | −0.093 |
| TIMIT+Buckeye | CC | TIMIT | 0.686 | 0.117 | −0.569 |
| TIMIT+Buckeye | CC | Buckeye | 0.679 | 0.030 | −0.649 |

## Round 4 — Key findings

1. **Best single-loss generalizer to OOD (Buckeye): BCE-on-TIMIT alone**
   (F1=0.786 lenient, R-val=0.817). Beats both tied configs evaluated with
   BCE-decode and tied with CC-decode. **Adding CC supervision on Buckeye to
   a BCE-on-TIMIT base actively HURTS** (R-val crashes to 0.657 lenient /
   0.346 strict — the BCE head over-fires R=0.90+).

2. **BCE-decode > CC-decode on tied checkpoints is purely an aggregation
   artifact.** Same per-frame mask, but BCE emits one event per True frame
   (multi-shot per plateau) while CC `up_switch` emits one event per
   plateau. Empirically, tied BCE / CC `up_switch` event-count ratios:
   1.20× on TIMIT, 1.32× on Buckeye; >85% utterances have ≥1 wide plateau.

3. **`any_switch` decoding gameplays lenient matching.** +0.08–0.12 F1
   lenient over `up_switch`, but strict matching reveals a 0.10–0.15 F1
   penalty (R-value drops by 0.3–0.5). Never publish lenient-only `any_switch`
   numbers.

4. **`mid_switch` is the right default for tied checkpoints** (closes the
   BCE-vs-CC aggregation gap from ~5 F1 to ~1 F1 in-domain), but **HURTS
   pure-CC checkpoints by 6–19 F1**. Default pure-CC stays on `up_switch`.

5. **Combining corpora is a stronger lever than encoder pretraining for
   BCE.** TIMIT+Buckeye BCE *scratch* (F1=0.768/0.725) matches
   single-corpus BCE *pretrained* on cross-domain — more dense boundary
   supervision overcomes losing HF init. CC supervision can never bootstrap
   a random encoder (scratch CC F1 ≤ 0.36 across all settings).

6. **Encoder-pretraining contribution scales with objective sparsity.**
   ΔF1 (pretrained − scratch) ranking, lenient @20ms:
   - Combined-corpus BCE: 0.08–0.10
   - Single-corpus BCE: 0.10–0.27
   - CC alone: 0.11–0.45
   - Tied BCE-decode: 0.17–0.60
   - Tied CC-decode: 0.46–0.70

   The harder it is to backprop boundary information (sparse signal,
   competing losses, shared projection), the more random encoders need a
   head start.

7. **Tied scratch joint runs degenerate.** BCE head over-fires
   (R=0.97–0.99, R-value goes negative on strict), CC head silences
   (P=0.99, R=0.06). Without pretrained features, two losses through a
   shared projection collapse into one head emitting everywhere and the
   other emitting nothing.

8. **WavLM-large CC-on-TIMIT generalizes well to English variants
   (TIMIT+gTIMIT-l1: F1≈0.63), degrades on dysarthric/Indian English
   (Torgo/ssnce: ~0.50), and collapses on multilingual phonology
   (VoxAngeles: 0.25).** Dataset-relative phonology distance is the dominant
   covariate.

9. **Pure-CC v2 strictly dominates v1** on every dataset by 9–21 F1. Every
   prior published CC-on-TIMIT inference used v2/last.ckpt; v1 was never
   evaluated until Round 4e.
