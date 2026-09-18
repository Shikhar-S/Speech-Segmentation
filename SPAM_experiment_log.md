# SPAM Experiment Log

Numbers and reproduction commands for the phonvec (`juice500/wavlm-24-phonemodel`,
layer 24), WavLM BCE/FCE/CTC, and MFA runs on branch `unified`. Segmentation
tables: boundary tolerance 20 ms, strict mode, outer silences stripped.

## Phonvec segmentation

**Reproduce:** `sbatch scripts/run_spam_segment.sh`

Job 19769858 (shipped silence config: `silence_threshold=0.4` +
`silence_gap_frames=5`). Outputs `exp/runs/phonvec_segmentation/19769858/<dataset>/`.

| dataset           | n_utts | precision | recall | F1     | R-val  | over_seg |
|-------------------|-------:|----------:|-------:|-------:|-------:|---------:|
| timit             | 1680   | 0.8669    | 0.7234 | 0.7887 | 0.7996 | -0.1655 |
| buckeye           | 1010   | 0.8083    | 0.6778 | 0.7373 | 0.7630 | -0.1615 |
| gtimit_l2simple   | 6000   | 0.7310    | 0.6892 | 0.7095 | 0.7523 | -0.0572 |
| gtimit_l2tbnk     | 6000   | 0.6800    | 0.5951 | 0.6347 | 0.6891 | -0.1249 |
| torgo             | 3946   | 0.7143    | 0.6405 | 0.6754 | 0.7224 | -0.1034 |
| ssnce             | 7860   | 0.5624    | 0.4917 | 0.5246 | 0.6029 | -0.1258 |
| voxangeles        | 5445   | 0.7331    | 0.6857 | 0.7086 | 0.7513 | -0.0647 |
| gtimit_tha        | 6000   | 0.7900    | 0.6746 | 0.7277 | 0.7583 | -0.1460 |

## WavLM BCE segmentation

**Train:** `sbatch scripts/run_bce_train.sh`
**Reproduce:** `sbatch scripts/run_bce_segment.sh exp/runs/timit_wavlm_bce/20260613_182302/last.ckpt`

Job 19752624. Outputs `exp/runs/bce_segmentation/19752624/<dataset>/`.

| dataset           | precision | recall | F1     | R-val  | over_seg |
|-------------------|----------:|-------:|-------:|-------:|---------:|
| timit             | 0.8221    | 0.8508 | 0.8362 | 0.8583 |  0.0350 |
| buckeye           | 0.7132    | 0.7810 | 0.7456 | 0.7696 |  0.0952 |
| gtimit_l2simple   | 0.6475    | 0.7122 | 0.6783 | 0.7106 |  0.0999 |
| gtimit_l2tbnk     | 0.5772    | 0.6138 | 0.5949 | 0.6454 |  0.0633 |
| torgo             | 0.6620    | 0.7733 | 0.7134 | 0.7193 |  0.1680 |
| ssnce             | 0.4551    | 0.4935 | 0.4735 | 0.5343 |  0.0845 |
| voxangeles        | 0.5998    | 0.6914 | 0.6424 | 0.6648 |  0.1527 |
| gtimit_tha        | 0.7152    | 0.7035 | 0.7093 | 0.7525 | -0.0163 |

## WavLM FCE segmentation

**Train:** `sbatch scripts/run_fce_train.sh`
**Reproduce:** `sbatch scripts/run_fce_segment.sh exp/runs/timit_wavlm_fce/20260627_004825/checkpoints/last.ckpt`

Job 19752171. Outputs `exp/runs/fce_segmentation/19752171/<dataset>/`.

| dataset           | precision | recall | F1     | R-val  | over_seg |
|-------------------|----------:|-------:|-------:|-------:|---------:|
| timit             | 0.8705    | 0.8855 | 0.8780 | 0.8955 |  0.0172 |
| buckeye           | 0.7553    | 0.8040 | 0.7789 | 0.8047 |  0.0645 |
| gtimit_l2simple   | 0.6685    | 0.7462 | 0.7052 | 0.7296 |  0.1163 |
| gtimit_l2tbnk     | 0.6437    | 0.6835 | 0.6630 | 0.7050 |  0.0618 |
| torgo             | 0.6265    | 0.7330 | 0.6756 | 0.6873 |  0.1700 |
| ssnce             | 0.4754    | 0.5626 | 0.5153 | 0.5433 |  0.1835 |
| voxangeles        | 0.5745    | 0.7021 | 0.6320 | 0.6304 |  0.2221 |
| gtimit_tha        | 0.6851    | 0.7210 | 0.7026 | 0.7409 |  0.0523 |

## WavLM CTC segmentation

**Reproduce:** `sbatch scripts/run_wavlm_ctc_segment.sh exp/runs/timit_wavlm_ctc_recognition/20260627_004824/checkpoints/last.ckpt`

Job 19752172 (boundaries from greedy CTC collapse + torchaudio forced align).
Outputs `exp/runs/wavlm_ctc_segmentation/19752172/<dataset>/`.

| dataset           | precision | recall | F1     | R-val   | over_seg |
|-------------------|----------:|-------:|-------:|--------:|---------:|
| timit             | 0.6565    | 0.6542 | 0.6554 |  0.7061 | -0.0035 |
| buckeye           | 0.6711    | 0.6644 | 0.6678 |  0.7170 | -0.0100 |
| gtimit_l2simple   | 0.5918    | 0.6173 | 0.6043 |  0.6569 |  0.0431 |
| gtimit_l2tbnk     | 0.5633    | 0.5599 | 0.5616 |  0.6264 | -0.0059 |
| torgo             | 0.6035    | 0.6125 | 0.6080 |  0.6638 |  0.0150 |
| ssnce             | 0.4474    | 0.4281 | 0.4375 |  0.5263 | -0.0430 |
| voxangeles        | 0.4056    | 0.4095 | 0.4075 |  0.4925 |  0.0095 |
| gtimit_tha        | 0.6068    | 0.5749 | 0.5904 |  0.6541 | -0.0525 |

## MFA topline (text-dependent, ground-truth phones)

**Reproduce:** `sbatch scripts/run_mfa_topline.sh`

| dataset           | job      | n_utts | precision | recall | F1    | R-val | over_seg |
|-------------------|----------|-------:|----------:|-------:|------:|------:|---------:|
| timit             | 19189522 | 1680   | 0.790     | 0.799  | 0.794 | 0.824 | 0.011 |
| buckeye           | 19189523 | 1010   | 0.804     | 0.825  | 0.814 | 0.840 | 0.025 |
| gtimit_l2simple   | 19189524 | 6000   | 0.766     | 0.806  | 0.786 | 0.812 | 0.053 |
| gtimit_l2tbnk     | 19189525 | 6000   | 0.804     | 0.821  | 0.812 | 0.839 | 0.022 |
| torgo             | 19189526 | 3946   | 0.752     | 0.804  | 0.777 | 0.803 | 0.069 |
| ssnce             | 19729890 | 7860   | 0.276     | 0.298  | 0.287 | 0.371 | 0.079 |
| gtimit_tha        | 19731930 | 6000   | 0.608     | 0.717  | 0.658 | 0.669 | 0.179 |

## MFA baseline (text-independent, Koel recognizer + MFA)

**Reproduce:** `sbatch scripts/run_koel_mfa_baseline.sh`

| dataset           | job      | n_utts | precision | recall | F1    | R-val | over_seg |
|-------------------|----------|-------:|----------:|-------:|------:|------:|---------:|
| timit             | 19189551 | 1680   | 0.737     | 0.794  | 0.764 | 0.790 | 0.077 |
| buckeye           | 19189552 | 1010   | 0.789     | 0.820  | 0.804 | 0.830 | 0.040 |
| gtimit_l2simple   | 19189553 | 6000   | 0.718     | 0.788  | 0.751 | 0.773 | 0.098 |
| gtimit_l2tbnk     | 19189554 | 6000   | 0.768     | 0.790  | 0.779 | 0.810 | 0.029 |
| torgo             | 19189555 | 3946   | 0.707     | 0.768  | 0.736 | 0.764 | 0.086 |

## PhoneticXEUS + MFA (text-independent cascade)

Per-utterance PhoneticXEUS phone recognition → MFA forced alignment of the
predicted transcript (baseline analog of Koel+MFA), via two staged scripts.

**Reproduce:**
```bash
# Stage 1 — PhoneticXEUS masked recognition dump (GPU)
sbatch scripts/run_pxeus_recognize.sh <dataset>
# Stage 2 — MFA forced-align the dump + eval (CPU)
sbatch scripts/run_mfa_align_dump.sh <dataset> \
  'exp/runs/pxeus_recognize/<jobid>/<dataset>/xeuspr_phones.*.jsonl'
```

Batch 19735400–19735404, outputs `exp/runs/mfa_xeuspr/<job>/<dataset>/`.

| dataset           | job      | n_utts | precision | recall | F1    | R-val | over_seg |
|-------------------|----------|-------:|----------:|-------:|------:|------:|---------:|
| timit             | 19735400 | 1680   | 0.696     | 0.803  | 0.746 | 0.751 | 0.153 |
| buckeye           | 19735401 | 1010   | 0.712     | 0.770  | 0.740 | 0.768 | 0.082 |
| gtimit_l2simple   | 19735402 | 6000   | 0.672     | 0.790  | 0.726 | 0.727 | 0.175 |
| gtimit_l2tbnk     | 19735403 | 6000   | 0.719     | 0.776  | 0.747 | 0.774 | 0.080 |
| torgo             | 19735404 | 3946   | 0.647     | 0.748  | 0.693 | 0.707 | 0.156 |

## Segmentation R-value — all methods side by side

**Reproduce:** `python -m scripts.eval_segmentation <glob> --tolerance-ms 20 --mode strict --strip-outer-silences --bootstrap 1000` per method/dataset (globs are the per-method `Outputs:` dirs above).

Point estimate `r ± h` (`h` = half the 95% bootstrap CI, N=1000). Bold = best
learned method per dataset (MFA topline/baseline excluded). Phonvec column is the
shipped thr-0.4 + gap-fill-5 run (job 19769858).

| dataset         | Phonvec            | WavLM-BCE          | WavLM-FCE          | WavLM-CTC          | MFA topline      | MFA baseline    |
|-----------------|--------------------|--------------------|--------------------|--------------------|------------------|-----------------|
| timit           | 0.800 ± 0.003      | 0.858 ± 0.003      | **0.895 ± 0.002**  | 0.706 ± 0.004      | 0.824 ± 0.003    | 0.790 ± 0.003   |
| buckeye         | 0.763 ± 0.003      | 0.770 ± 0.003      | **0.805 ± 0.003**  | 0.717 ± 0.003      | 0.840 ± 0.003    | 0.830 ± 0.003   |
| gtimit_l2simple | **0.752 ± 0.002**  | 0.711 ± 0.002      | 0.730 ± 0.002      | 0.657 ± 0.002      | 0.812 ± 0.002    | 0.773 ± 0.002   |
| gtimit_l2tbnk   | 0.689 ± 0.001      | 0.645 ± 0.002      | **0.705 ± 0.002**  | 0.626 ± 0.002      | 0.839 ± 0.001    | 0.810 ± 0.001   |
| torgo           | **0.722 ± 0.004**  | 0.719 ± 0.008      | 0.687 ± 0.009      | 0.664 ± 0.005      | 0.801 ± 0.005    | 0.764 ± 0.005   |
| ssnce           | **0.603 ± 0.002**  | 0.534 ± 0.002      | 0.543 ± 0.003      | 0.526 ± 0.002      | 0.371 ± 0.002    | —               |
| voxangeles      | **0.751 ± 0.004**  | 0.665 ± 0.006      | 0.630 ± 0.007      | 0.492 ± 0.004      | —                | —               |
| gtimit_tha      | **0.758 ± 0.001**  | 0.752 ± 0.001      | 0.741 ± 0.001      | 0.654 ± 0.002      | 0.669 ± 0.002    | —               |

## Phonvec recognition

**Reproduce:**
```bash
for ds in timit l2arctic_perceived gmuaccent doreco voxangeles tusom2021; do
  PYTHONPATH=. python scripts/build_phonvec_vocab.py "$ds" "configs/inference/phonvec_vocab/${ds}.json"
done
sbatch scripts/run_spam_recognition.sh
```

Job 19770276 (TIMIT = test-1680 filter; prism postproc is a no-op for phonvec).
Outputs `exp/runs/phonvec_recognition/19770276/<dataset>/`.

| dataset            | n_utts | PER    | PFER   |
|--------------------|-------:|-------:|-------:|
| timit              | 1680   | 0.6185 | 0.2230 |
| l2arctic_perceived | 3599   | 0.6766 | 0.2153 |
| gmuaccent          | 1242   | 0.7396 | 0.2283 |
| doreco             | 18734  | 0.6160 | 0.2567 |
| voxangeles         | 5445   | 0.6447 | 0.2306 |
| tusom2021          | 2255   | 0.7127 | 0.3415 |

## WavLM CTC recognition

**Train:** `sbatch scripts/run_wavlm_ctc_recognition_train.sh`
**Reproduce:** `sbatch scripts/run_wavlm_ctc_recognition.sh exp/runs/timit_wavlm_ctc_recognition/20260627_004824/checkpoints/last.ckpt`

Job 19770278 (TIMIT = test-1680 filter + `seg2prism`; VoxAngeles `seg2voxprism`;
other 4 raw prism). Outputs `exp/runs/wavlm_ctc_recognition/19770278/<dataset>/`.

| dataset            | n_utts | PER    | PFER   |
|--------------------|-------:|-------:|-------:|
| timit              | 1680   | 0.1680 | 0.0715 |
| l2arctic_perceived | 3599   | 0.4332 | 0.1522 |
| gmuaccent          | 1242   | 0.4317 | 0.1170 |
| doreco             | 18734  | 0.7612 | 0.2081 |
| voxangeles         | 5445   | 0.7508 | 0.2239 |
| tusom2021          | 2255   | 0.7746 | 0.2269 |

## WavLM FCE recognition

**Train:** `sbatch scripts/run_fce_train.sh`
**Reproduce:** `sbatch scripts/run_wavlm_fce_recognition.sh exp/runs/timit_wavlm_fce/20260627_004825/checkpoints/last.ckpt`

Job 19770279 (TIMIT = test-1680 filter + `seg2prism`; VoxAngeles `seg2voxprism`;
other 4 raw prism). Outputs `exp/runs/wavlm_fce_recognition/19770279/<dataset>/`.

| dataset            | n_utts | PER    | PFER   |
|--------------------|-------:|-------:|-------:|
| timit              | 1680   | 0.1658 | 0.0871 |
| l2arctic_perceived | 3599   | 0.4174 | 0.1884 |
| gmuaccent          | 1242   | 0.4439 | 0.1575 |
| doreco             | 18734  | 0.8340 | 0.2639 |
| voxangeles         | 5445   | 0.9341 | 0.4018 |
| tusom2021          | 2255   | 0.9318 | 0.3586 |

## PER / PFER — recognizers side by side

**Reproduce:** values from the per-method recognition sections above (jobs
19808648 / 19770278 / 19770279). TIMIT = test-1680, all prism-postprocessed.

PER:

| dataset            | Phonvec | WavLM-CTC | WavLM-FCE |
|--------------------|--------:|----------:|----------:|
| timit              | 0.8257  | 0.1680    | 0.1658    |
| l2arctic_perceived | 0.8522  | 0.4332    | 0.4174    |
| gmuaccent          | 0.8518  | 0.4317    | 0.4439    |
| doreco             | 0.8777  | 0.7612    | 0.8340    |
| voxangeles         | 0.9017  | 0.7508    | 0.9341    |
| tusom2021          | 0.9451  | 0.7746    | 0.9318    |

PFER:

| dataset            | Phonvec | WavLM-CTC | WavLM-FCE |
|--------------------|--------:|----------:|----------:|
| timit              | 0.2286  | 0.0715    | 0.0871    |
| l2arctic_perceived | 0.2255  | 0.1522    | 0.1884    |
| gmuaccent          | 0.2336  | 0.1170    | 0.1575    |
| doreco             | 0.2729  | 0.2081    | 0.2639    |
| voxangeles         | 0.2434  | 0.2239    | 0.4018    |
| tusom2021          | 0.3499  | 0.2269    | 0.3586    |

## Ablations

### Per-language vocab routing — with vs. without

**Reproduce:** `sbatch scripts/run_spam_recognition_novocab.sh`

"vocab" = per-language routing (job 19770276); "none" = open-vocab (job
19770463). Both TIMIT = test-1680, prism-postprocessed.

| dataset    | n_utts | PER (vocab) | PER (none) | PFER (vocab) | PFER (none) |
|------------|-------:|------------:|-----------:|-------------:|------------:|
| timit      | 1680   | 0.6185      | 0.5711     | 0.2230       | 0.2179      |
| voxangeles | 5445   | 0.6447      | 0.7868     | 0.2306       | 0.2349      |

### Recognizer inventory — vocab vs. panphon-unrestricted

**Reproduce:** `sbatch scripts/run_spam_recognition.sh` (decodes `vocab_file=null
+panphon_unrestricted=true`)

"vocab" = per-language `panphon_featmap` routing (job 19770276); "panphon" =
single `Recognizer` over the full PanPhon inventory, no featmap (job 19808648).
Both TIMIT = test-1680, prism-postprocessed.

| dataset            | n_utts | PER (vocab) | PER (panphon) | PFER (vocab) | PFER (panphon) |
|--------------------|-------:|------------:|--------------:|-------------:|---------------:|
| timit              | 1680   | 0.6185      | 0.8257        | 0.2230       | 0.2286         |
| l2arctic_perceived | 3599   | 0.6766      | 0.8522        | 0.2153       | 0.2255         |
| gmuaccent          | 1242   | 0.7396      | 0.8518        | 0.2283       | 0.2336         |
| doreco             | 18734  | 0.6160      | 0.8777        | 0.2567       | 0.2729         |
| voxangeles         | 5445   | 0.6447      | 0.9017        | 0.2306       | 0.2434         |
| tusom2021          | 2255   | 0.7127      | 0.9451        | 0.3415       | 0.3499         |

### Data-efficiency — segmentation

**Reproduce:** `sbatch scripts/run_efficiency_segmentation_ablation.sh`

Job 19769864 (wavlm-24, thr 0.4 + gap-fill 5).
Outputs `exp/runs/efficiency_segmentation_ablation/19769864/<frac>/<dataset>/`.

| data fraction | timit F1 | vox F1 | timit Rval | vox Rval |
|---------------|---------:|-------:|-----------:|---------:|
| 1/1 (full L24)| 0.789    | 0.709  | 0.800      | 0.751    |
| 1/2           | 0.788    | 0.709  | 0.799      | 0.752    |
| 1/4           | 0.788    | 0.708  | 0.799      | 0.751    |
| 1/8           | 0.786    | 0.707  | 0.797      | 0.750    |
| 1/16          | 0.789    | 0.709  | 0.800      | 0.752    |
| 1/32          | 0.789    | 0.708  | 0.799      | 0.750    |
| 1/64          | 0.786    | 0.705  | 0.798      | 0.748    |
| 1/128         | 0.787    | 0.704  | 0.796      | 0.747    |
| 1/256         | 0.786    | 0.704  | 0.799      | 0.748    |
| 1/512         | 0.771    | 0.687  | 0.782      | 0.733    |
| 1/1024        | 0.773    | 0.690  | 0.787      | 0.736    |

### Data-efficiency — recognition (PER / PFER)

**Reproduce:** `sbatch scripts/run_efficiency_recognition_ablation.sh`

Job 19770277 (thr 0.4 + gap-fill 5; TIMIT = test-1680 + prism postproc).
Outputs `exp/runs/efficiency_recognition_ablation/19770277/<frac>/<dataset>/`.

| data fraction | timit PER | timit PFER | vox PER | vox PFER |
|---------------|----------:|-----------:|--------:|---------:|
| 1/1 (full)    | 0.618     | 0.2230     | 0.645   | 0.2306   |
| 1/2           | 0.621     | 0.2238     | 0.646   | 0.2316   |
| 1/4           | 0.617     | 0.2230     | 0.644   | 0.2309   |
| 1/8           | 0.627     | 0.2262     | 0.642   | 0.2322   |
| 1/16          | 0.623     | 0.2235     | 0.660   | 0.2374   |
| 1/32          | 0.605     | 0.2227     | 0.643   | 0.2311   |
| 1/64          | 0.603     | 0.2211     | 0.649   | 0.2330   |
| 1/128         | 0.630     | 0.2310     | 0.686   | 0.2385   |
| 1/256         | 0.603     | 0.2212     | 0.658   | 0.2310   |
| 1/512         | 0.627     | 0.2413     | 0.659   | 0.2517   |
| 1/1024        | 0.641     | 0.2327     | 0.752   | 0.2482   |

### Architecture / tap stack (wavlm-24)

**Reproduce:** `sbatch scripts/run_arch_ablation.sh`

Job 19769863 (thr 0.4 + gap-fill 5). Full (+mel) row = wavlm-24 (job 19769858).
Outputs `exp/runs/arch_ablation/19769863/<variant>/<dataset>/`.

| variant                 | timit F1 | vox F1 | timit Rval | vox Rval |
|-------------------------|---------:|-------:|-----------:|---------:|
| oneframedelta           | 0.757    | 0.640  | 0.792      | 0.661    |
| stackframedelta         | 0.756    | 0.680  | 0.776      | 0.727    |
| stackframe+bwddelta     | 0.764    | 0.686  | 0.787      | 0.733    |
| full (+mel) = wavlm-24  | 0.789    | 0.709  | 0.800      | 0.751    |

### SSL encoder × layer — segmentation

**Reproduce:** `sbatch scripts/run_sslwlayer_ablation.sh`

Job 19769862, over `juice500/<family>-<layer>-phonemodel` on TIMIT (1680) and
VoxAngeles (5445). Outputs `exp/runs/sslwlayer_ablation/19769862/`. Cells are
`F1|Rval`; `wavlm-24` is the full model.

| layer | xls-r timit | xls-r vox | w2v2 timit | w2v2 vox | hubert timit | hubert vox | wavlm timit | wavlm vox |
|-------|-------------|-----------|------------|----------|--------------|------------|-------------|-----------|
| 0     | 0.733\|0.731 | 0.622\|0.680 | 0.758\|0.754 | 0.624\|0.682 | 0.731\|0.730 | 0.609\|0.669 | 0.718\|0.718 | 0.597\|0.660 |
| 3     | 0.764\|0.761 | 0.641\|0.696 | 0.770\|0.768 | 0.652\|0.704 | 0.756\|0.757 | 0.214\|0.383 | 0.751\|0.748 | 0.627\|0.683 |
| 6     | 0.775\|0.775 | 0.665\|0.714 | 0.779\|0.780 | 0.652\|0.704 | 0.770\|0.771 | 0.071\|0.319 | 0.766\|0.765 | 0.673\|0.721 |
| 9     | 0.733\|0.749 | 0.614\|0.667 | 0.705\|0.734 | 0.560\|0.630 | 0.768\|0.773 | 0.050\|0.311 | 0.770\|0.775 | 0.627\|0.653 |
| 12    | 0.700\|0.729 | 0.538\|0.581 | 0.724\|0.746 | 0.565\|0.634 | 0.748\|0.763 | 0.048\|0.311 | 0.751\|0.768 | 0.598\|0.646 |
| 15    | 0.740\|0.758 | 0.615\|0.672 | 0.760\|0.770 | 0.580\|0.639 | 0.633\|0.685 | 0.039\|0.307 | 0.538\|0.607 | 0.289\|0.432 |
| 18    | 0.780\|0.780 | 0.660\|0.709 | 0.765\|0.770 | 0.605\|0.660 | 0.627\|0.683 | 0.029\|0.304 | 0.577\|0.643 | 0.330\|0.410 |
| 21    | 0.761\|0.762 | 0.664\|0.712 | 0.782\|0.781 | 0.635\|0.688 | 0.755\|0.782 | 0.101\|0.332 | 0.737\|0.769 | 0.554\|0.593 |
| 24    | 0.544\|0.601 | 0.387\|0.169 | 0.225\|0.394 | 0.072\|0.321 | 0.761\|0.780 | 0.614\|0.662 | 0.789\|0.800 | 0.709\|0.751 |

### Oracle vs predicted boundaries — phonvec recognition

**Reproduce:** `sbatch scripts/run_phonvec_oracle.sh juice500/wavlm-24-phonemodel panphon`

Job 19771738, panphon-unrestricted recognizer (single `Recognizer` over the full
PanPhon inventory, no featmap), over the `changelinglab/*-segment` test sets;
scored against GT `phones`. Outputs `exp/runs/phonvec_oracle/19771738/panphon/`.

| dataset    | n_utts | PER (oracle) | PER (pred) | PFER (oracle) | PFER (pred) | macro-lang PER (oracle) |
|------------|-------:|-------------:|-----------:|--------------:|------------:|------------------------:|
| timit      | 1680   | 0.7907       | 0.8181     | 0.1106        | 0.2340      | 0.7907                  |
| voxangeles | 5445   | 0.8180       | 0.9094     | 0.0842        | 0.2420      | 0.8259                  |

**Seen/unseen split** (Vox oracle, panphon-unrestricted, per-phone PFER %).

```
python scripts/eval_seen_unseen.py \
  "exp/runs/phonvec_oracle/19771738/panphon/voxangeles/oracle/phonvec_oracle*.jsonl"
```

| split  | n_phones | PFER |
|--------|---------:|-----:|
| seen   | 13610    | 6.40 |
| unseen | 9006     | 9.36 |
| gap    |          | 2.96 |
