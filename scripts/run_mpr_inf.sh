# Runs for RQ1: Masked PR inference
# BUCKEYE
for mp in 0.0 0.2 0.4 0.6 0.8; do sbatch scripts/inference.sh data.mask_probability=$mp; done

for mp in 0.0 0.2 0.4 0.6 0.8; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/buckeye_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft; done

for mp in 0.0 0.2 0.4 0.6 0.8; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/buckeye_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft; done

for mp in 0.0 0.2 0.4 0.6 0.8; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/buckeye_pr_w2v2ph inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns; done


# TIMIT
for mp in 0.0 0.2 0.4 0.6 0.8; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/timit_pr_powsm; done

for mp in 0.0 0.2 0.4 0.6 0.8; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/timit_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft; done

for mp in 0.0 0.2 0.4 0.6 0.8; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/timit_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft; done

for mp in 0.0 0.2 0.4 0.6 0.8; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/timit_pr_w2v2ph inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns; done