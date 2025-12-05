PROB=(0.1 0.3 0.5 0.7 0.9)
PROB=(0.0 0.2 0.4 0.6 0.8)

# Runs for RQ1: Masked PR inference
# # BUCKEYE
# for mp in ${PROB[@]}; do sbatch scripts/inference.sh data.mask_probability=$mp; sleep 1s; done

# for mp in ${PROB[@]}; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/buckeye_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft; sleep 1s; done

# for mp in ${PROB[@]}; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/buckeye_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft; sleep 1s; done

# for mp in ${PROB[@]}; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/buckeye_pr_w2v2ph inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns; sleep 1s; done

# # TIMIT
# for mp in ${PROB[@]}; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/timit_pr_powsm; sleep 1s; done

# for mp in ${PROB[@]}; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/timit_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft; sleep 1s; done

# for mp in ${PROB[@]}; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/timit_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft; sleep 1s; done

# for mp in ${PROB[@]}; do sbatch scripts/inference.sh data.mask_probability=$mp experiment=inference/timit_pr_w2v2ph inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns; sleep 1s; done


# ZIPACTC
# for mp in ${PROB[@]}; do sbatch scripts/inference_delta.sh data.mask_probability=$mp experiment=inference/timit_pr_zipactc inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-500k; sleep 1s; done

# for mp in ${PROB[@]}; do sbatch scripts/inference_delta.sh data.mask_probability=$mp experiment=inference/timit_pr_zipactc inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-ns-800k; sleep 1s; done

# for mp in ${PROB[@]}; do sbatch scripts/inference_delta.sh data.mask_probability=$mp experiment=inference/buckeye_pr_zipactc inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-500k; sleep 1s; done

# for mp in ${PROB[@]}; do sbatch scripts/inference_delta.sh data.mask_probability=$mp experiment=inference/buckeye_pr_zipactc inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-ns-800k; sleep 1s; done
