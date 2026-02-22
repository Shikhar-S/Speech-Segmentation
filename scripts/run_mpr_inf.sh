# PROB=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Runs for RQ1: Masked PR inference
# BUCKEYE
# for mp in ${PROB[@]}; do 
#     scripts/run.sh \
#         --setup inference \
#         --model ctag,lv60,xlsr53,powsm \
#         --extra_args "data=buckeye data.mask_probability=$mp"
#     sleep 1s
# done

# TIMIT
# PROB=(0.2)
PROB=(0.0 0.7)
for mp in ${PROB[@]}; do 
    echo "Running for mask probability: $mp"
    sbatch --array=0-7 -p preempt --gres=gpu:1 -c 13 -t 48:00:00 --mem=40G \
        scripts/babel.batch \
        experiment=inference/timit_pr_powsm_ctc \
        data.mask_probability=$mp \
        run_folder=masked_pr_${mp} inference.num_workers=5
    sleep 1s
done

# # ZIPACTC , buckeye
# for mp in ${PROB[@]}; do 
#     scripts/run.sh \
#         --setup inference \
#         --model zipactc,zipactc_ns \
#         --extra_args "data=buckeye data.mask_probability=$mp"
#     sleep 1s
# done


##########

# scripts/run.sh \
#     --setup inference \
#     --model lv60 \
#     --extra_args "data=cmul2arcticl1"