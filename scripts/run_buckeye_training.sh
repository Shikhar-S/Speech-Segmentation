#!/bin/bash
# Buckeye segmentation training runs
# hf_repo: changelinglab/buckeye-segment
# Format: "frac|config|run_folder"

RUNS=()
RUNS+=(
    "0.0008|segmentation_xeus|seg_buckeye_frac0_0008" # 0.08% (~1 min)
    "0.0076|segmentation_xeus|seg_buckeye_frac0_0076" # 0.76% (~10 min)
    "0.0153|segmentation_xeus|seg_buckeye_frac0_0153" # 1.53% (~20 min)
    "0.0306|segmentation_xeus|seg_buckeye_frac0_0306" # 3.06% (~40 min)
    "0.0612|segmentation_xeus|seg_buckeye_frac0_0612" # 6.12% (~80 min)
    "0.1223|segmentation_xeus|seg_buckeye_frac0_1223" # 12.23% (~160 min)
    "0.2446|segmentation_xeus|seg_buckeye_frac0_2446" # 24.46% (~320 min)
    "0.4893|segmentation_xeus|seg_buckeye_frac0_4893" # 48.93% (~640 min)
    "0.9786|segmentation_xeus|seg_buckeye_frac0_9786" # 97.86% (~1280 min)
    "1.0|segmentation_xeus|seg_buckeye_frac1_0" # 100% (full)
)

RUNS+=(
    "0.0008|segmentation_pxeus|seg_buckeye_pxeus_frac0_0008" # 0.08% (~1 min)
    "0.0076|segmentation_pxeus|seg_buckeye_pxeus_frac0_0076" # 0.76% (~10 min)
    "0.0153|segmentation_pxeus|seg_buckeye_pxeus_frac0_0153" # 1.53% (~20 min)
    "0.0306|segmentation_pxeus|seg_buckeye_pxeus_frac0_0306" # 3.06% (~40 min)
    "0.0612|segmentation_pxeus|seg_buckeye_pxeus_frac0_0612" # 6.12% (~80 min)
    "0.1223|segmentation_pxeus|seg_buckeye_pxeus_frac0_1223" # 12.23% (~160 min)
    "0.2446|segmentation_pxeus|seg_buckeye_pxeus_frac0_2446" # 24.46% (~320 min)
    "0.4893|segmentation_pxeus|seg_buckeye_pxeus_frac0_4893" # 48.93% (~640 min)
    "0.9786|segmentation_pxeus|seg_buckeye_pxeus_frac0_9786" # 97.86% (~1280 min)
    "1.0|segmentation_pxeus|seg_buckeye_pxeus_frac1_0" # 100% (full)
)

for entry in "${RUNS[@]}"; do
    IFS='|' read -r frac config run_folder <<< "$entry"
    # -p ghx4-interactive 
    sbatch -c 72 -t 5:00:00 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
        scripts/daixpr.batch \
        experiment=train/${config} \
        data.hf_repo=changelinglab/buckeye-segment \
        data.batch_size=24 \
        data.train_fraction=${frac} \
        run_folder=${run_folder} "$@"
done
