#!/bin/bash
# Format: "frac|model|run_folder"

RUNS=()
RUNS+=(
    "0.0053|segmentation_xeus|seg_xeuspr_frac0_0053" # 0.53%
    # "0.053|segmentation_xeus|seg_xeuspr_frac0_053" # 5.3%
    # "0.106|segmentation_xeus|seg_xeuspr_frac0_106" # 10.6%
    # "0.2|segmentation_xeus|seg_xeuspr_frac0_2" # 20%
    # "0.4|segmentation_xeus|seg_xeuspr_frac0_4" # 40%
    # "0.6|segmentation_xeus|seg_xeuspr_frac0_6" # 60%
    # "0.8|segmentation_xeus|seg_xeuspr_frac0_8" # 80%
    # "1.0|segmentation_xeus|seg_xeuspr_frac1_0" # 100%
)

RUNS+=(
    "0.0053|segmentation_pxeus|seg_pxeus_frac0_0053" # 0.53%
)

for entry in "${RUNS[@]}"; do
    IFS='|' read -r frac model run_folder <<< "$entry"
    sbatch -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
        scripts/daixpr.batch \
        experiment=train/${model} \
        data.train_fraction=${frac} \
        run_folder=${run_folder}
done
