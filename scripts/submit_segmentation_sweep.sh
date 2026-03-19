#!/bin/bash
# FRACS=(0.2 0.4 0.6 0.8 1.0)
FRACS=(0.053 0.05 0.106)
MODELS=(segmentation_xeus segmentation_pxeus)
TAGS=(xeuspr pxeus)

for i in "${!MODELS[@]}"; do
    for frac in "${FRACS[@]}"; do
        frac_tag=${frac//./_}
        task_name="seg_${TAGS[$i]}_frac${frac_tag}"
        sbatch -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
            scripts/daixpr.batch \
            experiment=train/${MODELS[$i]} \
            data.train_fraction=${frac} \
            task_name=${task_name}
    done
done
