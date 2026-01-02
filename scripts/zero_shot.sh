DATASETS=("cmul2arcticl1|l1cls_cmul2arctic" \
    "vaanigeo|geolocation_vaani" \
    "edacc|l1cls_edacc" \
    "speechocean|l2as_speechocean" \
    "fleurs|lid_fleurs" \
    "easycall|atypical_easycall" \
    "uaspeech|atypical_uaspeech" \
    "ultrasuite_child|atypical_ultrasuite")

# DATASETS=("vaanigeo|geolocation_vaani")


for DP in "${DATASETS[@]}"; do 
    D=$(echo ${DP} | cut -d'|' -f1)
    P=$(echo ${DP} | cut -d'|' -f2)
    echo "Submitting job for dataset: ${D} with prompt: ${P}"
    # -p ghx4 --reservation sup-20848-4 --nodelist=gh046,gh137
    sbatch -p ghx4-interactive --array=0-9 \
    -t 30:00 scripts/vllm_dai.batch \
    experiment=inference/transcribe_qweninstruct.yaml \
    data=${D} prompt=${P} \
    task_name=infzs_${D}_qweni inference.num_workers=50 run_folder=1jobArr
done
