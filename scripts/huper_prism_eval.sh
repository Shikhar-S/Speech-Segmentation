# DS=(speechoceannotth l2arctic_perceived epadb gmuaccent buckeye voxangeles)
DS=(speechoceannotth epadb gmuaccent buckeye)

# # decode
# for ds in ${DS[@]}; do
#     echo "Transcribing with Huper, the dataset: $ds"
#     # Transcribe! -p gpuA40x4 scripts/deltaxpr.batch
#     sbatch -p ghx4-interactive --time=2:00:00 scripts/daixpr_inference.batch \
#         experiment=inference/transcribe_huper \
#         data=powsmeval data.dataset_name=$ds \
#         task_name=decodedv3.${ds} \
#         inference.num_workers=4 \
#         run_folder=huper
# done

# # evaluate
# for ds in ${DS[@]}; do
#     echo "Evaluating Huper on dataset: $ds"
#     python scripts/jsonl2json.py --dirname exp/runs/decodedv3.${ds}/huper
#     echo "merged entries, now evaluating metrics..."
#     python -m src.metrics.phone_recognition \
#         --prediction_file exp/runs/decodedv3.${ds}/huper/transcription.json \
#         --output_file exp/runs/ipapack_ctc/results-huper.csv \
#         --gt_field target \
#         --evaluation_name huper-${ds} \
#         --key_field utt_id &
# done
# wait
# echo "=========================="
# cut -d',' -f1-11 exp/runs/ipapack_ctc/results-huper.csv
# echo "=========================="


# evaluate against epitran
for ds in ${DS[@]}; do
    echo "Evaluating Huper on dataset: $ds"
    python scripts/jsonl2json.py --dirname exp/runs/decodedv3.${ds}/huper
    echo "merged entries, now evaluating metrics..."
    python -m src.metrics.phone_recognition \
        --prediction_file exp/runs/decodedv3.${ds}/huper/transcription.json \
        --output_file exp/runs/ipapack_ctc/results-huper.csv \
        --gt_file exp/data/epitran_outputs/${ds}.epitran \
        --evaluation_name huper-${ds}_epitran \
        --key_field utt_id &
done
wait
echo "=========================="
cut -d',' -f1-11 exp/runs/ipapack_ctc/results-huper.csv
echo "=========================="