DS=(speechoceannotth epadb gmuaccent buckeye)
# DS=(voxangeles  l2arctic_perceived)

# # decode
# for ds in ${DS[@]}; do
#     echo "Transcribing with KoelLabs XLSR, the dataset: $ds"
#     # Transcribe! -p gpuA40x4 scripts/deltaxpr.batch
#     sbatch -p ghx4-interactive --time=2:00:00 scripts/daixpr_inference.batch \
#         experiment=inference/transcribe_koel \
#         data=powsmeval data.dataset_name=$ds \
#         task_name=decodedv3.${ds} \
#         inference.num_workers=4 \
#         run_folder=koel
# done

# # evaluate
# for ds in ${DS[@]}; do
#     echo "Evaluating KoelLabs XLSR on dataset: $ds"
#     python scripts/jsonl2json.py --dirname exp/runs/decodedv3.${ds}/koel
#     echo "merged entries, now evaluating metrics..."
#     python -m src.metrics.phone_recognition \
#         --prediction_file exp/runs/decodedv3.${ds}/koel/transcription.json \
#         --output_file exp/runs/ipapack_ctc/results-koel.csv \
#         --gt_field target \
#         --evaluation_name koel-${ds} \
#         --key_field utt_id &
# done
# wait
# echo "=========================="
# cut -d',' -f1-11 exp/runs/ipapack_ctc/results-koel.csv
# echo "=========================="


# evaluate against epitran
for ds in ${DS[@]}; do
    echo "Evaluating KoelLabs XLSR on dataset: $ds"
    python scripts/jsonl2json.py --dirname exp/runs/decodedv3.${ds}/koel
    echo "merged entries, now evaluating metrics..."
    python -m src.metrics.phone_recognition \
        --prediction_file exp/runs/decodedv3.${ds}/koel/transcription.json \
        --output_file exp/runs/ipapack_ctc/results-koel.csv \
        --gt_file exp/data/epitran_outputs/${ds}.epitran \
        --evaluation_name koel-${ds}_epitran \
        --key_field utt_id &
done
wait
echo "=========================="
cut -d',' -f1-11 exp/runs/ipapack_ctc/results-koel.csv
echo "=========================="
