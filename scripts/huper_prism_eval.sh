extra_args=()
DS=(gmuaccent timit l2arctic_perceived voxangeles doreco tusom2021) # prism intrinsic
DS=(epadb buckeye speechoceannotth) # ood
DS=(aishell cv fleurs fleurs_indv kazakh librispeech mls_dutch mls_french mls_german mls_italian mls_polish mls_portuguese mls_spanish southengland tamil) # indomain

# # decode
# for ds in ${DS[@]}; do
#     echo "Transcribing with Huper, the dataset: $ds"
#     # skip if the jsonl already exists
#     if [ -f "data/powsmeval/decodedv3.${ds}/huper/predictions.jsonl" ]; then
#         echo "Predictions already exist for dataset: $ds, skipping transcription."
#         continue
#     fi
#     sbatch -p ghx4 --time=1:00:00 scripts/daixpr_inference.batch \
#         experiment=inference/transcribe_huper \
#         data=powsmeval data.dataset_name=$ds \
#         task_name=decodedv3.${ds} \
#         inference.num_workers=4 \
#         run_folder=huper
# done

# evaluate
for ds in ${DS[@]}; do
    echo "Evaluating Huper on dataset: $ds"
    python scripts/jsonl2json.py --dirname exp/runs/decodedv3.${ds}/huper
    echo "merged entries, now evaluating metrics..."
    python -m src.metrics.phone_recognition \
        --prediction_file exp/runs/decodedv3.${ds}/huper/transcription.json \
        --output_file exp/runs/ipapack_ctc/paperresults-huper.csv \
        --gt_field target \
        --evaluation_name huper-${ds} \
        --key_field utt_id &
done
wait
echo "=========================="
cut -d',' -f1-11 exp/runs/ipapack_ctc/paperresults-huper.csv
echo "=========================="


# # evaluate against epitran
# for ds in ${DS[@]}; do
#     echo "Evaluating Huper on dataset: $ds"
#     python scripts/jsonl2json.py --dirname exp/runs/decodedv3.${ds}/huper
#     echo "merged entries, now evaluating metrics..."
#     python -m src.metrics.phone_recognition \
#         --prediction_file exp/runs/decodedv3.${ds}/huper/transcription.json \
#         --output_file exp/runs/ipapack_ctc/paperresults-huper.csv \
#         --gt_file exp/data/epitran_outputs/${ds}.epitran \
#         --evaluation_name huper-${ds}_epitran \
#         --key_field utt_id &
# done
# wait
# echo "=========================="
# cut -d',' -f1-11 exp/runs/ipapack_ctc/paperresults-huper.csv
# echo "=========================="