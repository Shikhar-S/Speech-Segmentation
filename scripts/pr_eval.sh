ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt')

# DS=(gmuaccent buckeye epadb speechoceanotth l2arctic)
DS=(gmuaccent)

for ckpt in ${ckpts[@]}; do
   train_run_folder=$(basename "${ckpt%/checkpoints/*}")
   stepnum=$(basename $ckpt | sed 's/checkpoint-\(.*\).ckpt/\1/')
   for ds in ${DS[@]}; do
      echo "Evaluating checkpoint: $ckpt on dataset: $ds"
      python scripts/jsonl2json.py --dirname exp/runs/decode.${ds}/${train_run_folder}.ck${stepnum}
      python -m src.metrics.phone_recognition \
          --prediction_file exp/runs/decode.${ds}/${train_run_folder}.ck${stepnum}/transcription.json \
          --output_file exp/runs/ipapack_ctc/results-${train_run_folder}.csv \
          --gt_field target \
          --evaluation_name ${train_run_folder}-${ds}-${stepnum} \
          --key_field utt_id &
   done
   wait
   echo "=========================="
   cut -d',' -f1-6 exp/runs/ipapack_ctc/results-${train_run_folder}.csv
   echo "=========================="
done
