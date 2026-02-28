DS=(gmuaccent timit l2arctic_perceived voxangeles doreco tusom2021) # prism intrinsic
# DS=(epadb buckeye speechoceannotth) # ood
# DS=(aishell cv fleurs fleurs_indv kazakh librispeech mls_dutch mls_french mls_german mls_italian mls_polish mls_portuguese mls_spanish southengland tamil) # indomain
# DS=(timit)

# # ctc variants
# # vanilla
# # interctc
# ckpts=(
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.losssched_half30k_m12tomp5.panphonk8.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.interctc_l4_8_12.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt' \
# )

# # ssl variants
# # ebranchformer - none
# ckpts=(
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/ebranch12l_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-24000.ckpt'
# )

# # mms 300m, 1b - vanilla ctc
# '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/mms1b_multiaccent.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt' \
ckpts=(
   '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/mms_multiaccent.bs320.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-8000.ckpt' \
)

# xeus on same data as other ssl variants
ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-18000.ckpt')


for ckpt in ${ckpts[@]}; do
   train_run_folder=$(basename "${ckpt%/checkpoints/*}")
   stepnum=$(basename $ckpt | sed 's/checkpoint-\(.*\).ckpt/\1/')
   for ds in ${DS[@]}; do
      echo "Evaluating checkpoint: $ckpt on dataset: $ds"
      # if there are no ouptuts skip this evaluation and tell user
      if [ ! -f exp/runs/decodedv3.${ds}/${train_run_folder}.ck${stepnum}/*.jsonl ]; then
         echo "No transcription found for checkpoint ${ckpt} on dataset ${ds}. Skipping evaluation for this dataset."
         continue
      fi
      python scripts/jsonl2json.py --dirname exp/runs/decodedv3.${ds}/${train_run_folder}.ck${stepnum}
      echo "merged entries, now computing metrics..."
      python -m src.metrics.phone_recognition \
          --prediction_file exp/runs/decodedv3.${ds}/${train_run_folder}.ck${stepnum}/transcription.json \
          --output_file exp/runs/ipapack_ctc/paperresults-${train_run_folder}.csv \
          --gt_field target \
          --evaluation_name ${train_run_folder}-${ds}-${stepnum} \
          --key_field utt_id &
   done
   wait
   echo "=========================="
   cut -d',' -f1-11 exp/runs/ipapack_ctc/paperresults-${train_run_folder}.csv
   echo "=========================="
done



# FOR COMPARISON WITH EPITRAN

# EPITRAN_BASE="exp/data/epitran_outputs"
# for ckpt in ${ckpts[@]}; do
#    train_run_folder=$(basename "${ckpt%/checkpoints/*}")
#    stepnum=$(basename $ckpt | sed 's/checkpoint-\(.*\).ckpt/\1/')
#    for ds in ${DS[@]}; do
#       echo "Evaluating checkpoint: $ckpt on dataset: $ds with epitran"
#       if [ ! -f ${EPITRAN_BASE}/${ds}.epitran ]; then
#          echo "Epitran output for dataset ${ds} not found at ${EPITRAN_BASE}/${ds}.epitran. Skipping evaluation for this dataset."
#          continue
#       fi
#       if [ ! -f exp/runs/decodedv3.${ds}/${train_run_folder}.ck${stepnum}/transcription.json ]; then
#          python scripts/jsonl2json.py --dirname exp/runs/decodedv3.${ds}/${train_run_folder}.ck${stepnum}
#       fi
#       python -m src.metrics.phone_recognition \
#           --prediction_file exp/runs/decodedv3.${ds}/${train_run_folder}.ck${stepnum}/transcription.json \
#           --output_file exp/runs/ipapack_ctc/paperresults-${train_run_folder}.csv \
#           --gt_file ${EPITRAN_BASE}/${ds}.epitran \
#           --evaluation_name ${train_run_folder}-${ds}_epitran-${stepnum} \
#           --key_field utt_id &
#    done
#    wait
#    echo "=========================="
#    cut -d',' -f1-11 exp/runs/ipapack_ctc/paperresults-${train_run_folder}.csv
#    echo "=========================="
# done