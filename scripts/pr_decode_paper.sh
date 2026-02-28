extra_args=()
DS=(gmuaccent timit l2arctic_perceived voxangeles doreco tusom2021) # prism intrinsic
# DS=(epadb buckeye speechoceannotth) # ood
# DS=(aishell cv fleurs fleurs_indv kazakh librispeech mls_dutch mls_french mls_german mls_italian mls_polish mls_portuguese mls_spanish southengland tamil) # indomain
# DS=(timit)
###########################################################################
# ssl variants
# ebranchformer - none
# ckpts=(
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/ebranch12l_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-24000.ckpt'
# )
# transcription_config=inference/transcribe_xeuspr
# extra_args=( inference.inference_runner.config_file='exp/data/xeus_configs/xeus.12layer.yaml' )
# mms-300M
# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/mms_multiaccent.bs320.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-8000.ckpt')
# transcription_config=inference/transcribe_mmspr
# extra_args=()
# mms-1b
# vanilla
# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/mms1b_multiaccent.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt')
# transcription_config=inference/transcribe_mmspr
# extra_args=( inference.inference_runner.hf_repo='facebook/mms-1b' )
# xeus with same data as used for other ssl variatns
ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-18000.ckpt')
transcription_config=inference/transcribe_xeuspr
extra_args=()
###########################################################################

# # ctc variants
# # vanilla
# # interctc
# ckpts=(
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.losssched_half30k_m12tomp5.panphonk8.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.interctc_l4_8_12.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt' \
# )
# transcription_config=inference/transcribe_xeuspr
##############################################################################

for ds in ${DS[@]}; do
   for ckpt in ${ckpts[@]}; do
      train_run_folder=$(basename "${ckpt%/checkpoints/*}")
      stepnum=$(basename $ckpt | sed 's/checkpoint-\(.*\).ckpt/\1/')
      # skip if the jsonl already exists
      if [ -f "data/powsmeval/decodedv3.${ds}/${train_run_folder}.ck${stepnum}/predictions.jsonl" ]; then
         echo "Predictions already exist for checkpoint: $ckpt on dataset: $ds, skipping transcription."
         continue
      fi
      echo "Transcribing checkpoint: $ckpt on dataset: $ds"
      
      sbatch_cmd_args=(
         experiment=${transcription_config} \
         data=powsmeval data.dataset_name=$ds \
         task_name=decodedv3.${ds} \
         inference.num_workers=4 \
         inference.inference_runner.checkpoint=$ckpt \
         run_folder="${train_run_folder}.ck${stepnum}"
      )
      sbatch_cmd_args+=( "${extra_args[@]}" )
     
      # RUN
      sbatch -p ghx4 --reservation sup-22955 --nodelist=gh091,gh146 --time=1:30:00 \
         scripts/daixpr_inference.batch \
         "${sbatch_cmd_args[@]}"
   done
done
