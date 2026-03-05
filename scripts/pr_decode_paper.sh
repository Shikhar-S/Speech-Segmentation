extra_args=()
DS=(gmuaccent timit l2arctic_perceived voxangeles doreco tusom2021) # prism intrinsic
# DS=(epadb buckeye speechoceannotth) # ood
# DS=(aishell cv fleurs fleurs_indv kazakh librispeech mls_dutch mls_french mls_german mls_italian mls_polish mls_portuguese mls_spanish southengland tamil) # indomain
# DS=(timit)
###########################################################################
# ssl variants
# ebranchformer - none
# ckpts=(
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/ebranch12l_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-18000.ckpt'
# )
# transcription_config=inference/transcribe_xeuspr
# extra_args=( inference.inference_runner.config_file='exp/data/xeus_configs/xeus.12layer.yaml' )

# mms-300M
# this is actually 18k
# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/mms_multiaccent.bs320.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-6000-v3.ckpt')
# transcription_config=inference/transcribe_mmspr
# extra_args=()

# mms-1b
# vanilla
# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/mms1b_multiaccent.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt')
# transcription_config=inference/transcribe_mmspr
# extra_args=( inference.inference_runner.hf_repo='facebook/mms-1b' )


# # mms-1b on same data as other ssl variants
# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/mms1b_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-18000.ckpt')
# transcription_config=inference/transcribe_mmspr
# extra_args=( inference.inference_runner.hf_repo='facebook/mms-1b' )

# xeus with same data as used for other ssl variatns
# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-18000.ckpt')
# transcription_config=inference/transcribe_xeuspr
# extra_args=()

# # ebranchformer - 700M
# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/ebranch.selfctc_l4_8_12.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt')
# transcription_config=inference/transcribe_xeuspr_selfctc
# extra_args=() # same architecture as xeus! no config overrides needed
###########################################################################

# # ctc variants
# # vanilla
# # interctc
# ckpts=(
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.losssched_half30k_m12tomp5.panphonk8.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.interctc_l4_8_12.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt' \
# )
# transcription_config=inference/transcribe_xeuspr

# # joint-ctc-attn
# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/cjli/ctcattn/checkpoint-22000.ckpt')
# transcription_config=inference/transcribe_xeuspr_joint
# extra_args=( inference.num_workers=2 )
# # joint-ctc-attn but decoded using only encoder
# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/cjli/ctcattn/checkpoint-22000.ckpt')
# transcription_config=inference/transcribe_xeuspr
# extra_args=()

# orthographic auxiliary loss with interctc
# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_auxctc/xeus_multiaccent.ortho_ch8k_ctc_l4_8_12.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt')
# transcription_config=inference/transcribe_xeuspr_auxctc
# extra_args=(inference.inference_runner.ctc_aux_config.vocab_file='exp/data/train8M.char8k.model')

# mms1-b w/ selfctc inference w/o conditioning
ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/mms1b_multiaccent.selfctc_l4_8_12_16.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt')
transcription_config=inference/transcribe_mmspr
extra_args=( inference.inference_runner.hf_repo='facebook/mms-1b' )
##############################################################################

for ds in ${DS[@]}; do
   for ckpt in ${ckpts[@]}; do
      train_run_folder=$(basename "${ckpt%/checkpoints/*}")
      stepnum=$(basename $ckpt | sed 's/checkpoint-\(.*\).ckpt/\1/')
      # skip if the jsonl already exists
      if [ -f "data/powsmeval/decodedv3.${ds}/${train_run_folder}.woselfctc.ck${stepnum}/predictions.jsonl" ]; then
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
         run_folder="${train_run_folder}.woselfctc.ck${stepnum}"
      )
      sbatch_cmd_args+=( "${extra_args[@]}" )
     
      # RUN --reservation sup-22955 --nodelist=gh091,gh146
      sbatch -p ghx4 --reservation sup-22955 --nodelist=gh146 --time=8:00:00 \
         scripts/daixpr_inference.batch \
         "${sbatch_cmd_args[@]}"
   done
done
