# ckpts=( \
#    # '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.panphon.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    # '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.schedule_panphon_4k_vanilla.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt'
# )

# DS=(epadb gmuaccent buckeye speechoceannotth l2arctic_perceived voxangeles timit)
# DS=(aishell cv fleurs fleurs_indv kazakh librispeech mls_dutch mls_french mls_german mls_italian mls_polish mls_portuguese mls_spanish southengland tamil)
DS=(timit)
# in order
# 1. panphon w/ ls point2 and then vanilla ctc
# 2. accent mapping based denoising
# 3. panphon w/ ls point2 throughout
# 4. l1-l2 mapping based denoising

# ckpts=( \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.accent_ls2.bs240.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-8000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.accent_ls2.bs240.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-8000.ckpt' \
   
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.schedule_panphonlsp2_4k_vanilla.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.panphon_ls2.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.manlang_ls2.bs240.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
   
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.schedule_vanilla_4k_panphon.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_huper.vanilla.bs128.lr3em5.sched_p10warm_p90const_500unfreeze.8ksteps/checkpoints/checkpoint-2500.ckpt' \
#    # huper data checkpoint
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.oracle_ls7.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-8000.ckpt' \
   
# )

# # vanilla then oracle
# # oracle with scheduling -- 1 run
# # full run on ipapack with panphon
# ckpts=( \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.sched_vanilla_oracle_ls2.bs256.lr3em5.sched_p0warm_p100const_1unfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.losssched_half2.5k_m12to0.oracle.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-8000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.losssched_half30k_m12tomp5.panphonk8.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt' \
# )
# # more oracle runs? if not, then delete

# inter-ctc
ckpts=( '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.interctc_l4_8_12.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt' \
        '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.losssched_half30k_m12tomp5.panphonk8.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps/checkpoints/checkpoint-22000.ckpt' \
)

for ds in ${DS[@]}; do
   for ckpt in ${ckpts[@]}; do
      echo "Evaluating checkpoint: $ckpt on dataset: $ds"
      train_run_folder=$(basename "${ckpt%/checkpoints/*}")
      stepnum=$(basename $ckpt | sed 's/checkpoint-\(.*\).ckpt/\1/')
      echo "Transcribing checkpoint: $ckpt on dataset: $ds"
      # Transcribe! -p gpuA40x4 scripts/deltaxpr.batch
      sbatch -p ghx4-interactive --time=1:00:00 scripts/daixpr_inference.batch \
         experiment=inference/transcribe_xeuspr \
         data=powsmeval data.dataset_name=$ds \
         task_name=decodedv3.${ds} \
         inference.num_workers=4 \
         inference.inference_runner.checkpoint=$ckpt \
         run_folder="${train_run_folder}.ck${stepnum}"
   done
done