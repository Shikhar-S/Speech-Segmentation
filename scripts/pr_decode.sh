# ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt')
ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.panphon.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt'
   '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt')
# DS=(gmuaccent buckeye epadb speechoceanotth l2arctic)
DS=(gmuaccent)

for ds in ${DS[@]}; do
   for ckpt in ${ckpts[@]}; do
      echo "Evaluating checkpoint: $ckpt on dataset: $ds"
      train_run_folder=$(basename "${ckpt%/checkpoints/*}")
      stepnum=$(basename $ckpt | sed 's/checkpoint-\(.*\).ckpt/\1/')
      # Transcribe!
      # --array=0-3
      # sbatch --array=0-3 -p gpuA40x4 scripts/deltaxpr.batch \
      python src/main.py \
         experiment=inference/transcribe_xeuspr \
         data=powsmeval data.dataset_name=$ds \
         task_name=decoded.${ds} \
         inference.num_workers=4 \
         inference.inference_runner.checkpoint=$ckpt \
         run_folder="${train_run_folder}.ck${stepnum}"
   done
done