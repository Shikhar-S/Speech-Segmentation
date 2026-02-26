DS=(epadb gmuaccent buckeye speechoceannotth l2arctic_perceived voxangeles)
DS=(aishell cv fleurs fleurs_indv kazakh librispeech mls_dutch mls_french mls_german mls_italian mls_polish mls_portuguese mls_spanish southengland tamil)

ckpts=(
    /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/mms_multiaccent.bs256.lr5em5/checkpoints/checkpoint-14000.ckpt \
)

for ds in ${DS[@]}; do
   for ckpt in ${ckpts[@]}; do
      echo "Evaluating checkpoint: $ckpt on dataset: $ds"
      train_run_folder=$(basename "${ckpt%/checkpoints/*}")
      stepnum=$(basename $ckpt | sed 's/checkpoint-\(.*\).ckpt/\1/')
      echo "Transcribing checkpoint: $ckpt on dataset: $ds"
      # Transcribe! -p gpuA40x4 scripts/deltaxpr.batch
      sbatch -p ghx4 --time=4:00:00 scripts/daixpr_inference.batch \
         experiment=inference/transcribe_mmspr \
         data=powsmeval data.dataset_name=$ds \
         task_name=decodedv3.${ds} \
         inference.inference_runner.checkpoint=$ckpt \
         run_folder="${train_run_folder}.ck${stepnum}"
   done
done