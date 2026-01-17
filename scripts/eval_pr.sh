#  ckpts=( '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251225_221757/checkpoints/step_010000.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251225_221757/checkpoints/step_020000.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251225_221757/checkpoints/step_030000.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251225_221757/checkpoints/step_042324.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251225_221757/checkpoints/step_052324.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251225_221757/checkpoints/step_062324.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251225_221757/checkpoints/step_074648.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251225_221757/checkpoints/step_084648.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251227_095929/checkpoints/step_111972.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251227_095929/checkpoints/step_116972.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251227_095929/checkpoints/step_121972.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251227_095929/checkpoints/step_126972.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251227_095929/checkpoints/step_131972.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251227_095929/checkpoints/step_136972.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251227_095929/checkpoints/step_144296.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251227_095929/checkpoints/step_149296.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251227_095929/checkpoints/step_154296.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251227_095929/checkpoints/step_159296.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251229_100016/checkpoints/step_186620.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251229_100016/checkpoints/step_191620.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251229_100016/checkpoints/step_196620.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251229_100016/checkpoints/step_201620.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251229_100016/checkpoints/step_208944.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251229_100016/checkpoints/step_213944.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251229_100016/checkpoints/step_218944.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251229_100016/checkpoints/step_223944.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251229_100016/checkpoints/step_228944.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251229_100016/checkpoints/step_233944.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251231_100230/checkpoints/step_241268.ckpt' '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251231_100230/checkpoints/step_246268.ckpt')
ckpts=('/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/train_ipapack_xeuspr/20251225_221757/checkpoints/step_030000.ckpt')

# OOD sets # buckeye cv epadb fleurs fleurs_indv  
# DS=(gmuaccent l2arctic_perceived timit tusom2021 voxangeles doreco) # buckeye speechoceannotth
# Indomain sets
DS=(librispeech mls_german mls_dutch mls_french mls_italian mls_spanish mls_portuguese mls_polish tamil kazakh aishell)

for ds in ${DS[@]}; do
   for ckpt in ${ckpts[@]}; do
      echo "Evaluating checkpoint: $ckpt"
      stepnum=$(basename $ckpt | sed 's/step_\(.*\).ckpt/\1/')
      # Transcribe!
      # sbatch --array=0-20 scripts/deltaxpr.batch \
      #    experiment=inference/transcribe_xeuspr \
      #    data=powsmeval data.dataset_name=$ds \
      #    task_name=inf_${ds}_xeuspr \
      #    inference.num_workers=4 \
      #    inference.inference_runner.checkpoint=$ckpt \
      #    run_folder="$stepnum"
      
      python scripts/jsonl2json.py --dirname exp/runs/inf_${ds}_xeuspr/$stepnum
      python -m src.metrics.phone_recognition \
          --prediction_file exp/runs/inf_${ds}_xeuspr/$stepnum/transcription.json \
          --output_file exp/runs/xeuspr_results/results.csv \
          --gt_field target \
          --evaluation_name ${ds}-${stepnum} \
          --key_field utt_id &
   done
done

wait
echo "=========================="
cut -d',' -f1-6 exp/runs/xeuspr_results/results.csv
echo "=========================="