# ckpts=( \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.panphon.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.schedule_vanilla_4k_panphon.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
#    '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.schedule_panphon_4k_vanilla.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt'
# )

DS=(speechoceannotth l2arctic_perceived)
# DS=(epadb gmuaccent buckeye)

# in order
# 1. panphon w/ ls point2 and then vanilla ctc
# 2. accent mapping based denoising
# 3. panphon w/ ls point2 throughout
# 4. l1-l2 mapping based denoising

ckpts=( \
   '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
   '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.accent_ls2.bs240.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
   '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-8000.ckpt' \
   '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.accent_ls2.bs240.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-8000.ckpt' \
   
   '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.schedule_panphonlsp2_4k_vanilla.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
   '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.panphon_ls2.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
   '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/ipaaccent_ctc/xeus_multiaccent.manlang_ls2.bs240.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps/checkpoints/checkpoint-4000.ckpt' \
)

for ckpt in ${ckpts[@]}; do
   train_run_folder=$(basename "${ckpt%/checkpoints/*}")
   stepnum=$(basename $ckpt | sed 's/checkpoint-\(.*\).ckpt/\1/')
   for ds in ${DS[@]}; do
      echo "Evaluating checkpoint: $ckpt on dataset: $ds"
      python scripts/jsonl2json.py --dirname exp/runs/decodedv3.${ds}/${train_run_folder}.ck${stepnum}
      echo "merged entries, now evaluating metrics..."
      python -m src.metrics.phone_recognition \
          --prediction_file exp/runs/decodedv3.${ds}/${train_run_folder}.ck${stepnum}/transcription.json \
          --output_file exp/runs/ipapack_ctc/results-${train_run_folder}.csv \
          --gt_field target \
          --evaluation_name ${train_run_folder}-${ds}-${stepnum} \
          --key_field utt_id &
   done
   wait
   echo "=========================="
   cut -d',' -f1-11 exp/runs/ipapack_ctc/results-${train_run_folder}.csv
   echo "=========================="
done
