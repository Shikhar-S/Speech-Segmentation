for mixing in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
   echo "Submitting joint CTC+Attention training job with Epitran mix ratio: $mixing"
   sbatch --time=2:30:00 -p ghx4 \
        -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
        scripts/daixpr.batch \
        experiment=train/timit_xeuspr_joint \
        run_folder=xeus_timit.joint_ctc_att.mix${mixing}.5ksteps \
        data.epitran_mix_ratio=$mixing
done
