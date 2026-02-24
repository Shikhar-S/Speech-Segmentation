# 0.0 0.2 0.4 0.6 0.8 1.0
for mixing in 0.0; do
   echo "Submitting training job with Epitran mix ratio: $mixing"
   # -p ghx4-interactive
   sbatch -p ghx4-interactive --time=2:00:00 \
        -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
        scripts/daixpr.batch \
        experiment=train/timit_xeuspr_epitran_mix \
        model.net.ctc_config.ctc_type=builtin \
        run_folder=xeus_timit.vanilla.mix${mixing}.bs32.lr5em5.5ksteps \
        data.epitran_mix_ratio=$mixing
done