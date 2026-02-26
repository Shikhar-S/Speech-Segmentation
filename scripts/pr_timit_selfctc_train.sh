for mixing in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
   echo "Submitting self-conditioned interCTC training job with Epitran mix ratio: $mixing"
   sbatch --time=2:30:00 -p ghx4 \
        -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
        scripts/daixpr.batch \
        experiment=train/timit_xeuspr_selfctc \
        callbacks.model_checkpoint.save_top_k=0 \
        model.net.vocab_file=src/model/xeusphoneme/resources/arpabet_vocab.json \
        model.net.ctc_config.ctc_type=builtin \
        model.net.interctc_layer_idx='[4,8,12]' \
        run_folder=xeus_timit.selfctc_l4_8_12.mix${mixing}.5ksteps \
        data.epitran_mix_ratio=$mixing
done
