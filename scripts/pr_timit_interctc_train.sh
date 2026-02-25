#  0.2 0.4 0.6 0.8 1.0
for mixing in 0.2 0.4 0.6 0.8 1.0; do
   echo "Submitting interCTC training job with Epitran mix ratio: $mixing"
   sbatch --time=2:00:00 -p ghx4 --reservation sup-22955 --nodelist=gh093 \
        -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
        scripts/daixpr.batch \
        experiment=train/timit_xeuspr_interctc \
        model.net.interctc_layer_idx='[4,8,12]' \
        callbacks.model_checkpoint.save_top_k=0 \
        model.net.vocab_file=src/model/xeusphoneme/resources/arpabet_vocab.json \
        model.net.ctc_config.ctc_type=builtin \
        run_folder=xeus_timit.vanilla_interctc_l4_8_12.mix${mixing}.5ksteps \
        data.epitran_mix_ratio=$mixing
done

# for mixing in 0.0; do
#    echo "Submitting interCTC training job with Epitran mix ratio: $mixing"
#    sbatch --time=2:00:00 -p ghx4-interactive \
#         -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
#         scripts/daixpr.batch \
#         experiment=train/timit_xeuspr_interctc \
#         model.net.interctc_layer_idx='[4,8,12]' \
#         callbacks.model_checkpoint.save_top_k=0 \
#         model.net.vocab_file=src/model/xeusphoneme/resources/arpabet_vocab.json \
#         model.net.ctc_config.ctc_type=builtin \
#         run_folder=xeus_timit.vanilla_interctc_l4_8_12.mix${mixing}.5ksteps \
#         data.epitran_mix_ratio=$mixing
# done