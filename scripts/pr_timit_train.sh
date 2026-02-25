# for mixing in 0.0 0.2 0.4 0.6 0.8 1.0; do
#    echo "Submitting training job with Epitran mix ratio: $mixing"
#    # -p ghx4-interactive
#    sbatch --time=2:00:00 \
#         -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
#         scripts/daixpr.batch \
#         callbacks.model_checkpoint.save_top_k=0 \
#         experiment=train/timit_xeuspr_epitran_mix \
#         model.net.ctc_config.ctc_type=builtin \
#         run_folder=xeus_timit.vanilla.mix${mixing}.bs32.lr5em5.5ksteps \
#         data.epitran_mix_ratio=$mixing
# done

# # 0.0 0.2 0.4 0.6 0.8 1.0
# for mixing in 0.1 0.3 0.5 0.7 0.9; do
#    echo "Submitting training job with Epitran mix ratio: $mixing"
#    # -p ghx4-interactive
#    sbatch --time=2:00:00 \
#         -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
#         scripts/daixpr.batch \
#         callbacks.model_checkpoint.save_top_k=0 \
#         experiment=train/timit_xeuspr_epitran_mix \
#         model.net.ctc_config.ctc_type=builtin \
#         run_folder=xeus_timit.arpa_vanilla.mix${mixing}.bs32.lr5em5.5ksteps \
#         model.net.vocab_file=src/model/xeusphoneme/resources/arpabet_vocab.json \
#         data.epitran_mix_ratio=$mixing
# done


#### approaches

# for mixing in 0.0 0.2 0.4 0.6 0.8 1.0; do
#    echo "Submitting training job with Epitran mix ratio: $mixing"
#    # -p ghx4-interactive
#    sbatch --time=2:00:00 \
#         -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
#         scripts/daixpr.batch \
#         experiment=train/timit_xeuspr_epitran_mix \
#         callbacks.model_checkpoint.save_top_k=0 \
#         model.net.vocab_file=src/model/xeusphoneme/resources/arpabet_vocab.json \
#         model.net.ctc_config.ctc_type=panphon_distance \
#         model.net.ctc_config.artctc_beta=10 \
#         +model.net.ctc_config.penalty_halflife=500 \
#         +model.net.ctc_config.penalty_init=-12 \
#         +model.net.ctc_config.penalty_final=-1.5 \
#         run_folder=xeus_timit.arpa_panphon.mix${mixing}.losssched_half5h_m12tom1p5_beta10.bs32.lr5em5.5ksteps \
#         data.epitran_mix_ratio=$mixing
# done



# for mixing in 0.1 0.3 0.5 0.7 0.9; do
#    echo "Submitting training job with Epitran mix ratio: $mixing"
#    # -p ghx4-interactive
#    sbatch --time=2:00:00 \
#         -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
#         scripts/daixpr.batch \
#         experiment=train/timit_xeuspr_epitran_mix \
#         callbacks.model_checkpoint.save_top_k=0 \
#         model.net.vocab_file=src/model/xeusphoneme/resources/arpabet_vocab.json \
#         model.net.ctc_config.ctc_type=panphon_distance \
#         model.net.ctc_config.artctc_beta=10 \
#         +model.net.ctc_config.penalty_halflife=500 \
#         +model.net.ctc_config.penalty_init=-12 \
#         +model.net.ctc_config.penalty_final=-1.5 \
#         run_folder=xeus_timit.arpa_panphon.mix${mixing}.losssched_half5h_m12tom1p5_beta10.bs32.lr5em5.5ksteps \
#         data.epitran_mix_ratio=$mixing
# done


for mixing in 0.1 0.3 0.5 0.7 0.9 0.0 0.2 0.4 0.6 0.8 1.0; do
   echo "Submitting training job with Epitran mix ratio: $mixing"
   # -p ghx4-interactive
   sbatch --time=2:00:00 \
        -c 72 --ntasks-per-node=1 --gpus-per-node=1 --mem=120G \
        scripts/daixpr.batch \
        experiment=train/timit_xeuspr_epitran_mix \
        callbacks.model_checkpoint.save_top_k=0 \
        model.net.vocab_file=src/model/xeusphoneme/resources/arpabet_vocab.json \
        model.net.ctc_config.ctc_type=panphon_distance \
        model.net.ctc_config.artctc_beta=10 \
        +model.net.ctc_config.penalty_halflife=500 \
        +model.net.ctc_config.penalty_init=-12 \
        +model.net.ctc_config.penalty_final=-0.5 \
        run_folder=xeus_timit.fix.arpa_panphon.mix${mixing}.losssched_half5h_m12tom0p5_beta10.bs32.lr5em5.5ksteps \
        data.epitran_mix_ratio=$mixing
done