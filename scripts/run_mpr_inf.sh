PROB=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Runs for RQ1: Masked PR inference
# BUCKEYE
for mp in ${PROB[@]}; do 
    scripts/run.sh \
        --setup inference \
        --model ctag,lv60,xlsr53,powsm \
        --extra_args "data=buckeye data.mask_probability=$mp"
    sleep 1s
done

# TIMIT
for mp in ${PROB[@]}; do 
    scripts/run.sh \
        --setup inference \
        --model ctag,lv60,xlsr53,powsm \
        --extra_args "data=timit data.mask_probability=$mp"
    sleep 1s
done

# ZIPACTC , buckeye
for mp in ${PROB[@]}; do 
    scripts/run.sh \
        --setup inference \
        --model zipactc,zipactc_ns \
        --extra_args "data=buckeye data.mask_probability=$mp"
    sleep 1s
done