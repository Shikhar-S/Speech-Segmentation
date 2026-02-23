set -e

DS=(
  buckeye
  epadb
  gmuaccent
  speechoceannotth
)

OUTCSV="exp/runs/ipapack_ctc/results-epitran.csv"
BASE="/work/hdd/bbjs/shared/powsm/s2t1/dump/raw"
HYP_DIR="exp/data/epitran_outputs"

# remove old file if you want fresh results
rm -f ${OUTCSV}

for ds in "${DS[@]}"; do
    echo "Evaluating Epitran on dataset: $ds"

    python -m src.recipe.phone_recognition.local.epitran_pfer \
        --ref ${BASE}/test_${ds}/text.good \
        --hyp ${HYP_DIR}/${ds}.epitran \
        --evaluation_name epitran-${ds} \
        --output_file ${OUTCSV} &
done

wait

echo "=========================="
cut -d',' -f1-11 ${OUTCSV}
echo "=========================="