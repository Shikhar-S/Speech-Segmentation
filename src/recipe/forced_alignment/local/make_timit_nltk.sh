#!/usr/bin/env bash
# Restructure timit data downlaoded from https://figshare.com/articles/dataset/TIMIT_zip/5802597 
# so that it works with 
# https://www.nltk.org/api/nltk.corpus.reader.timit.html#nltk.corpus.reader.timit.TimitCorpusReader
# Reorganizes LDC TIMIT into NLTK-style layout in timit_nltk/
# Run from the TIMIT root (where TRAIN/, TEST/, DOC/ live).
# Usage: bash /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/src/recipe/forced_alignment/local/make_timit_nltk.sh /work/hdd/bbjs/shared/corpora/TIMIT

set -euo pipefail

ROOT=$1
OUT="${ROOT}/timit_nltk"

echo "Creating NLTK-style TIMIT tree in: ${OUT}"
mkdir -p "${OUT}"

##############################
# Copy dictionary & speakers #
##############################

# timitdic.txt
if [[ -f "${ROOT}/DOC/TIMITDIC.TXT" ]]; then
    cp "${ROOT}/DOC/TIMITDIC.TXT" "${OUT}/timitdic.txt"
else
    echo "WARNING: Could not find TIMIT dictionary file in DOC/."
fi

# spkrinfo.txt
if [[ -f "${ROOT}/DOC/SPKRINFO.TXT" ]]; then
    cp "${ROOT}/DOC/SPKRINFO.TXT" "${OUT}/spkrinfo.txt"
elif [[ -f "${ROOT}/DOC/spkrinfo.txt" ]]; then
    cp "${ROOT}/DOC/spkrinfo.txt" "${OUT}/spkrinfo.txt"
else
    echo "WARNING: Could not find SPKRINFO file in DOC/."
fi

#########################################
# Copy utterance files into new layout  #
# - speaker dir: drX-speakerid (lower)  #
# - files: sa1.txt/.wrd/.phn/.wav etc   #
#########################################

split_index_file=${OUT}/split_index.txt
# Process both TRAIN and TEST trees if they exist
for SPLIT in TRAIN TEST; do
    [[ -d "${ROOT}/${SPLIT}" ]] || continue
    echo "Processing split: ${SPLIT}"

    # Find each WAV file and use it to drive copying of all four file types
    find "${ROOT}/${SPLIT}" -type f -name "*.WAV" | while read -r wav; do
        # Example path: TRAIN/DR1/FDAW0/SA1.WAV
        speakerdir="$(dirname "${wav}")"                 # TRAIN/DR1/FDAW0
        filebase="$(basename "${wav}" .WAV)"            # SA1
        dialect="$(basename "$(dirname "${speakerdir}")")"  # DR1
        speaker="$(basename "${speakerdir}")"               # FDAW0

        # Lowercase
        dialect_lc="${dialect,,}"   # dr1
        speaker_lc="${speaker,,}"   # fdaw0
        utt_lc="${filebase,,}"      # sa1

        # NLTK-style speaker dir: dr1-fdaw0
        new_speaker_dir="${OUT}/${dialect_lc}-${speaker_lc}"
        mkdir -p "${new_speaker_dir}"

        # Copy & rename the four associated files, if they exist
        for EXT in TXT WRD PHN WAV; do
            src="${speakerdir}/${filebase}.${EXT}"
            if [[ -f "${src}" ]]; then
                # lowercase extension and basename
                ext_lc="${EXT,,}"
                dest="${new_speaker_dir}/${utt_lc}.${ext_lc}"
                cp "${src}" "${dest}"
                echo "${SPLIT} ${dest}" >> "${split_index_file}"
            fi
        done
    done
done

echo "Done. NLTK-style corpus root is: ${OUT}"
echo "Example speaker dir: dr1-fdaw0/ with sa1.txt / sa1.wrd / sa1.phn / sa1.wav"