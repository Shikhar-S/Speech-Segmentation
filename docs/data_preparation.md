# Data preparation

The segmentation datasets are derived from licensed corpora and cannot be
redistributed. The scripts in `scripts/data_prep/` build them from the original
releases. Each script reads a downloaded corpus, writes a Hugging Face
`DatasetDict` with the fields `utt_id`, `audio`, `text`, `phones`,
`phone_starts`, `phone_ends`, `language`, `speaker_id`, `duration`, and `split`,
and uploads it to a private Hugging Face repository that you specify.

After uploading, set the corresponding variable to the repository id. The run
scripts, Hydra configs, and per-dataset label transforms all resolve the dataset
through it:

```bash
export SEG_REPO_TIMIT=<org>/<repo>
# Likewise: SEG_REPO_BUCKEYE, SEG_REPO_GTIMIT_L2SIMPLE, SEG_REPO_GTIMIT_L2TBNK,
#           SEG_REPO_TORGO, SEG_REPO_SSNCE, SEG_REPO_VOXANGELES, SEG_REPO_GTIMIT_THA
```

## Sources and scripts

| Dataset | Source | Scripts |
|---|---|---|
| TIMIT | [LDC93S1W](https://catalog.ldc.upenn.edu/LDC93S1W) (RIFF WAV). The [LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1) release uses NIST SPHERE audio and must be converted first. | `make_timit_nltk.sh`, `timit_data_prep.py`, `processed_data_convert.py --audio_root` |
| Buckeye | [buckeyecorpus.osu.edu](https://buckeyecorpus.osu.edu) (`s01.zip` to `s40.zip`) | `buckeye_data_prep.py`, `processed_data_convert.py --clips_dir` |
| Global TIMIT L2 Simple | [LDC2020S11](https://catalog.ldc.upenn.edu/LDC2020S11) | `gtimit_data_prep.py --subset L2ENGsimple`, `push_gtimit_to_hub.py` |
| Global TIMIT L2 Treebank | [LDC2020S09](https://catalog.ldc.upenn.edu/LDC2020S09) | `gtimit_data_prep.py --subset L2ENGtreebank`, `push_gtimit_to_hub.py` |
| Global TIMIT Thai | [LDC2022S13](https://catalog.ldc.upenn.edu/LDC2022S13) | `gtimit_data_prep.py --subset THA`, `push_gtimit_to_hub.py` |
| TORGO | [LDC2012S02](https://catalog.ldc.upenn.edu/LDC2012S02) or [University of Toronto](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) | `seg_data_convert.py` |
| SSNCE | [LDC2021S04](https://catalog.ldc.upenn.edu/LDC2021S04) | `seg_data_convert.py` |
| VoxAngeles | [pacscilab/voxangeles](https://github.com/pacscilab/voxangeles) and the PRiSM `test_voxangeles` dump | `voxangeles_data_convert.py` |

## Example: TIMIT

```bash
bash scripts/data_prep/make_timit_nltk.sh $TIMIT_ROOT
python scripts/data_prep/timit_data_prep.py \
    --timit_root  $TIMIT_ROOT/timit_nltk \
    --split_index $TIMIT_ROOT/timit_nltk/split_index.txt \
    --output_dir  exp/data/timit_meta
python scripts/data_prep/processed_data_convert.py \
    --metadata_dir exp/data/timit_meta \
    --audio_root   $TIMIT_ROOT/timit_nltk \
    --language     eng \
    --output_dir   exp/data/timit-segment \
    --hf_repo      <org>/<repo>
```

For Buckeye, `buckeye_data_prep.py` replaces the first two steps and also
writes the clipped audio; convert with `--clips_dir exp/data/buckeye_meta/speech_clips`.
For Global TIMIT, unpack the three LDC releases under one directory and pass it
as `--downloads_root`. Every script documents its arguments in its header.

## Correspondence with the paper

- **TIMIT.** The official 168-speaker test set is used for evaluation. A
  speaker-disjoint 10% of the training speakers, selected with a fixed seed,
  forms the validation set.
- **Buckeye.** Recordings are segmented at pauses of at least 1 s into
  utterances of 0.5 to 20 s, then split speaker-disjoint 80/10/10 with a fixed
  seed.
- **Global TIMIT, TORGO, SSNCE, VoxAngeles.** Evaluation only; all utterances
  are assigned to the `test` split. VoxAngeles follows the PRiSM utterance list.
- **Phone boundaries.** Boundaries are taken from each release's phone
  segmentation files: TIMIT `.phn`, Global TIMIT `.phones`, and TextGrids for
  SSNCE and TORGO. The TORGO and SSNCE TextGrids used in the paper are forced
  alignments corrected by annotators. `seg_data_convert.py` accepts any
  directory of `<utt_id>.TextGrid` files.
