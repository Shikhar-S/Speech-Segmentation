# UASpeech Dataset

## Overview

This folder contains the data module for the UASpeech dataset, an English Dysarthric Speech dataset used for intelligibility classification. The dataset includes both dysarthric and healthy speech samples from speakers with cerebral palsy.

## Dataset Information

The UASpeech dataset is an English dysarthric speech corpus. Each speaker is labeled with severity scores ranging from 0 to 4:
- **0**: Healthy speakers (control group)
- **1**: High intelligibility
- **2**: Mid intelligibility
- **3**: Low intelligibility
- **4**: Very low intelligibility

The dataset contains speech samples from speakers with cerebral palsy, reading isolated words from a wordlist. The dataset is split into train (60%), validation (20%), and test (20%) sets using stratified splitting based on intelligibility level.

## Dataset Download

The UASpeech dataset can be downloaded from: https://speechtechnology.web.illinois.edu/uaspeech/

## Citation

If you use this dataset, please cite the following paper:

```
Kim, H., Hasegawa-Johnson, M., Perlman, A., Gunderson, J., Huang, T.S., Watkin, K., Frame, S. (2008) Dysarthric speech database for universal access research. Proc. Interspeech 2008, 1741-1744, doi: 10.21437/Interspeech.2008-480
```

## License

Please be aware of the dataset license (directly citing dataset's license):
  * Permission to redistribute the data is EXPLICITLY WITHHELD.  Users
    of the database may not redistribute audio or video files outside
    of their own institution.  

  * Permission is granted for researchers at academic or government labs
    to use this database in any scientific or technological experiments.
    Permission is explicitly granted to train statistical models for purposes 
    such as speech technology and computer vision.  Permission is explicitly 
    granted to redistribute any models so trained, provided that it is not
    possible to reconstruct any original waveform segment or video
    segment from the distributed models.

  * Permission is granted to use images, video, and waveforms in
    presentations at professional conferences and/or in professional
    journals provided that the following reference is cited:

    Heejin Kim, Mark Hasegawa-Johnson, Adrienne Perlman, Jon
    Gunderson, Thomas Huang, Kenneth Watkin and Simone Frame,
    "Dysarthric Speech Database for Universal Access Research."
    In Proc. Interspeech, 2008, pp. 1741-1744

  * Neither the name of the University of Illinois nor the names of
    its contributors may be used to endorse or promote products
    derived from this database without specific prior written
    permission.

## Data Format

The dataset returns the following fields for each sample:

- **utt_id**: Utterance identifier (filename, e.g., `CF02_B1_D3.wav`)
- **text**: Text transcript extracted from filename using wordlist mapping
- **label**: Intelligibility label (0 for healthy, 1-4 for dysarthric speakers based on intelligibility level)
- **split**: Dataset split (`train`, `validation`, or `test`)
- **speech_length**: Length of speech signal in samples
- **target_length**: Number of phonemes in the target sequence
- **phones**: Space-separated string of IPA phonemes (e.g., `"θ r i"`)
- **target**: List of token IDs corresponding to the phonemes (e.g., `[45, 123, 89]`)
- **speech_shape**: Shape of the speech tensor
- **target_shape**: Shape of the target tensor

## Data Structure

The dataset comes in two compressed archive files:
- **Control group** (`.tgz`): Contains healthy speaker recordings
- **Dysarthric group** (`.tgz`): Contains dysarthric speaker recordings

When extracted, files are organized in subdirectories like `audio/noisereduce/`.

## Processing Pipeline

1. **Extraction**: Extract audio files from `.tgz` archives to a cache directory
2. **Text Extraction**: Text is extracted from audio filenames using a wordlist CSV mapping (format: `SPEAKER_SESSION_WORDID.wav`)
3. **IPA Conversion**: Text is converted to IPA phonemes using Epitran (English language model)
4. **Phoneme Segmentation**: IPA strings are segmented into individual phonemes using ipatok
5. **Tokenization**: Phonemes are converted to token IDs using the wav2vec2phoneme tokenizer

## Creating Splits

To create stratified train/validation/test splits (60%/20%/20%):

```bash
python phonebench/uaspeech/create_stratified_splits.py \
    --input_csv phonebench/uaspeech/uaspeech_sev.csv \
    --output_csv phonebench/uaspeech/uaspeech_meta.csv \
    --train_ratio 0.6 \
    --valid_ratio 0.2 \
    --test_ratio 0.2
```

This creates `uaspeech_meta.csv` with columns: `speaker`, `sex`, `severity`, `split`.

## Usage

### Python API

```python
from src.data.uaspeech.common_datamodule import UASpeechDataModule
from src.model.wav2vec2phoneme.builders import build_wav2vec2phoneme_tokenizer

# Initialize tokenizer
tokenizer = build_wav2vec2phoneme_tokenizer(
    hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
)

# Create data module
data_module = UASpeechDataModule(
    local_cache_path="/path/to/cache",
    uaspeech_meta_csv="/path/to/uaspeech_meta.csv",
    uaspeech_wordlist_csv="/path/to/uaspeech_wordlist.csv",
    tokenizer=tokenizer,
    tgz_file1="/path/to/UASpeech_noisereduce_C.tgz",
    tgz_file2="/path/to/UASpeech_noisereduce_FM.tgz",
    batch_size=32,
    num_workers=4,
)

# Setup datasets
data_module.setup()

# Get dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
```

### Command Line

#### Option 1: Extract files first, then use datamodule

```bash
# Extract files
python -m src.data.uaspeech.common_datamodule \
    --data_dir /path/to/cache \
    --tgz_file_C /path/to/UASpeech_noisereduce_C.tgz \
    --tgz_file_D /path/to/UASpeech_noisereduce_FM.tgz \
    --extract_only

# Use datamodule (files already extracted)
python -m src.data.uaspeech.common_datamodule \
    --data_dir /path/to/cache \
    --uaspeech_meta_csv src/data/uaspeech/uaspeech_meta.csv \
    --uaspeech_wordlist_csv src/data/uaspeech/uaspeech_wordlist.csv \
    --batch_size 32 \
    --num_workers 4 \
    --show_examples
```

#### Option 2: Let datamodule extract automatically

```bash
python -m src.data.uaspeech.common_datamodule \
    --data_dir /path/to/cache \
    --tgz_file_C /path/to/UASpeech_noisereduce_C.tgz \
    --tgz_file_D /path/to/UASpeech_noisereduce_FM.tgz \
    --uaspeech_meta_csv src/data/uaspeech/uaspeech_meta.csv \
    --uaspeech_wordlist_csv src/data/uaspeech/uaspeech_wordlist.csv \
    --batch_size 32 \
    --num_workers 4 \
    --show_examples
```

**Note**: 
- The `--show_examples` flag will print 5 example samples from the test loader for inspection.
- If files are already extracted in `data_dir`, `tgz_file_C` and `tgz_file_D` are optional.
- `--uaspeech_meta_csv` and `--uaspeech_wordlist_csv` have defaults and are optional.

## Dependencies

- `epitran`: For text-to-IPA conversion (English)
- `ipatok`: For IPA phoneme segmentation
- `torchaudio`: For audio loading and processing
- `pandas`: For CSV metadata handling
- `tqdm`: For extraction progress bars

