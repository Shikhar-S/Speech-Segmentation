# EasyCall Dataset

## Overview

This folder contains the data module for the EasyCall dataset, an Italian Dysarthric Speech dataset used for dysarthria severity classification. The dataset includes both dysarthric and healthy speech samples.

## Dataset Information

The EasyCall dataset is an Italian dysarthric speech corpus. Each speaker is labeled with Therapy Output Measurement (TOM) scores, which originally range from 1 to 5, corresponding to:
- 1: Mild
- 2: Mild-moderate
- 3: Moderate
- 4: Moderate-severe
- 5: Severe

For this experiment, we merge the labels as follows:
- **Score 0**: Healthy speakers (control group)
- **Score 1**: Mild dysarthria
- **Score 2**: Mild-moderate + Moderate dysarthria
- **Score 3**: Moderate-severe + Severe dysarthria

## Dataset Download

The EasyCall dataset can be downloaded from: http://neurolab.unife.it/easycallcorpus/

## Citation

If you use this dataset, please cite the following paper:

```
Turrisi, R., Braccia, A., Emanuele, M., Giulietti, S., Pugliatti, M., Sensi, M., Fadiga, L., Badino, L. (2021) 
EasyCall Corpus: A Dysarthric Speech Dataset. 
Proc. Interspeech 2021, 41-45, doi: 10.21437/Interspeech.2021-549
```

## Data Format

The dataset returns the following fields for each sample:

- **utt_id**: Utterance identifier (filename, e.g., `f06_03_Mali.wav`)
- **text**: Text transcript extracted from filename
- **label**: Dysarthria severity label (0 for healthy, 1-3 for dysarthric speakers)
- **split**: Dataset split (`train`, `validation`, or `test`)
- **speech_length**: Length of speech signal in samples
- **target_length**: Number of phonemes in the target sequence
- **phones**: Space-separated string of IPA phonemes (e.g., `"m a l i"`)
- **target**: List of token IDs corresponding to the phonemes (e.g., `[36, 231, 156, 238]`)
- **speech_shape**: Shape of the speech tensor
- **target_shape**: Shape of the target tensor

## Processing Pipeline

1. **Text Extraction**: Text is extracted from audio filenames (format: `speaker_session_text.wav`)
2. **IPA Conversion**: Text is converted to IPA phonemes using Epitran (Italian language model)
3. **Phoneme Segmentation**: IPA strings are segmented into individual phonemes using ipatok
4. **Tokenization**: Phonemes are converted to token IDs using the wav2vec2phoneme tokenizer

## Usage

### Python API

```python
from src.data.easycall.common_datamodule import EasyCallDataModule
from src.model.wav2vec2phoneme.builders import build_wav2vec2phoneme_tokenizer

# Initialize tokenizer
tokenizer = build_wav2vec2phoneme_tokenizer(
    hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
)

# Create data module
data_module = EasyCallDataModule(
    easycall_root="/path/to/EasyCall",
    local_cache_path="/path/to/cache",
    easycall_meta_csv="/path/to/easycall_meta.csv",
    tokenizer=tokenizer,
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

```bash
python -m src.data.easycall.common_datamodule \
    --easycall_root /path/to/EasyCall \
    --data_dir /path/to/cache \
    --easycall_meta_csv /path/to/easycall_meta.csv \
    --batch_size 32 \
    --num_workers 4 \
    --show_examples
```

**Note**: The `--show_examples` flag will print 5 example samples from the test loader for inspection.
