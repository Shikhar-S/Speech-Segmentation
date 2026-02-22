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

The code directly downloads data from huggingface repo `speech31/easycall-dysarthria` thanks to Eunjung Yeo!

## Citation

If you use this dataset, please cite the following paper:

```
Turrisi, R., Braccia, A., Emanuele, M., Giulietti, S., Pugliatti, M., Sensi, M., Fadiga, L., Badino, L. (2021) 
EasyCall Corpus: A Dysarthric Speech Dataset. 
Proc. Interspeech 2021, 41-45, doi: 10.21437/Interspeech.2021-549
```

## Processing Pipeline

1. **Text Extraction**: Text is extracted from audio filenames (format: `speaker_session_text.wav`)
2. **IPA Conversion**: Text is converted to IPA phonemes using Epitran (Italian language model)
3. **Phoneme Segmentation**: IPA strings are segmented into individual phonemes using ipatok
4. **Tokenization**: Phonemes are converted to token IDs using the wav2vec2phoneme tokenizer