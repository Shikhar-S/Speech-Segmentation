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

## Creating Splits

Stratified train/validation/test splits (60%/20%/20%) are already created
in `uaspeech_meta.csv` with columns: `speaker`, `sex`, `severity`, `split`.

