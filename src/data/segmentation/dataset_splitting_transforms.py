"""DatasetDict-level split transforms for SegmentationDataModule.

``HF_REPO_SPLIT_TRANSFORMS`` maps a HuggingFace repo id to a function that
takes a ``DatasetDict`` and returns a new one with possibly different
splits. Used to carve out a deterministic ``tune`` split disjoint from
fit / eval. Applied inside ``SegmentationDataModule.setup`` and
``build_segmentation_dataset``.

# TODO(shikhar): Check these thoroughly again! Especially the splitting, speaker independent?
"""

from typing import Any, Callable, Dict

import numpy as np


def split_timit_train_for_tuning(ddict: Any) -> Any:
    """Carve a 1000-utt seed-42 ``tune`` split out of TIMIT ``train``.

    The remaining train utts stay in ``train`` (used for fitting
    PhonologicalVectors). The tune subset is disjoint.
    """
    train = ddict["train"]
    n = len(train)
    rng = np.random.default_rng(42)
    tune_idx = rng.choice(n, size=min(1000, n), replace=False)
    tune_set = set(tune_idx.tolist())
    train_idx = [i for i in range(n) if i not in tune_set]

    new_dd = {k: v for k, v in ddict.items()}
    new_dd["tune"] = train.select(sorted(tune_idx.tolist()))
    new_dd["train"] = train.select(train_idx)
    return new_dd


def split_voxangeles_test_for_tuning(ddict: Any) -> Any:
    """Carve VoxAngeles ``test`` into disjoint ``test`` (eval) + ``tune``."""
    test = ddict["test"]
    n = len(test)
    rng = np.random.default_rng(50)
    selected = rng.choice(n, size=min(2000, n), replace=False)
    eval_idx = selected[:1000].tolist()
    tune_idx = selected[1000:2000].tolist()

    new_dd = {k: v for k, v in ddict.items()}
    new_dd["test"] = test.select(eval_idx)
    new_dd["tune"] = test.select(tune_idx)
    return new_dd


def split_gtimit_test_speaker_independent(ddict: Any) -> Any:
    """Speaker-independent tune carve from GTIMIT ``test``.

    Picks 10 of the 50 speakers at random (seed=42); all their utterances
    go to ``tune``, all other speakers' utterances stay in ``test``. Tune
    and test speaker sets are disjoint, so tuning never sees an evaluation
    speaker. Yields ~1200 tune / ~4800 test rows on the standard 50-spk
    ×120-utt GTIMIT subsets. Used by all 5 changelinglab/gtimit-* repos.
    """
    test = ddict["test"]
    speakers = list(test["speaker_id"])
    unique = sorted(set(speakers))
    rng = np.random.default_rng(42)
    n_tune_spk = min(10, len(unique))
    chosen = rng.choice(len(unique), size=n_tune_spk, replace=False)
    tune_spk_set = {unique[int(i)] for i in chosen}
    tune_idx = [i for i, s in enumerate(speakers) if s in tune_spk_set]
    test_idx = [i for i, s in enumerate(speakers) if s not in tune_spk_set]

    new_dd = {k: v for k, v in ddict.items()}
    new_dd["tune"] = test.select(tune_idx)
    new_dd["test"] = test.select(test_idx)
    return new_dd


# Registry: HuggingFace repo id -> default DatasetDict-level split transform.
HF_REPO_SPLIT_TRANSFORMS: Dict[str, Callable[[Any], Any]] = {
    "changelinglab/timit-segment": split_timit_train_for_tuning,
    "changelinglab/voxangeles-segment": split_voxangeles_test_for_tuning,
    "changelinglab/gtimit-l2simple-segment": split_gtimit_test_speaker_independent,
    "changelinglab/gtimit-l2tbnk-segment": split_gtimit_test_speaker_independent,
    "changelinglab/gtimit-l1simple-segment": split_gtimit_test_speaker_independent,
    "changelinglab/gtimit-l1tbnk-segment": split_gtimit_test_speaker_independent,
    "changelinglab/gtimit-tha-segment": split_gtimit_test_speaker_independent,
}
