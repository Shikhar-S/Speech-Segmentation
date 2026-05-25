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
from datasets import concatenate_datasets


def merge_librispeech_train_clean_460(ddict: Any) -> Any:
    """Expose ``train.clean.100 + train.clean.360`` (460h) as ``train``."""
    out = {k: v for k, v in ddict.items()}
    out["train"] = concatenate_datasets(
        [ddict["train.clean.100"], ddict["train.clean.360"]]
    )
    return out


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


def split_buckeye_val_as_tune(ddict: Any) -> Any:
    """Use Buckeye's pre-existing ``val`` split as ``tune``.

    Buckeye ships disjoint train/val/test speaker sets (32/4/4 speakers).
    The ``val`` split (4 speakers, ~997 utts) is already held out from
    both train and test, so we simply expose it under the name ``tune``
    that the phonvec pipeline expects. ``test`` is left untouched.
    """
    new_dd = {k: v for k, v in ddict.items()}
    new_dd["tune"] = ddict["val"]
    return new_dd


def split_torgo_test_speaker_independent(ddict: Any) -> Any:
    """Speaker-independent tune carve from Torgo ``test``.

    Torgo has 14 speakers: 7 dysarthric (id starts with ``M`` or ``F``,
    no ``C``) and 7 typical-controls (id starts with ``MC`` or ``FC``).
    Picks 2 dysarthric + 2 control speakers at random (seed=42); their
    utterances form ``tune`` and the remaining 10 speakers form ``test``.
    Tune and test speaker sets are disjoint and the tune set is balanced
    across both speaking conditions.
    """
    test = ddict["test"]
    speakers = list(test["speaker_id"])
    unique = sorted(set(speakers))
    control = [s for s in unique if s.startswith(("MC", "FC"))]
    dysarthric = [s for s in unique if s not in control]
    rng = np.random.default_rng(42)
    n_each = 2
    tune_spk_set = set()
    for group in (dysarthric, control):
        if not group:
            continue
        idx = rng.choice(
            len(group), size=min(n_each, len(group)), replace=False
        )
        tune_spk_set.update(group[int(i)] for i in idx)
    tune_idx = [i for i, s in enumerate(speakers) if s in tune_spk_set]
    test_idx = [i for i, s in enumerate(speakers) if s not in tune_spk_set]

    new_dd = {k: v for k, v in ddict.items()}
    new_dd["tune"] = test.select(tune_idx)
    new_dd["test"] = test.select(test_idx)
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


def split_ssnce_test_speaker_independent(ddict: Any) -> Any:
    """Speaker-independent tune carve from SSNCE ``test``.

    SSNCE has 30 speakers (262 utts each). Speaker IDs follow the Torgo
    convention: ``MC*/FC*`` are typical controls (5 male + 5 female),
    everyone else is dysarthric (13 male + 7 female).

    Picks 4 dysarthric (2 male + 2 female) and 2 control speakers (1 male
    + 1 female) at random (seed=42). Their utterances form ``tune``; the
    remaining 24 speakers' utterances stay in ``test``. Tune and test
    speaker sets are disjoint and balanced across both gender and
    condition.
    """
    test = ddict["test"]
    speakers = list(test["speaker_id"])
    unique = sorted(set(speakers))
    control = [s for s in unique if s.startswith(("MC", "FC"))]
    dysarthric = [s for s in unique if s not in control]

    def _pick(group, n, rng):
        if not group or n <= 0:
            return []
        idx = rng.choice(len(group), size=min(n, len(group)), replace=False)
        return [group[int(i)] for i in idx]

    rng = np.random.default_rng(42)
    tune_spk = set()
    tune_spk.update(_pick([s for s in dysarthric if s.startswith("M")], 2, rng))
    tune_spk.update(_pick([s for s in dysarthric if s.startswith("F")], 2, rng))
    tune_spk.update(_pick([s for s in control if s.startswith("MC")], 1, rng))
    tune_spk.update(_pick([s for s in control if s.startswith("FC")], 1, rng))

    tune_idx = [i for i, s in enumerate(speakers) if s in tune_spk]
    test_idx = [i for i, s in enumerate(speakers) if s not in tune_spk]

    new_dd = {k: v for k, v in ddict.items()}
    new_dd["tune"] = test.select(tune_idx)
    new_dd["test"] = test.select(test_idx)
    return new_dd


# Registry: HuggingFace repo id -> default DatasetDict-level split transform.
HF_REPO_SPLIT_TRANSFORMS: Dict[str, Any] = {
    "changelinglab/timit-segment": False,
    "changelinglab/voxangeles-segment": False,
    "changelinglab/buckeye-segment": False,
    "changelinglab/gtimit-l2simple-segment": False,
    "changelinglab/gtimit-l2tbnk-segment": False,
    "changelinglab/gtimit-l1simple-segment": False,
    "changelinglab/gtimit-l1tbnk-segment": False,
    "changelinglab/gtimit-tha-segment": False,
    "changelinglab/torgo-segment": False,
    "changelinglab/ssnce-segment": False,
    "exp/downloads/librispeech-mfa-seg": merge_librispeech_train_clean_460,
}
