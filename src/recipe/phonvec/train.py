"""Phonvec training + tuning entrypoint.
# NOTE(shikhar): The logic concering tune split for individual datasets 
# is deprecated now since we moved to a single training phase.
Process
-------
1. Fit: PhonologicalVectors + cross-position regressors on a ``train``
   split (one row per GT phone, center-frame encoder embedding).
2. Per-signal tuning: correlation analysis on the ``tune`` split to
   find optimal kwargs + shift per signal.
3. Combined tuning: grid-search over combined hparams (drop_k,
   norm_method, prominence, min_correlation, …) on the same
   ``tune`` split.
4. Save: a self-contained artifact (vectors + regressors + best
   hparams + net spec) that ``src.model.phonvec.inference`` can load.

Stage 1 is run optionally.  If you pass
``phonvec_tune.fit_artifact=<path>`` the code loads a pre-fit
artifact and only runs the tuning stages.  Both per-signal and
combined tuning always re-run on the current tune split, so retuning
on a different dataset adapts all parameters.

Tune split source
-----------------
Comes from the tune_dataset() of the datamodule at
phonvec_tune.tune_data in config.  The logic to define tune and test
split for each dataset (hf_repo) is defined in
``HF_REPO_SPLIT_TRANSFORMS``
(src/data/segmentation/dataset_splitting_transforms.py)

Usage (GPU)
-----------
Fit and tune on TIMIT::

    python -m src.recipe.phonvec.train experiment=train/phonvec_tune

Reuse TIMIT fit, retune on VoxAngeles tune slice::

    python -m src.recipe.phonvec.train \\
        experiment=train/phonvec_tune_voxangeles \\
        phonvec_tune.fit_artifact=path/to/phonvec_artifact.pt
"""

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

# Hydra boilerplate. Sets PROJECT_ROOT and adds repo root to PYTHONPATH so
# Hydra interpolations in `configs/paths/default.yaml` resolve.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.recipe.phonvec.utils import (
    collect_eval_inputs,
    fit_or_load_segmenter,
    instantiate_fit_dataset,
    instantiate_tune_dataset,
    log,
    make_cache_id,
    run_grid_search,
    run_signal_tuning,
    save_phonvec_artifact,
)


@hydra.main(
    version_base="1.3", config_path="../../../configs", config_name="main"
)
def main(cfg: DictConfig) -> None:
    pt = cfg.phonvec_tune
    device = pt.get("device", "cuda")
    frame_shift = int(pt.frame_shift)
    sr = int(pt.sr)
    mel_frame_shift_ms = int(pt.mel_frame_shift_ms)
    skip_fit = pt.get("fit_artifact") is not None
    strip_outer = bool(pt.get("strip_outer_silences", True))

    fit_ds = instantiate_fit_dataset(cfg, skip=skip_fit)
    tune_ds = instantiate_tune_dataset(pt)

    log.info(f"Instantiating net <{pt.net._target_}>")
    net = hydra.utils.instantiate(pt.net).to(device).eval()

    fit_cache_id = make_cache_id(pt.net, cfg.data.hf_repo, "train")
    tune_cache_id = make_cache_id(
        pt.net,
        pt.tune_data.hf_repo,
        "tune",
    )

    base_seg, saved_net_spec = fit_or_load_segmenter(
        pt,
        fit_ds,
        net,
        device,
        frame_shift,
        sr,
        mel_frame_shift_ms,
        fit_cache_id=fit_cache_id,
    )

    log.info("Caching encoder features for tune subset...")
    tune_cache = collect_eval_inputs(
        tune_cache_id,
        tune_ds,
        net,
        device,
        frame_shift,
        sr,
    )

    # --- Per-signal correlation analysis ---
    signal_grid = OmegaConf.to_container(
        pt.signal_grid,
        resolve=True,
    )
    shift_values = list(
        OmegaConf.to_container(pt.shift_values, resolve=True),
    )
    signal_configs = run_signal_tuning(
        base_seg,
        tune_cache,
        signal_grid,
        shift_values,
        frame_shift,
        sr,
        strip_outer_silences=strip_outer,
    )

    # --- Combined grid search ---
    best_seg, results = run_grid_search(
        base_seg,
        tune_cache,
        signal_configs,
        pt,
        frame_shift,
        sr,
    )
    save_phonvec_artifact(
        best_seg,
        saved_net_spec,
        pt,
        results,
        frame_shift,
        sr,
        mel_frame_shift_ms,
    )


if __name__ == "__main__":
    main()
