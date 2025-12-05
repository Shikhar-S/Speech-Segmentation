from typing import Optional
import torch.nn as nn
from src.model.powsm.powsm_model import PowsmModel
from pathlib import Path
from src.core.utils import download_hf_snapshot

import yaml
import torch
import argparse

from src.model.powsm.ctc import CTC
from src.model.powsm.frontend import DefaultFrontend, GlobalMVN
from src.model.powsm.specaug import SpecAug
from src.model.powsm.e_branchformer import EBranchformerEncoder
from src.model.powsm.transformer_decoder import TransformerDecoder
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class PowsmVariablerateModel(PowsmModel):
    def __init__(self, points_by_frames, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points_by_frames_ = points_by_frames
        self.sampling_ratio = super().points_by_frames() / points_by_frames
        # >1 means upsample, <1 means downsample
        encoder_size = super().encoder_output_size()
        self.sampling_net = nn.Sequential(
            nn.Linear(encoder_size, int(encoder_size * self.sampling_ratio)),
            nn.LayerNorm(int(encoder_size * self.sampling_ratio)),
        )

    def points_by_frames(self):
        return self.points_by_frames_

    def encode(self, speech, speech_lengths):
        feat, featlen = super().encode(speech, speech_lengths)
        feat = self.sampling_net(feat)
        feat = feat.view(
            feat.size(0), -1, int(feat.size(2) / self.sampling_ratio)
        ).contiguous()
        featlen = (featlen.float() * self.sampling_ratio).long()
        return feat, featlen


def build_powsm_vr_from_files(
    config_file: str,
    model_file: str,
    stats_file: str,
) -> PowsmModel:
    with open(config_file, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args = argparse.Namespace(**args)
    args.normalize_conf["stats_file"] = stats_file  # pass absolute path to stats file

    if isinstance(args.token_list, str):
        with open(args.token_list, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]
        args.token_list = list(token_list)
    elif isinstance(args.token_list, (tuple, list)):
        token_list = list(args.token_list)
    else:
        raise RuntimeError("token_list must be str or list")

    vocab_size = len(token_list)
    log.info(f"Vocabulary size: {vocab_size}")

    # 1. frontend
    assert args.input_size is None, "Set frontend in the powsm config."
    assert args.frontend == "default", "Only default frontend is supported!"
    frontend = DefaultFrontend(**args.frontend_conf)
    input_size = frontend.output_size()

    # 2. Data augmentation for spectrogram
    assert args.specaug == "specaug", "Only SpecAug is supported!"
    specaug = SpecAug(**args.specaug_conf)

    # 3. Normalization layer
    assert args.normalize == "global_mvn", "Only GlobalMVN is supported!"
    normalize = GlobalMVN(**args.normalize_conf)

    # 4. Encoder
    assert args.encoder == "e_branchformer", "Only Branchformer is supported!"
    encoder = EBranchformerEncoder(input_size=input_size, **args.encoder_conf)
    encoder_output_size = encoder.output_size()

    # 5. Decoder
    assert args.decoder == "transformer", "Only Transformer decoder is supported!"
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        **args.decoder_conf,
    )

    # 6. CTC
    ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf)

    # 7. Build model
    model = PowsmVariablerateModel(
        points_by_frames=320,
        vocab_size=vocab_size,
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        token_list=token_list,
        **args.model_conf,
    )

    # 8. Load weights
    state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
    load_info = model.load_state_dict(state_dict, strict=False)
    log.info(f"Model loaded: {model_file} with info: {load_info}")
    model.training_args = args
    return model


def build_powsm_vr(
    *,
    work_dir: str,
    hf_repo: Optional[str] = "espnet/powsm",
    force: bool = False,
    config_file: Optional[str] = None,
    model_file: Optional[str] = None,
    stats_file: Optional[str] = None,
):
    """Build Powsm model from local files or huggingface repo.
    Args:
        work_dir: Directory to store downloaded files from hf repo.
        hf_repo: Huggingface repo name. If None, load from local files.
        force: Whether to force re-download from hf repo.
        config_file: Path to config file. If None, use default path in hf repo.
          Takes precedence over hf_repo.
        model_file: Path to model file. If None, use default path in hf repo.
          Takes precedence over hf_repo.
        stats_file: Path to stats file. If None, use default path in hf repo.
          Takes precedence over hf_repo.
    Returns: PowsmModel
    """
    # Relative paths from hf repo structure (espnet style)
    # TODO(shikhar): Convert to patterns and match patterns within downloaded files.
    REL_CONFIG = "exp/s2t_train_s2t_ebf_conv2d_size768_e9_d9_piecewise_lr5e-4_warmup60k_flashattn_raw_bpe40000/config.yaml"
    REL_CKPT = "exp/s2t_train_s2t_ebf_conv2d_size768_e9_d9_piecewise_lr5e-4_warmup60k_flashattn_raw_bpe40000/valid.acc.ave_5best.till45epoch.pth"
    REL_STATS = "exp/s2t_stats_raw_bpe40000/train/feats_stats.npz"

    if hf_repo:
        download_hf_snapshot(
            repo_id=hf_repo,
            force_download=force,
            work_dir=work_dir,
        )

    root = Path(work_dir)
    cfg = config_file or str(root / REL_CONFIG)
    mdl = model_file or str(root / REL_CKPT)
    stats = stats_file or str(root / REL_STATS)
    # assert files exist
    assert Path(cfg).exists(), f"Config file not found: {cfg}"
    assert Path(mdl).exists(), f"Model file not found: {mdl}"
    assert Path(stats).exists(), f"Stats file not found: {stats}"

    return build_powsm_vr_from_files(cfg, mdl, stats)
