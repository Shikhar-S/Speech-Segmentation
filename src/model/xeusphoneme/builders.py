from pathlib import Path
from typing import Optional
import argparse
import yaml
import json
import torch

from src.model.powsm.specaug import SpecAug
from src.model.powsm.e_branchformer import EBranchformerEncoder
from src.model.xeusphoneme.cnn_frontend import CNNFrontend as Wav2VecCNN
from src.model.xeusphoneme.linear_layer import LinearProjection
from src.core.utils import download_hf_snapshot
from src.model.xeusphoneme.xeuspr_model import XeusPRModel
from src.model.xeusphoneme.xeuspr_inference import XeusPRInference
from src.model.powsm.ctc import CTC
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


def build_xeus_pr(
    config_file: str, checkpoint: Optional[str] = None, vocab_file: Optional[str] = None
) -> XeusPRModel:
    """Build Xeus PR model from config and optional checkpoint.

    Args:
        config_file: Path to config yaml file
        checkpoint: Path to model checkpoint (pretrained or fully trained)
        vocab_file: Path to vocabulary file. If None, use vocab in config.

    Returns:
        XeusPRModel
    """
    with open(config_file, "r", encoding="utf-8") as f:
        args = argparse.Namespace(**yaml.safe_load(f))
    if vocab_file is not None:
        with open(vocab_file) as f:
            tok2id = json.load(f)
            id2tok = {v: k for k, v in tok2id.items()}
            token_list = [id2tok[i] for i in range(len(id2tok))]
    elif isinstance(args.token_list, str):
        with open(args.token_list, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]
    else:
        token_list = list(args.token_list)
    vocab_size = len(token_list)
    log.info(f"Vocabulary size: {vocab_size}")

    assert (
        getattr(args, "frontend") == "wav2vec_cnn"
    ), "Config must specify wav2vec_cnn frontend"
    frontend = Wav2VecCNN(**args.frontend_conf)
    input_size = frontend.output_size()

    specaug = None
    if hasattr(args, "specaug") and args.specaug == "specaug":
        specaug = SpecAug(**args.specaug_conf)

    normalize = None
    assert (
        getattr(args, "preencoder") == "linear"
    ), "Config must specify linear preencoder"
    preencoder = LinearProjection(input_size=input_size, **args.preencoder_conf)
    input_size = preencoder.output_size()
    assert (
        args.encoder == "e_branchformer"
    ), f"Only e_branchformer supported, got {args.encoder}"
    encoder = EBranchformerEncoder(input_size=input_size, **args.encoder_conf)

    # Build CTC
    ctc = CTC(
        odim=vocab_size,
        encoder_output_size=encoder.output_size(),
        **getattr(args, "ctc_conf", {}),
    )

    # Build model
    model = XeusPRModel(
        encoder=encoder,
        ctc=ctc,
        token_list=token_list,
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        preencoder=preencoder,
        ignore_id=getattr(args, "ignore_id", -1),
        sym_blank=getattr(args, "sym_blank", "<blank>"),
    )

    if checkpoint:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            # convert to standard xeus style checkpoint
            state_dict = state_dict["state_dict"]  # for finetuned lightning checkpoints
            state_dict = {
                k.replace("net.", ""): v
                for k, v in state_dict.items()
                if k.startswith("net.")
            }
        load_info = model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded checkpoint: {checkpoint} with load info: {load_info}")

    model.training_args = args
    return model


def build_xeus_pr_from_hf(
    *,
    work_dir: str,
    hf_repo: Optional[str] = None,
    force: bool = False,
    config_file: Optional[str] = None,
    checkpoint: Optional[str] = None,
    vocab_file: Optional[str] = None,
) -> XeusPRModel:
    """Build Xeus PR model from local files or HuggingFace repo.

    Args:
        work_dir: Directory to store downloaded files from HF repo
        hf_repo: HuggingFace repo name (e.g., "username/xeus-pr")
            If None, load from local files only
        force: Whether to force re-download from HF repo
        config_file: Path to config file. If None, use default path in work_dir.
            Takes precedence over hf_repo download.
        checkpoint: Path to checkpoint file. If None, use default path in work_dir.
            Takes precedence over hf_repo download.
        vocab_file: Path to vocabulary file. If None, use path in config.

    Returns:
        XeusPRModel
    """
    # Default relative paths in HF repo
    REL_CONFIG = "model/config.yaml"
    REL_CKPT = "model/xeus_checkpoint_new.pth"

    # Download from HF if repo specified
    if hf_repo:
        log.info(f"Downloading snapshot from HuggingFace: {hf_repo}")
        download_hf_snapshot(
            repo_id=hf_repo,
            force_download=force,
            work_dir=work_dir,
        )

    # Resolve file paths
    root = Path(work_dir)
    cfg = config_file or str(root / REL_CONFIG)
    ckpt = checkpoint or str(root / REL_CKPT)

    # Verify files exist
    assert Path(cfg).exists(), f"Config file not found: {cfg}"
    assert Path(ckpt).exists(), f"Checkpoint file not found: {ckpt}"

    log.info(f"Building model from config: {cfg}")
    log.info(f"Loading checkpoint: {ckpt}")

    return build_xeus_pr(config_file=cfg, checkpoint=ckpt, vocab_file=vocab_file)


def build_xeus_pr_inference(
    work_dir: str,
    checkpoint: str,
    vocab_file: str,
    device,
    config_file: Optional[str] = None,
    hf_repo: Optional[str] = None,
    force_download: bool = False,
    dtype: str = "float32",
) -> XeusPRInference:
    model = build_xeus_pr_from_hf(
        work_dir=work_dir,
        hf_repo=hf_repo,
        force=force_download,
        config_file=config_file,
        checkpoint=checkpoint,
        vocab_file=vocab_file,
    )
    inference_obj = XeusPRInference(model, device=device, dtype=dtype)
    return inference_obj
