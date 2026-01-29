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


def build_panphon_distance_matrix(vocab: list[str]) -> torch.Tensor:
    """Build panphon distance matrix for articulatory CTC.
    Args:
        vocab: List where vocab[i] is the phone string for token id i
    Returns:
        Distance matrix of shape (V, V) with values in [0, 1]
    """
    from panphon.distance import Distance

    dst = Distance()
    V = len(vocab)
    special = {"<blank>", "<sos>", "<eos>", "<unk>", "<pad>"}

    dist_matrix = torch.zeros((V, V), dtype=torch.float32)

    for i in range(V):
        for j in range(i + 1, V):
            if vocab[i] in special or vocab[j] in special:
                dist = float("inf")
            else:
                try:
                    dist = dst.feature_edit_distance(vocab[i], vocab[j])
                except Exception:
                    log.warning(
                        f'!! Distance between "{vocab[i]}" and "{vocab[j]}" failed, setting to inf !!'
                    )
                    dist = float("inf")
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # Replace inf with 2x max finite distance, then normalize to [0, 1]
    finite = dist_matrix[torch.isfinite(dist_matrix)]
    max_dist = finite.max().item() if finite.numel() > 0 else 1.0
    dist_matrix = torch.where(
        torch.isfinite(dist_matrix), dist_matrix, torch.tensor(max_dist * 2.0)
    )
    dist_matrix = dist_matrix / dist_matrix.max()
    dist_matrix.fill_diagonal_(0.0)
    log.info(f"Built panphon distance matrix with shape {dist_matrix.shape}")
    return dist_matrix


def build_xeus_pr(
    config_file: str,
    checkpoint: Optional[str] = None,
    vocab_file: Optional[str] = None,
    ctc_config: Optional[dict] = None,
) -> XeusPRModel:
    """Build Xeus PR model from config and optional checkpoint.

    Args:
        config_file: Path to config yaml file
        checkpoint: Path to model checkpoint (pretrained or fully trained)
        vocab_file: Path to vocabulary file. If None, use vocab in config.
        ctc_config: Optional dict of CTC config

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

    ctc_config = ctc_config or getattr(args, "ctc_conf", {})
    if ctc_config.get("ctc_type", "builtin") == "articulatory_ctc":
        dist_matrix = build_panphon_distance_matrix(token_list)
        ctc_config["artctc_dist"] = dist_matrix
    # Build CTC
    ctc = CTC(
        odim=vocab_size,
        encoder_output_size=encoder.output_size(),
        **ctc_config,
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
        freeze_frontend=checkpoint is not None,
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
    ctc_config: Optional[dict] = None,
    load_ckpt: bool = True,
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
        ctc_config: Optional dict of CTC config
        load_ckpt: Whether to load checkpoint weights
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
    if not load_ckpt:
        ckpt = None
    else:
        assert Path(ckpt).exists(), f"Checkpoint file not found: {ckpt}"

    log.info(f"Building model from config: {cfg}")
    log.info(f"Loading checkpoint: {ckpt}")

    return build_xeus_pr(
        config_file=cfg, checkpoint=ckpt, vocab_file=vocab_file, ctc_config=ctc_config
    )


def build_xeus_pr_inference(
    work_dir: str,
    checkpoint: str,
    vocab_file: str,
    device,
    config_file: Optional[str] = None,
    hf_repo: Optional[str] = None,
    force_download: bool = False,
    dtype: str = "float32",
    ctc_config: Optional[dict] = None,
) -> XeusPRInference:
    model = build_xeus_pr_from_hf(
        work_dir=work_dir,
        hf_repo=hf_repo,
        force=force_download,
        config_file=config_file,
        checkpoint=checkpoint,
        vocab_file=vocab_file,
        ctc_config=ctc_config,
    )
    inference_obj = XeusPRInference(model, device=device, dtype=dtype)
    return inference_obj


if __name__ == "__main__":
    # python -m src.model.xeusphoneme.builders
    import torch
    import numpy as np

    vocab_path = "src/model/xeusphoneme/resources/ipa_vocab.json"
    V = json.load(open(vocab_path))
    revV = {v: k for k, v in V.items()}
    print("Loaded vocab of size:", len(V))
    dist_matrix = build_panphon_distance_matrix(vocab=list(V.keys()))
    print("Distance matrix shape:", dist_matrix.shape)
    TOPK = 10
    BETA = 80.0
    THRESH = 100
    BLANK_ID = 0
    mxlen = 0
    for sym in V:
        p = V[sym]
        drow = torch.as_tensor(dist_matrix[p], dtype=torch.float32).clone()
        drow[BLANK_ID] = float("inf")
        k = min(TOPK, len(V) - 1)
        cand = torch.topk(drow, k=k, largest=False).indices
        if not (cand == p).any().item():
            cand = torch.cat([cand[:-1], torch.tensor([p], dtype=cand.dtype)])
        logits = -BETA * drow[cand]  # shape (k,)
        scores = torch.log_softmax(logits, dim=0)  # log-probs
        keep = drow[cand] < THRESH
        cand_keep = cand[keep].tolist()
        scores_keep = scores[keep].tolist()
        mxlen = max(mxlen, len(cand_keep))
        if True or sym == "bʰ":
            pairs = sorted(zip(cand_keep, scores_keep), key=lambda x: x[0])
            print(
                f"Neighbors shown (dist<{THRESH}) for {sym}:",
                sorted(
                    [(revV[i], s) for i, s in pairs], key=lambda x: x[1], reverse=True
                ),
                sorted([drow[i].item() for i in cand_keep]),
            )
    print("Max neighbors with dist < 0.02 (shown):", mxlen)
