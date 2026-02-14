from pathlib import Path
from typing import Dict, Optional, Tuple
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
from src.model.xeusphoneme.resources.phonetic_substitutions import (
    ENGLISH_PHONEME_SUBSTITUTIONS,
    get_substitutions,
)

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


def build_diacritic_distance_matrix(vocab: list[str]) -> torch.Tensor:
    """
    Distance matrix where phones in the same "base-variant set" have distance 0,
    and all others have distance 1. Handles affricates like 'd͡ʒ' / 't͡ɕʰ' as a single unit.
    Using this effectively reduces vocabulary size.
    """

    def base_key(s: str) -> str:
        if not s:
            return ""
        # affricate/connected: base is first 3 codepoints, e.g. "t͡ɕ"
        return s[:3] if len(s) >= 3 and s[1] == "͡" else s[0]

    # build base index: base_key -> list of symbol strings
    base_idx: dict[str, list[str]] = {}
    for s in vocab:
        k = base_key(s)
        if k:
            base_idx.setdefault(k, []).append(s)

    def same_base_variants(sym: str) -> list[str]:
        if not sym:
            return []
        if len(sym) >= 3 and sym[1] == "͡":
            base = sym[:3]  # e.g. "t͡ɕ"
            # only those that truly share the connected base (diacritics may follow)
            return [s for s in base_idx.get(base, []) if s.startswith(base)]
        return base_idx.get(sym[0], [])

    V = len(vocab)
    dist_matrix = torch.ones((V, V), dtype=torch.float32)
    dist_matrix.fill_diagonal_(0.0)

    # string -> id (assumes vocab[i] is token i)
    sid = {s: i for i, s in enumerate(vocab)}

    # set dist_matrix=0 within each base-variant set
    for s in vocab:
        ids = [sid[t] for t in same_base_variants(s) if t in sid]
        if len(ids) <= 1:
            continue
        idx = torch.tensor(ids, dtype=torch.long)
        dist_matrix[idx[:, None], idx[None, :]] = 0.0

    return dist_matrix


def build_manual_distance_matrix(vocab: list[str]) -> torch.Tensor:
    """
    Distance matrix derived from ENGLISH_PHONEME_SUBSTITUTIONS / get_substitutions:

    For each source symbol x (treated as an "English phoneme" key), define its neighbor set
    as the UNION over all languages of get_substitutions(lang, x), intersected with vocab.

    Distance is:
      - 0 if y is in that union-substitution set for x (and symmetric closure is applied)
      - 1 otherwise

    Notes:
      - We also include x itself via get_substitutions' behavior.
      - We symmetrize to make dist[i,j]==dist[j,i].
    """
    V = len(vocab)
    sid = {s: i for i, s in enumerate(vocab)}
    langs = list(ENGLISH_PHONEME_SUBSTITUTIONS.keys())

    dist_matrix = torch.ones((V, V), dtype=torch.float32)
    dist_matrix.fill_diagonal_(0.0)

    # Build directed edges x -> y if y is a substitution of x in ANY language (union).
    edges = [set() for _ in range(V)]
    for x in vocab:
        i = sid[x]
        union_syms = set()
        for lang in langs:
            union_syms.update(get_substitutions(lang, x))
        edges[i] = {sid[y] for y in union_syms if y in sid}

    # Symmetric closure: i~j if i->j or j->i
    for i in range(V):
        for j in edges[i]:
            dist_matrix[i, j] = 0.0
            dist_matrix[j, i] = 0.0

    return dist_matrix


def matrix_to_neighbor_lists(
    dist: torch.Tensor,
    topk: int,
    blank_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert (V, V) distance matrix to compact neighbor lists: at most topk-1 others
    per phone (self excluded, always implied). Blank is never a neighbor.

    Returns:
        neighbor_ids: (V, topk - 1) int64, -1 for unused slots
        neighbor_dists: (V, topk - 1) float32
    """
    V = dist.size(0)
    k_others = max(0, topk - 1)
    neighbor_ids = torch.full((V, k_others), -1, dtype=torch.long)
    neighbor_dists = torch.zeros((V, k_others), dtype=torch.float32)
    if k_others == 0:
        return neighbor_ids, neighbor_dists
    dist = dist.clone()
    dist[:, blank_id] = float("inf")
    k_actual = min(topk, V)
    for p in range(V):
        idx = torch.topk(dist[p], k=k_actual, largest=False).indices
        others = idx[idx != p][:k_others]
        n = others.size(0)
        neighbor_ids[p, :n] = others
        neighbor_dists[p, :n] = dist[p, others]
    return neighbor_ids, neighbor_dists


def build_language_specific_distance_matrix(vocab: list[str], lang_code: str) -> torch.Tensor:
    """
    Distance matrix for one language: 0 between phoneme pairs that are
    substitutions of each other for that language (from get_substitutions),
    1 otherwise. Diagonal 0, symmetric.
    """
    V = len(vocab)
    sid = {s: i for i, s in enumerate(vocab)}
    dist_matrix = torch.ones((V, V), dtype=torch.float32)
    dist_matrix.fill_diagonal_(0.0)
    edges = [set() for _ in range(V)]
    for x in vocab:
        i = sid[x]
        subs = get_substitutions(lang_code, x)
        edges[i] = {sid[y] for y in subs if y in sid}
    for i in range(V):
        for j in edges[i]:
            dist_matrix[i, j] = 0.0
            dist_matrix[j, i] = 0.0
    return dist_matrix


def build_all_language_distance_matrices(vocab: list[str]) -> Dict[str, torch.Tensor]:
    """Returns dict lang_code -> (V, V) distance matrix for each language in ENGLISH_PHONEME_SUBSTITUTIONS."""
    return {
        lang: build_language_specific_distance_matrix(vocab, lang)
        for lang in ENGLISH_PHONEME_SUBSTITUTIONS.keys()
    }


def build_xeus_pr(
    config_file: str,
    checkpoint: Optional[str] = None,
    vocab_file: Optional[str] = None,
    ctc_config: Optional[dict] = None,
    weighted_sum: bool = False,
) -> XeusPRModel:
    """Build Xeus PR model from config and optional checkpoint.

    Args:
        config_file: Path to config yaml file
        checkpoint: Path to model checkpoint (pretrained or fully trained)
        vocab_file: Path to vocabulary file. If None, use vocab in config.
        ctc_config: Optional dict of CTC config
        weighted_sum: Whether to use weighted sum of transformer layers

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
    topk = ctc_config.get("artctc_topk", 8)
    if ctc_config.get("ctc_type", "builtin") == "panphon_distance":
        dist_matrix = build_panphon_distance_matrix(token_list)
        nids, ndists = matrix_to_neighbor_lists(dist_matrix, topk=topk, blank_id=0)
        ctc_config["artctc_neighbors_by_lang"] = {"global": (nids, ndists)}
    elif ctc_config.get("ctc_type", "builtin") == "diacritic_distance":
        dist_matrix = build_diacritic_distance_matrix(token_list)
        nids, ndists = matrix_to_neighbor_lists(dist_matrix, topk=topk, blank_id=0)
        ctc_config["artctc_neighbors_by_lang"] = {"global": (nids, ndists)}
    elif ctc_config.get("ctc_type", "builtin") == "manual_distance":
        dist_matrix = build_manual_distance_matrix(token_list)
        nids, ndists = matrix_to_neighbor_lists(dist_matrix, topk=topk, blank_id=0)
        ctc_config["artctc_neighbors_by_lang"] = {"global": (nids, ndists)}
    elif ctc_config.get("ctc_type", "builtin") == "manual_distance_per_lang":
        dist_by_lang = build_all_language_distance_matrices(token_list)
        V = len(token_list)
        # Identity (self-only): no other neighbors; use (V, 1) with -1 to avoid allocating (V, topk-1)
        identity_ids = torch.full((V, 1), -1, dtype=torch.long)
        identity_dists = torch.zeros((V, 1), dtype=torch.float32)
        per_lang = {
            lang: matrix_to_neighbor_lists(dist_by_lang[lang], topk=topk, blank_id=0)
            for lang in dist_by_lang
        }
        ctc_config["artctc_neighbors_by_lang"] = {
            "global": (identity_ids, identity_dists),
            **per_lang,
        }
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
        weighted_sum=weighted_sum,
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
    weighted_sum: bool = False,
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
        weighted_sum: Whether to use weighted sum of transformer layers
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
        config_file=cfg,
        checkpoint=ckpt,
        vocab_file=vocab_file,
        ctc_config=ctc_config,
        weighted_sum=weighted_sum,
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
