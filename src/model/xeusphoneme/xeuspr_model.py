#!/usr/bin/env python3
"""Xeus Phoneme Recognition Model.
# -*- coding: utf-8 -*-

# Copyright 2025 William Chen. Adapted from ESPnet.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

Usage:
    python -m src.model.xeusphoneme.xeuspr_model \
        --work_dir /scratch/sbharad2/PhoneBench/exp/cache/xeus
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import argparse
import yaml
import json

import torch
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask

from src.model.powsm.ctc import CTC
from src.model.powsm.specaug import SpecAug
from src.model.powsm.e_branchformer import EBranchformerEncoder
from src.model.xeusphoneme.cnn_frontend import CNNFrontend as Wav2VecCNN
from src.model.xeusphoneme.linear_layer import LinearProjection
from src.core.utils import download_hf_snapshot
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class XeusPRModel(torch.nn.Module):
    """Encoder-only CTC model for phone recognition using Xeus pretrained weights."""

    def __init__(
        self,
        encoder: Any,
        ctc: CTC,
        token_list: Union[Tuple, list],
        frontend: Optional[Any] = None,
        specaug: Optional[Any] = None,
        normalize: Optional[Any] = None,
        preencoder: Optional[Any] = None,
        ignore_id: int = -1,
        sym_blank: str = "<blank>",
        **kwargs,
    ):
        super().__init__()
        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.ctc = ctc

        self.token_list = list(token_list)
        self.ignore_id = ignore_id
        self.blank_id = token_list.index(sym_blank) if sym_blank in token_list else 0

    def collect_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract features for stats collection."""
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        """Forward pass with CTC loss computation."""
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        )
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        loss_ctc, acc = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )
        stats = {
            "acc": acc,
        }
        loss, stats, weight = force_gatherable(
            (loss_ctc, stats, speech.shape[0]), loss_ctc.device
        )
        # loss = loss / weight  # normalize by batch size
        return {"loss": loss, "stats": stats, "weight": weight}

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features using frontend."""
        speech = speech[:, : speech_lengths.max()]
        return (
            self.frontend(speech, speech_lengths)
            if self.frontend
            else (speech, speech_lengths)
        )

    def _apply_preprocessing(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply frontend, specaug, normalize, and preencoder."""
        speech, speech_lengths = self._extract_feats(speech, speech_lengths)

        if self.specaug and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        if self.normalize:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        if self.preencoder:
            speech, speech_lengths = self.preencoder(speech, speech_lengths)

        return speech, speech_lengths

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode speech to frame-level representations."""
        speech, speech_lengths = self._apply_preprocessing(speech, speech_lengths)
        pad_masks = make_pad_mask(speech_lengths).to(speech.device)
        encoder_out, encoder_out_lens, _ = self.encoder(
            speech, speech_lengths, masks=pad_masks, return_all_hs=True
        )
        return encoder_out[0], encoder_out_lens

    def ctc_collapse_batch(self, x: torch.Tensor, max_length: int, pad: int = -1):
        B, T = x.shape
        # if T > max_length:
        #     x = x[:, :max_length]
        # elif T < max_length:
        #     pad_tensor = torch.full(
        #         (B, max_length - T), pad, device=x.device, dtype=x.dtype
        #     )
        #     x = torch.cat([x, pad_tensor], dim=1)
        # T = max_length
        blank = self.blank_id
        x_prev = torch.cat(
            [torch.full((B, 1), blank, device=x.device, dtype=x.dtype), x[:, :-1]],
            dim=1,
        )
        keep = (x != blank) & ((x_prev == blank) | (x != x_prev))
        pos = keep.long().cumsum(1) - 1
        lengths = keep.sum(1)
        out = torch.full((B, T), pad, device=x.device, dtype=x.dtype)
        # Compute batch indices and output positions for kept elements
        batch_idx = (
            torch.arange(B, device=x.device, dtype=torch.long).unsqueeze(1).expand_as(x)
        )
        output_pos = pos.clone()
        # Only use positions where keep is True
        batch_idx_keep = batch_idx[keep]
        output_pos_keep = output_pos[keep]
        # Flatten the output and set values at correct positions
        flat_out = out.view(-1)
        flat_idx = batch_idx_keep * T + output_pos_keep
        flat_out[flat_idx] = x[keep]
        out = flat_out.view(B, T)
        ##### Trim to max_length from ground truth lengths
        out = out[:, :max_length]
        lengths = torch.clamp(lengths, max=max_length)
        return out, lengths

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """Calculate CTC loss."""
        ys_pad = torch.where(ys_pad == -1, self.ignore_id, ys_pad)
        ys_pad = ys_pad[:, : ys_pad_lens.max()]
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        acc = None
        with torch.no_grad():
            ys_hat = self.ctc.ctc_lo(encoder_out).argmax(dim=-1)
            ys_hat = self.ctc_collapse_batch(
                ys_hat.detach(), max_length=ys_pad.shape[1], pad=self.ignore_id
            )[0]
            acc = (
                ys_hat.eq(ys_pad)
                .masked_select(~make_pad_mask(ys_pad_lens).to(ys_hat.device))
                .float()
                .mean()
                .item()
            )
        return loss_ctc, acc

    def ctc_logits(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get CTC logits for inference."""
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        return self.ctc.ctc_lo(encoder_out), encoder_out_lens

    def encoder_output_size(self) -> int:
        return self.encoder.output_size()

    def get_blank_id(self) -> int:
        return self.blank_id

    def get_trainable_parameters(self, freeze_xeus=False):
        trainable_params = []
        for n, p in self.named_parameters():
            if n.startswith("ctc"):
                trainable_params.append(p)
                continue
            if freeze_xeus:
                p.requires_grad = False
            else:
                trainable_params.append(p)
        return trainable_params


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="Working directory for model files",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=None,
        help="Path to config file (overrides work_dir download)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to checkpoint file (overrides work_dir download)",
    )

    args = parser.parse_args()

    model = build_xeus_pr_from_hf(
        work_dir=args.work_dir,
        hf_repo="espnet/xeus",
        force=False,
        config_file=args.config,
        checkpoint=args.checkpoint,
    )
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    dummy_input = torch.randn(2, 16000)
    dummy_lengths = torch.tensor([16000, 16000])
    encoder_out, encoder_out_lens = model.encode(dummy_input, dummy_lengths)
    print(f"Encoder output shape: {encoder_out.shape}")
