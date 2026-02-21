#!/usr/bin/env python3
"""Xeus Phoneme Recognition Model.
# -*- coding: utf-8 -*-

# Copyright 2025 William Chen. Adapted from ESPnet.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

Usage:
    python -m src.model.xeusphoneme.xeuspr_model \
        --work_dir /scratch/sbharad2/PhoneBench/exp/cache/xeus
"""
from typing import Any, Dict, Optional, Tuple, Union
import argparse

import torch
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet_import.nets.pytorch_backend.nets_utils import make_pad_mask

# from espnet_import.nets.e2e_asr_common import ErrorCalculator
from src.recipe.phone_recognition.error_calculator import ErrorCalculator

from src.model.powsm.ctc import CTC
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
        freeze_frontend: bool = True,
        weighted_sum: bool = False,
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
        sym_space = kwargs.get("sym_space", "<space>")
        self.freeze_frontend = freeze_frontend
        # self.error_calculator = ErrorCalculator(
        #     token_list, sym_space, sym_blank, report_cer=True, report_wer=False
        # )
        self.error_calculator = ErrorCalculator(
            token_list,
            blank_id=self.blank_id,
            sym_space=sym_space,
            ignore_id=ignore_id,
            log_phone_metrics=True,
        )

        self.weighted_sum = weighted_sum
        if self.weighted_sum:
            n_layers = encoder.num_blocks
            assert (
                n_layers is not None and n_layers > 0
            ), "Cannot infer number of encoder layers for weighted_sum"
            self.layer_weights = torch.nn.Parameter(torch.zeros(int(n_layers)))

    def collect_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract features for stats collection."""
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        loss_ctc, stats = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text, text_lengths, **kwargs
        )
        loss, stats, weight = force_gatherable(
            (loss_ctc, stats, speech.shape[0]), loss_ctc.device
        )
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
        if not self.weighted_sum:
            return encoder_out[0], encoder_out_lens

        hs_list = encoder_out[1]
        assert len(hs_list) == self.layer_weights.numel()
        w = torch.softmax(self.layer_weights, dim=0).to(
            hs_list[0].device, hs_list[0].dtype
        )
        hs = torch.stack(hs_list, dim=0)  # (L, B, T, D)
        return (w.view(-1, 1, 1, 1) * hs).sum(0), encoder_out_lens

    def ctc_collapse_batch(self, x: torch.Tensor, max_length: int, pad: int = -1):
        B, T = x.shape
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
        self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, **kwargs
    ):
        ys_pad = torch.where(ys_pad == -1, self.ignore_id, ys_pad)
        ys_pad = ys_pad[:, : ys_pad_lens.max()]
        loss_ctc = self.ctc(
            encoder_out,
            encoder_out_lens,
            ys_pad,
            ys_pad_lens,
            lang_sym=kwargs.get("lang_sym"),
            accent_sym=kwargs.get("accent_sym"),
        )
        stats = {}
        assert self.error_calculator is not None, "ErrorCalculator not initialized"
        if not self.training:  # err calc, slow?
            with torch.no_grad():
                ys_hat = self.ctc.argmax(encoder_out).data  # greedy-top1
                # stats["cer_ctc"] = self.error_calculator(
                #     ys_hat.cpu(), ys_pad.cpu(), is_ctc=True
                # )
                metrics = self.error_calculator(
                    ys_hat.cpu(), ys_pad.cpu(), ys_pad_lens.cpu()
                )
                for k, v in metrics.items():
                    stats[k + "_ctc"] = v
        return loss_ctc, stats

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

    def get_frontend(self):
        return self.frontend

    def get_trainable_parameters(self):
        trainable_params = {"head": [], "encoder": []}
        for n, p in self.named_parameters():
            if n.startswith("ctc"):
                trainable_params["head"].append(p)
            elif n.startswith("encoder"):
                trainable_params["encoder"].append(p)
            elif n.startswith("frontend"):
                if self.freeze_frontend:
                    p.requires_grad = False
                else:
                    trainable_params["encoder"].append(p)
            else:
                # freeze other parts:
                p.requires_grad = False
        return trainable_params


if __name__ == "__main__":
    # python -m src.model.xeusphoneme.xeuspr_model
    import argparse
    from src.model.xeusphoneme.builders import build_xeus_pr_from_hf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        type=str,
        default="exp/cache/xeus",
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
        weighted_sum=True,
    )
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    dummy_input = torch.randn(2, 16000)
    dummy_lengths = torch.tensor([16000, 16000])
    encoder_out, encoder_out_lens = model.encode(dummy_input, dummy_lengths)
    print(f"Encoder output shape: {encoder_out.shape}")
