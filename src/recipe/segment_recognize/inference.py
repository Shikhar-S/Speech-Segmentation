"""Inference wrapper for SegmentRecognizeModel.

Supports src/core/distributed_inference.py

Usage:
    python -m src.recipe.segment_recognize.inference
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from src.recipe.segment_recognize.model_module import (
    SegmentRecognizeModel,
)


class SegmentRecognizeInference:
    def __init__(
        self,
        model: SegmentRecognizeModel,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        speech: torch.Tensor,
        speech_length: Any,
        phones: torch.Tensor = None,
        phone_length: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Decode a single utterance across all registered heads."""
        sp = speech[: int(speech_length)].unsqueeze(0).to(self.device)
        sl = torch.as_tensor([int(speech_length)], device=self.device)
        batch: Dict[str, Any] = {"speech": sp, "speech_length": sl}
        
        if phones is not None and phone_length is not None:
            batch["phones"] = (
                phones[: int(phone_length)].unsqueeze(0).to(self.device)
            )
            batch["phone_length"] = torch.as_tensor(
                [int(phone_length)], device=self.device,
            )

        return self.model.predict_step(batch, 0)


def build_segment_recognize_inference(
    net: nn.Module,
    ckpt_path: str,
    device: str = "cuda",
) -> SegmentRecognizeInference:
    """Build an inference wrapper from a Lightning checkpoint. 
    Heads are loaded from ckpt_path."""
    model = SegmentRecognizeModel.load_from_checkpoint(
        ckpt_path,
        net=net,
        map_location="cpu",
        strict=True,
        weights_only=False,
    )
    return SegmentRecognizeInference(model=model, device=device)


if __name__ == "__main__":
    # python -m src.recipe.segment_recognize.inference
    from src.model.xeusphoneme.builders import build_xeus_pr_from_hf
    ckpt_path = "/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/vibe_timit_single/bce/checkpoints/last.ckpt"
    ckpt_path='/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/vibe_timit_single/ctc/checkpoints/last.ckpt'
    ckpt_path='/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/vibe_timit_single/fce/checkpoints/last.ckpt'
    
    net = build_xeus_pr_from_hf(
        work_dir=(
            "/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/cache/xeus"
        ),
        hf_repo="espnet/xeus",
        load_ckpt=False,
        vocab_file="src/model/xeusphoneme/resources/ipa_vocab.json",
        interctc_weight=0.3,
        interctc_layer_idx=[4, 8, 12],
        interctc_use_conditioning=True,
        ctc_weight=1.0,
    )
    inference = build_segment_recognize_inference(
        net=net, ckpt_path=ckpt_path, device='cpu',
    )
    
    speech = torch.randn(32000 * 5)
    speech_length = torch.tensor(32000 * 5)
    
    result = inference(speech=speech, speech_length=speech_length)
    print(result)