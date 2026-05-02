"""Quick inference with omniASR CTC + LLM 7B on a single wav file."""

import sys

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


AUDIO = "/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/download/vaani_hindibelt/WAVS/Uttarakhand_TehriGarhwal_441.wav"
LANG = "gbm_Deva"
MODELS = ("omniASR_CTC_7B", "omniASR_LLM_7B")


def main() -> None:
    for model_card in MODELS:
        print(f"\n=== {model_card} ===", flush=True)
        pipeline = ASRInferencePipeline(model_card=model_card)
        out = pipeline.transcribe([AUDIO], lang=[LANG], batch_size=1)
        print(f"[{model_card}] {out[0]}", flush=True)
        del pipeline
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
