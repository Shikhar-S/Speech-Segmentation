# TODO(shikhar): Refactor to allow training
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2PhonemeInference:
    def __init__(self, device="cpu", **kwargs):
        self.device = device
        self.model_path = kwargs.get("model_path", "")
        assert (
            self.model_path != ""
        ), "model_path must be specified for Wav2Vec2PhonemeInference"
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        # Load model with appropriate dtype based on device
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_path, torch_dtype=torch.float32
        )
        self.dtype = torch.float32
        self.model.eval().to(device)

    def infer(self, input_batch):
        audio = input_batch["wav"].squeeze(0).numpy()
        inputs = self.processor(
            audio,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            input_values = inputs.input_values.to(self.device)
            input_values = input_values.to(self.dtype)
            if hasattr(inputs, "attention_mask"):
                attention_mask = inputs.attention_mask.to(self.device)
                logits = self.model(input_values, attention_mask=attention_mask).logits
            else:
                logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        recognized = "".join(transcription)
        return recognized
