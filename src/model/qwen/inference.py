"""vLLM Inference wrapper.
Usage:
    python -m src.model.qwen.inference
"""

import base64
import io
import json
import string
import unicodedata
from pathlib import Path
from typing import Any, Optional, Union, List, Dict

import httpx
import numpy as np
import soundfile as sf
from openai import OpenAI
import torch


class VllmInference:
    """
    Compatible with:
    - Qwen/Qwen3-Omni-30B-A3B-Instruct
    - Qwen/Qwen3-Omni-30B-A3B-Thinking
    """

    def __init__(
        self,
        client_config: dict[str, Any],
        prompt_config: dict[str, Any],
        clean_response: bool = False,
        output_key: Optional[str] = None,
        device: Optional[str] = "cpu",
        cache_path: Optional[str | Path] = None,
        resume: bool = True,
        cache_key_field: str = "utt_id",
        error_log_path: Optional[str | Path] = None,
        timeout: float = 600.0,
        save_thoughts: bool = False,
    ) -> None:
        """
        Args:
            client_config: keys: base_url, model_name, api_key, max_tokens, temperature
            prompt_config: keys: system_prompt, user_prompt_template
            clean_response: Normalize text (remove punct, lower, etc.)
            output_key: Key to extract if output is JSON
        """
        # Client Config
        self.base_url = client_config.get("base_url", "http://localhost:8000/v1")
        self.model_name = client_config.get(
            "model_name", "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        )
        self.api_key = client_config.get("api_key", "EMPTY")

        # Generation Parameters
        self.max_tokens = client_config.get("max_tokens", 512)
        self.temperature = client_config.get("temperature", 0.0)
        self.top_p = client_config.get("top_p", 0.95)
        self.save_thoughts = save_thoughts
        # Initialize Client with Timeout
        http_client = httpx.Client(timeout=timeout)
        self.client = OpenAI(
            base_url=self.base_url, api_key=self.api_key, http_client=http_client
        )

        # Prompt Config
        self.system_prompt = prompt_config.get("system_prompt", "")
        self.user_prompt_template = prompt_config.get(
            "user_prompt_template", "{prompt}"
        )
        self.cache_key_field = cache_key_field
        assert self.cache_key_field, "cache_key_field must be non-empty."

        # Processing Config
        self.clean_response = clean_response
        self.output_key = output_key
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.error_log_path = Path(error_log_path) if error_log_path else None
        self.resume = resume

        # Load Cache
        self._cache: Dict[str, Any] = {}
        if self.cache_path and self.resume:
            self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            # load all available cache to check if any
            # other worker already completed the item?
            prefix = self.cache_path.name
            all_paths = sorted(self.cache_path.parent.glob(f"{prefix}.*.*.jsonl"))
            for path in all_paths:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            k = rec.get("key")
                            pred = rec.get("pred")
                            if k and pred:
                                self._cache[str(k)] = pred
                        except Exception:
                            continue
        except Exception:
            pass

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            print(f"Failed to write to {path}: {e}", flush=True)

    @staticmethod
    def _numpy_to_base64_wav(audio: np.ndarray, sample_rate: int) -> str:
        # Handle shape (Channels, Samples) vs (Samples,)
        if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _extract_thought_and_response(self, content: str) -> tuple[str, str]:
        if "<think>" not in content:
            return "", content.strip()
        pre, _, rest = content.partition("<think>")
        thought, _, post = rest.partition("</think>")
        # 'pre' preserves text before <think>,
        # 'post' handles text after or is empty if cutoff due to token budget
        return thought.strip(), (pre + post).strip()

    def __call__(
        self, speech: Union[np.ndarray, str, Path], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        uttid = kwargs[self.cache_key_field]
        if self.cache_path and self.resume and uttid in self._cache:
            return self._cache[uttid]

        try:
            prompt = self.user_prompt_template.format(**kwargs)
        except KeyError:
            raise KeyError(
                f"Warning: Missing keys in provided arguments. Provided keys: {list(kwargs.keys())},"
                f" whereas user_prompt_template requires keys used in: {self.user_prompt_template}"
            )

        try:
            if isinstance(speech, torch.Tensor):
                speech = speech.cpu().numpy()
            assert isinstance(speech, np.ndarray), "Speech input must be a numpy array."
            sr = kwargs.get("sample_rate", 16000)
            b64_audio = self._numpy_to_base64_wav(speech, sr)
            data_uri = f"data:audio/wav;base64,{b64_audio}"
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            # Qwen-Audio vLLM specific format: Audio before text
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            )

            # 3. API Call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            raw_content = response.choices[0].message.content

        except Exception as e:
            err_record = [
                {
                    "processed_transcript": "",
                    "predicted_transcript": "",
                    "raw_model_response": "",
                    "error": {"type": type(e).__name__, "message": str(e)},
                }
            ]
            if self.error_log_path:
                self._append_jsonl(
                    self.error_log_path,
                    {"key": uttid, "error": err_record[0]["error"]},
                )
            return err_record

        thought, clean_text = self._extract_thought_and_response(raw_content)

        # JSON Extraction (if model outputs JSON inside/outside thinking)
        predicted_transcript = clean_text
        if self.output_key:
            predicted_transcript = self._parse_json_response(
                clean_text, self.output_key
            )

        # Cleaning (for WER/CER calculation)
        processed_transcript = predicted_transcript
        if self.clean_response:
            processed_transcript = self._clean_response(predicted_transcript)

        pred = [
            {
                "processed_transcript": processed_transcript,
                "predicted_transcript": predicted_transcript,
            }
        ]
        # This prevents json blow up
        if thought:
            if self.save_thoughts:
                pred[0]["thought"] = thought
                pred[0]["raw_model_response"] = raw_content
        else:
            pred[0]["raw_model_response"] = raw_content

        # 5. Write Cache
        if self.cache_path and uttid:
            self._cache[uttid] = pred
            self._append_jsonl(self.cache_path, {"key": uttid, "pred": pred})

        return pred

    @staticmethod
    def _parse_json_response(response: str, key: str) -> str:
        try:
            # Attempt to find JSON block if surrounded by text
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                response = response[start:end]

            parsed = json.loads(response)
            if isinstance(parsed, dict) and key in parsed:
                return str(parsed[key])
            return response
        except Exception:
            return response

    @staticmethod
    def _clean_response(text: str) -> str:
        text = "".join(text.split())
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = unicodedata.normalize("NFD", text)
        return text.strip()


if __name__ == "__main__":
    PORT = 43019
    client_cfg = {
        "base_url": f"http://localhost:{PORT}/v1",
        # "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "model_name": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "max_tokens": 6144,
    }
    import yaml

    prompt_path = "/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/configs/experiment/inference/transcribe_qwen.yaml"
    with open(prompt_path, "r") as f:
        prompt_cfg = yaml.safe_load(f)["inference"]["inference_runner"]["prompt_config"]
    inf = VllmInference(
        client_config=client_cfg, prompt_config=prompt_cfg, clean_response=True
    )
    # 10 acc
    speechpath = "/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/download/speechocean762/WAVE/SPEAKER2892/028920128.WAV"
    # arabic
    speechpath = "/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/download/cmu_l2arctic/resampled_16000Hz/l2arctic/ABA/wav/arctic_a0586.wav"
    import soundfile as sf

    speech, sr = sf.read(speechpath)
    # speech = np.random.randn(16000 * 5)  # 5 seconds of dummy audio
    result = inf(speech=speech, prompt="Transcribe this audio.", utt_id="test_utt")
    print(result)
