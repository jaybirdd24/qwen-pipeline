"""Official non-streaming Qwen3-TTS voice cloning adapter."""

from __future__ import annotations

import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..config import PipelineConfig
from ..errors import EngineError
from ..models import QWEN_LANGUAGE_NAMES, VoiceReference

QWEN_AUDIO_TOKENS_PER_SECOND = 12.5


def _reached_generation_limit(
    sample_count: int,
    sample_rate: int,
    max_new_tokens: int | None,
) -> bool:
    """Detect output that consumed the token budget instead of producing an EOS token."""
    if max_new_tokens is None or sample_count <= 0 or sample_rate <= 0:
        return False
    duration = sample_count / sample_rate
    limit_duration = max(0.0, (max_new_tokens - 2) / QWEN_AUDIO_TOKENS_PER_SECOND)
    return duration >= limit_duration


class QwenTTSEngine:
    name = "qwen"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.model_name = config.model_name
        self.model_revision = config.model_revision
        self._prompts: dict[str, Any] = {}
        try:
            import torch
            from huggingface_hub import snapshot_download
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise EngineError(
                "Qwen dependencies are unavailable; install CUDA PyTorch and then install "
                "the project's `qwen` optional dependency"
            ) from exc
        self._torch = torch
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise EngineError("CUDA was requested but PyTorch cannot see a CUDA device")
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(config.dtype)
        if dtype is None:
            raise EngineError(f"Unsupported dtype: {config.dtype}")
        try:
            configured_path = Path(config.model_name).expanduser()
            model_source = (
                str(configured_path.resolve())
                if configured_path.exists()
                else snapshot_download(
                    repo_id=config.model_name,
                    revision=config.model_revision,
                )
            )
            self._model_source = model_source
            self._model = Qwen3TTSModel.from_pretrained(
                model_source,
                device_map=config.device,
                dtype=dtype,
                attn_implementation=config.attention_implementation,
            )
        except Exception as exc:
            raise EngineError(f"Failed to load Qwen model: {exc}") from exc

    def prepare_reference(self, reference: VoiceReference) -> None:
        if reference.language in self._prompts:
            return
        try:
            self._prompts[reference.language] = self._model.create_voice_clone_prompt(
                ref_audio=str(reference.audio_path),
                ref_text=reference.transcript,
                x_vector_only_mode=False,
            )
        except Exception as exc:
            raise EngineError(
                f"Failed to prepare {reference.language} voice-clone reference: {exc}"
            ) from exc

    def generate(
        self,
        text: str,
        language: str,
        reference_language: str,
        output: Path,
        *,
        max_new_tokens: int | None = None,
        seed_offset: int = 0,
    ) -> float:
        prompt = self._prompts.get(reference_language)
        if prompt is None:
            raise EngineError(f"Reference {reference_language} was not prepared")
        if language not in QWEN_LANGUAGE_NAMES:
            raise EngineError(f"Unsupported Qwen language: {language}")
        kwargs: dict[str, Any] = {"do_sample": True}
        effective_max_new_tokens = (
            self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        )
        if effective_max_new_tokens is not None:
            kwargs["max_new_tokens"] = effective_max_new_tokens
        started = time.perf_counter()
        for attempt in range(self.config.max_generation_attempts):
            attempt_seed = self.config.seed + seed_offset + attempt
            self._torch.manual_seed(attempt_seed)
            if self.config.device.startswith("cuda"):
                self._torch.cuda.manual_seed_all(attempt_seed)
            try:
                wavs, sample_rate = self._model.generate_voice_clone(
                    text=text,
                    language=QWEN_LANGUAGE_NAMES[language],
                    voice_clone_prompt=prompt,
                    non_streaming_mode=True,
                    **kwargs,
                )
            except Exception as exc:
                raise EngineError(f"Qwen generation failed: {exc}") from exc
            if not wavs:
                raise EngineError("Qwen generation returned no waveform")
            samples = np.asarray(wavs[0], dtype=np.float32)
            if _reached_generation_limit(
                len(samples),
                int(sample_rate),
                effective_max_new_tokens,
            ):
                if attempt + 1 == self.config.max_generation_attempts:
                    raise EngineError(
                        "Qwen reached the generation token limit on "
                        f"{self.config.max_generation_attempts} attempts; rejecting noisy output"
                    )
                continue
            elapsed = time.perf_counter() - started
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, samples, int(sample_rate), subtype="PCM_16")
            return elapsed
        raise EngineError("Qwen generation attempts were exhausted")

    def runtime_metadata(self) -> dict[str, object]:
        torch = self._torch
        result: dict[str, object] = {
            "engine": self.name,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "model_snapshot_path": self._model_source,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        try:
            result["qwen_tts_version"] = metadata.version("qwen-tts")
        except metadata.PackageNotFoundError:
            result["qwen_tts_version"] = None
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            result.update(
                {
                    "gpu_name": properties.name,
                    "gpu_total_memory_bytes": properties.total_memory,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                }
            )
        return result
