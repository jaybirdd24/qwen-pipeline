"""Pipeline configuration and defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import PipelineError

DEFAULT_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_MODEL_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
CHUNKING_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PipelineConfig:
    output_root: Path = Path("data")
    model_name: str = DEFAULT_MODEL_NAME
    model_revision: str = DEFAULT_MODEL_REVISION
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    sample_rate: int = 24_000
    max_chunk_characters: int = 400
    chunk_silence_ms: int = 350
    peak_dbfs: float = -1.0
    seed: int = 42
    max_new_tokens: int | None = None
    max_generation_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.model_name.strip() or not self.model_revision.strip():
            raise PipelineError("Model name and revision must not be empty")
        if self.sample_rate <= 0:
            raise PipelineError("Sample rate must be positive")
        if self.max_chunk_characters <= 0:
            raise PipelineError("Maximum chunk characters must be positive")
        if self.chunk_silence_ms < 0:
            raise PipelineError("Chunk silence must not be negative")
        if not -60 <= self.peak_dbfs <= 0:
            raise PipelineError("Peak dBFS must be between -60 and 0")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise PipelineError("Maximum new tokens must be positive")
        if self.max_generation_attempts <= 0:
            raise PipelineError("Maximum generation attempts must be positive")

    def generation_parameters(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("output_root")
        return values
