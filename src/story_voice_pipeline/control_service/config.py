"""Control-service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import PipelineConfig
from ..errors import PipelineError


@dataclass(frozen=True)
class ControlSettings:
    data_root: Path = Path("data")
    story_library_path: Path = Path("story_library/library.yaml")
    database_url: str | None = None
    tts_engine: str = "fake"
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_bytes: int = 50 * 1024 * 1024
    pipeline: PipelineConfig | None = None

    def __post_init__(self) -> None:
        if self.tts_engine not in {"fake", "qwen"}:
            raise PipelineError("CONTROL_TTS_ENGINE must be fake or qwen")
        if not 1 <= self.port <= 65_535:
            raise PipelineError("CONTROL_PORT must be between 1 and 65535")
        if self.max_upload_bytes <= 0:
            raise PipelineError("CONTROL_MAX_UPLOAD_BYTES must be positive")

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        database_path = (self.data_root / "control.db").resolve()
        return f"sqlite:///{database_path}"

    @property
    def pipeline_config(self) -> PipelineConfig:
        if self.pipeline is not None:
            return self.pipeline
        return PipelineConfig(output_root=self.data_root)

    @classmethod
    def from_environment(cls) -> ControlSettings:
        data_root = Path(os.getenv("CONTROL_DATA_ROOT", "data"))
        return cls(
            data_root=data_root,
            story_library_path=Path(
                os.getenv("CONTROL_STORY_LIBRARY", "story_library/library.yaml")
            ),
            database_url=os.getenv("DATABASE_URL"),
            tts_engine=os.getenv("CONTROL_TTS_ENGINE", "fake").lower(),
            host=os.getenv("CONTROL_HOST", "127.0.0.1"),
            port=int(os.getenv("CONTROL_PORT", "8000")),
            max_upload_bytes=int(os.getenv("CONTROL_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))),
            pipeline=PipelineConfig(
                output_root=data_root,
                model_name=os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
                model_revision=os.getenv(
                    "QWEN_MODEL_REVISION", "5d83992436eae1d760afd27aff78a71d676296fc"
                ),
                device=os.getenv("QWEN_DEVICE", "cuda:0"),
                dtype=os.getenv("QWEN_DTYPE", "bfloat16"),
                attention_implementation=os.getenv("QWEN_ATTENTION", "sdpa"),
                max_chunk_characters=int(os.getenv("MAX_CHUNK_CHARACTERS", "400")),
                chunk_silence_ms=int(os.getenv("CHUNK_SILENCE_MS", "350")),
                seed=int(os.getenv("QWEN_SEED", "42")),
                max_new_tokens=int(os.getenv("QWEN_MAX_NEW_TOKENS", "600")),
                max_generation_attempts=int(os.getenv("QWEN_MAX_GENERATION_ATTEMPTS", "3")),
            ),
        )
