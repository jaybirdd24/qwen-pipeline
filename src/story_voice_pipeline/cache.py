"""Content-addressed cache for validated generated chunks."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import validate_wav
from .errors import PipelineError
from .hashing import stable_json_hash


@dataclass(frozen=True)
class CacheResult:
    hit: bool
    wav_path: Path
    metadata: dict[str, Any]


def chunk_cache_key(payload: dict[str, Any]) -> str:
    return stable_json_hash(payload)


class ChunkCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def paths(self, key: str) -> tuple[Path, Path]:
        directory = self.root / key[:2] / key
        return directory / "chunk.wav", directory / "metadata.json"

    def get(self, key: str) -> CacheResult | None:
        wav_path, metadata_path = self.paths(key)
        if not wav_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cache_key") != key:
                return None
            validation = validate_wav(wav_path)
            if validation.sha256 != metadata.get("audio_sha256"):
                return None
        except (OSError, ValueError, TypeError, PipelineError):
            return None
        return CacheResult(hit=True, wav_path=wav_path, metadata=metadata)

    def put(self, key: str, generated_wav: Path, metadata: dict[str, Any]) -> CacheResult:
        validation = validate_wav(generated_wav, text_length=metadata.get("text_length"))
        wav_path, metadata_path = self.paths(key)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_wav = wav_path.with_suffix(".tmp.wav")
        shutil.copyfile(generated_wav, temporary_wav)
        os.replace(temporary_wav, wav_path)
        stored = {
            **metadata,
            "cache_key": key,
            "audio_sha256": validation.sha256,
            "duration_seconds": validation.duration_seconds,
            "sample_rate": validation.sample_rate,
            "warnings": list(validation.warnings),
        }
        temporary_metadata = metadata_path.with_suffix(".tmp.json")
        temporary_metadata.write_text(
            json.dumps(stored, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata, metadata_path)
        return CacheResult(hit=False, wav_path=wav_path, metadata=stored)

    @staticmethod
    def materialize(result: CacheResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.wav_path, destination)
        destination.with_suffix(".json").write_text(
            json.dumps(result.metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
