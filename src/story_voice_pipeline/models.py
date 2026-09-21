"""Shared immutable domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Mandarin Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
}
SUPPORTED_LANGUAGES = tuple(LANGUAGE_NAMES)

QWEN_LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
}


@dataclass(frozen=True)
class Story:
    story_id: str
    version: int
    title: dict[str, str]
    spoken_word: dict[str, str]
    trigger_objects: tuple[str, ...]
    texts: dict[str, str]
    source_hashes: dict[str, str]
    directory: Path


@dataclass(frozen=True)
class StoryLibrary:
    library_id: str
    version: int
    required_languages: tuple[str, ...]
    stories: tuple[Story, ...]
    manifest_path: Path


@dataclass(frozen=True)
class VoiceReference:
    language: str
    audio_path: Path
    transcript: str
    audio_hash: str
    transcript_hash: str
    duration_seconds: float
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AudioValidation:
    path: Path
    sha256: str
    duration_seconds: float
    sample_rate: int
    channels: int
    peak_amplitude: float
    rms_amplitude: float
    warnings: tuple[str, ...]
