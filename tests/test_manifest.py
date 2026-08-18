from __future__ import annotations

from pathlib import Path

import pytest

from story_voice_pipeline.errors import PipelineError
from story_voice_pipeline.manifest import load_manifest, write_manifest


def manifest_fixture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "pack_id": "pack_1",
        "status": "ready_for_review",
        "created_at": "2026-07-24T00:00:00Z",
        "voice": {},
        "library": {},
        "generation": {},
        "stories": [
            {
                "story_id": "story",
                "audio": {
                    "en": {
                        "path": "story/en.wav",
                        "sha256": "a" * 64,
                        "duration_seconds": 1.0,
                        "sample_rate": 24_000,
                    }
                },
                "word_audio": {
                    "en": {
                        "path": "story/word/en.wav",
                        "sha256": "b" * 64,
                        "duration_seconds": 0.7,
                        "sample_rate": 24_000,
                    }
                },
            }
        ],
    }


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    expected = manifest_fixture()
    write_manifest(path, expected)
    assert load_manifest(path) == expected


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    value = manifest_fixture()
    value["stories"][0]["audio"]["en"]["path"] = "../secret.wav"  # type: ignore[index]
    with pytest.raises(PipelineError, match="Unsafe"):
        write_manifest(tmp_path / "manifest.json", value)


def test_manifest_accepts_added_language_codes(tmp_path: Path) -> None:
    value = manifest_fixture()
    audio = value["stories"][0]["audio"]  # type: ignore[index]
    audio["es"] = {  # type: ignore[index]
        "path": "story/es.wav",
        "sha256": "b" * 64,
        "duration_seconds": 1.0,
        "sample_rate": 24_000,
    }
    word_audio = value["stories"][0]["word_audio"]  # type: ignore[index]
    word_audio["es"] = {  # type: ignore[index]
        "path": "story/word/es.wav",
        "sha256": "c" * 64,
        "duration_seconds": 0.7,
        "sample_rate": 24_000,
    }
    write_manifest(tmp_path / "manifest.json", value)


def test_manifest_rejects_word_audio_language_mismatch(tmp_path: Path) -> None:
    value = manifest_fixture()
    value["stories"][0]["word_audio"]["zh"] = {  # type: ignore[index]
        "path": "story/word/zh.wav",
        "sha256": "d" * 64,
        "duration_seconds": 0.7,
        "sample_rate": 24_000,
    }
    with pytest.raises(PipelineError, match="word_audio languages"):
        write_manifest(tmp_path / "manifest.json", value)
