"""Manifest writing and minimal compatibility validation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .config import MANIFEST_SCHEMA_VERSION
from .errors import PipelineError
from .models import SUPPORTED_LANGUAGES

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def validate_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise PipelineError("Manifest must be a JSON object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PipelineError("Unsupported manifest schema_version")
    if manifest.get("status") not in {"ready_for_review", "approved"}:
        raise PipelineError("Manifest status must be ready_for_review or approved")
    for key in ("pack_id", "created_at", "voice", "library", "generation", "stories"):
        if key not in manifest:
            raise PipelineError(f"Manifest is missing {key}")
    if not isinstance(manifest["pack_id"], str) or not SAFE_ID.fullmatch(manifest["pack_id"]):
        raise PipelineError("Manifest pack_id is unsafe")
    if manifest["status"] == "approved" and not manifest.get("approved_at"):
        raise PipelineError("Approved manifest must include approved_at")
    if any(not isinstance(manifest[key], dict) for key in ("voice", "library", "generation")):
        raise PipelineError("Manifest voice, library, and generation fields must be objects")
    if not isinstance(manifest["stories"], list) or not manifest["stories"]:
        raise PipelineError("Manifest must contain at least one story")
    seen_story_ids: set[str] = set()
    for story in manifest["stories"]:
        if not isinstance(story, dict):
            raise PipelineError("Every manifest story must be an object")
        story_id = story.get("story_id")
        if not isinstance(story_id, str) or not SAFE_ID.fullmatch(story_id):
            raise PipelineError("Manifest contains an unsafe story_id")
        if story_id in seen_story_ids:
            raise PipelineError(f"Manifest contains duplicate story_id: {story_id}")
        seen_story_ids.add(story_id)
        if not isinstance(story.get("audio"), dict) or not story["audio"]:
            raise PipelineError(f"Story {story_id} has no audio")
        audio_groups = {"audio": story["audio"]}
        if "word_audio" in story:
            if not isinstance(story["word_audio"], dict) or not story["word_audio"]:
                raise PipelineError(f"Story {story_id} has invalid word_audio")
            if set(story["word_audio"]) != set(story["audio"]):
                raise PipelineError(f"Story {story_id} word_audio languages do not match audio")
            audio_groups["word_audio"] = story["word_audio"]
        for group_name, group in audio_groups.items():
            for language, audio in group.items():
                if language not in SUPPORTED_LANGUAGES:
                    raise PipelineError(f"Unsupported manifest language: {language}")
                if not isinstance(audio, dict):
                    raise PipelineError(f"Story {story_id} {group_name} entry must be an object")
                path = audio.get("path", "")
                if not path or Path(path).is_absolute() or ".." in Path(path).parts:
                    raise PipelineError(f"Unsafe audio path in manifest: {path}")
                for key in ("sha256", "duration_seconds", "sample_rate"):
                    if key not in audio:
                        raise PipelineError(f"Audio entry {path} is missing {key}")
                if not isinstance(audio["sha256"], str) or not SHA256.fullmatch(audio["sha256"]):
                    raise PipelineError(f"Audio entry {path} has an invalid sha256")
                if (
                    isinstance(audio["duration_seconds"], bool)
                    or not isinstance(audio["duration_seconds"], (int, float))
                    or audio["duration_seconds"] <= 0
                ):
                    raise PipelineError(f"Audio entry {path} has an invalid duration")
                if (
                    isinstance(audio["sample_rate"], bool)
                    or not isinstance(audio["sample_rate"], int)
                    or audio["sample_rate"] <= 0
                ):
                    raise PipelineError(f"Audio entry {path} has an invalid sample rate")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read manifest {path}: {exc}") from exc
    validate_manifest(manifest)
    return manifest
