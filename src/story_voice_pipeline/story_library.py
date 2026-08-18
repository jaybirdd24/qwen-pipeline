"""Folder-based story library discovery and validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .errors import LibraryValidationError
from .hashing import sha256_text
from .models import SUPPORTED_LANGUAGES, Story, StoryLibrary

SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LibraryValidationError(f"Required YAML file does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LibraryValidationError(f"Cannot read YAML file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LibraryValidationError(f"YAML document must be a mapping: {path}")
    return value


def _required_int(data: dict[str, Any], key: str, path: Path) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise LibraryValidationError(f"{path}: {key} must be a positive integer")
    return value


def _required_id(data: dict[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise LibraryValidationError(
            f"{path}: {key} must contain lowercase letters, digits, hyphens, or underscores"
        )
    return value


def load_story_library(manifest_path: Path) -> StoryLibrary:
    manifest_path = manifest_path.resolve()
    data = _load_yaml(manifest_path)
    library_id = _required_id(data, "library_id", manifest_path)
    version = _required_int(data, "version", manifest_path)

    required = data.get("required_languages")
    if not isinstance(required, list) or not required:
        raise LibraryValidationError(f"{manifest_path}: required_languages must be a list")
    if len(required) != len(set(required)):
        raise LibraryValidationError(f"{manifest_path}: required_languages contains duplicates")
    invalid_languages = [language for language in required if language not in SUPPORTED_LANGUAGES]
    if invalid_languages:
        raise LibraryValidationError(
            f"{manifest_path}: unsupported languages: {', '.join(map(str, invalid_languages))}"
        )

    story_entries = data.get("stories")
    if not isinstance(story_entries, list) or not story_entries:
        raise LibraryValidationError(f"{manifest_path}: stories must be a non-empty list")
    if any(not isinstance(entry, str) or not SAFE_ID.fullmatch(entry) for entry in story_entries):
        raise LibraryValidationError(f"{manifest_path}: every story directory must be a safe ID")
    if len(story_entries) != len(set(story_entries)):
        raise LibraryValidationError(f"{manifest_path}: stories contains duplicate directory IDs")

    stories: list[Story] = []
    seen_ids: set[str] = set()
    for directory_name in story_entries:
        directory = manifest_path.parent / "stories" / directory_name
        story_path = directory / "story.yaml"
        story_data = _load_yaml(story_path)
        story_id = _required_id(story_data, "story_id", story_path)
        if story_id != directory_name:
            raise LibraryValidationError(
                f"{story_path}: story_id {story_id!r} must match directory {directory_name!r}"
            )
        if story_id in seen_ids:
            raise LibraryValidationError(f"Duplicate story_id: {story_id}")
        seen_ids.add(story_id)

        story_version = _required_int(story_data, "version", story_path)
        titles = story_data.get("title")
        if not isinstance(titles, dict) or any(
            not isinstance(titles.get(language), str) or not titles[language].strip()
            for language in required
        ):
            raise LibraryValidationError(
                f"{story_path}: title must contain every required language"
            )

        spoken_words = story_data.get("spoken_word")
        if not isinstance(spoken_words, dict) or any(
            not isinstance(spoken_words.get(language), str) or not spoken_words[language].strip()
            for language in required
        ):
            raise LibraryValidationError(
                f"{story_path}: spoken_word must contain every required language"
            )

        triggers = story_data.get("trigger_objects")
        if (
            not isinstance(triggers, list)
            or not triggers
            or any(not isinstance(item, str) or not SAFE_ID.fullmatch(item) for item in triggers)
        ):
            raise LibraryValidationError(
                f"{story_path}: trigger_objects must be a non-empty list of safe IDs"
            )

        texts: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for language in required:
            text_path = directory / f"{language}.txt"
            if not text_path.is_file():
                raise LibraryValidationError(f"Missing translation: {text_path}")
            try:
                text = text_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                raise LibraryValidationError(f"Cannot read story text {text_path}: {exc}") from exc
            if not text:
                raise LibraryValidationError(f"Story text is empty: {text_path}")
            texts[language] = text
            hashes[language] = sha256_text(text)

        stories.append(
            Story(
                story_id=story_id,
                version=story_version,
                title={language: titles[language].strip() for language in required},
                spoken_word={language: spoken_words[language].strip() for language in required},
                trigger_objects=tuple(triggers),
                texts=texts,
                source_hashes=hashes,
                directory=directory.resolve(),
            )
        )

    return StoryLibrary(
        library_id=library_id,
        version=version,
        required_languages=tuple(required),
        stories=tuple(stories),
        manifest_path=manifest_path,
    )
