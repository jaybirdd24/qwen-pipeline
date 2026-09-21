from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from story_voice_pipeline.errors import LibraryValidationError
from story_voice_pipeline.hashing import sha256_text
from story_voice_pipeline.story_library import load_story_library

SAMPLE_LIBRARY = Path(__file__).parents[1] / "story_library"


def copy_library(tmp_path: Path) -> Path:
    destination = tmp_path / "library"
    shutil.copytree(SAMPLE_LIBRARY, destination)
    return destination / "library.yaml"


def test_loads_sample_library_and_stable_hashes() -> None:
    library = load_story_library(SAMPLE_LIBRARY / "library.yaml")
    assert library.library_id == "starter_samples"
    assert [story.story_id for story in library.stories] == [
        "forest",
        "dad",
        "moon",
        "dog",
        "river",
        "house",
        "bear",
    ]
    assert library.stories[0].spoken_word["en"] == "Forest"
    assert library.stories[0].spoken_word["zh"] == "森林"
    assert set(library.required_languages) == {"en", "zh", "ja", "ko", "de", "pt", "es"}
    assert library.stories[0].source_hashes["en"] == sha256_text(library.stories[0].texts["en"])


def test_rejects_duplicate_story_directory(tmp_path: Path) -> None:
    manifest = copy_library(tmp_path)
    manifest.write_text(manifest.read_text() + "  - forest\n", encoding="utf-8")
    with pytest.raises(LibraryValidationError, match="duplicate"):
        load_story_library(manifest)


def test_rejects_missing_translation(tmp_path: Path) -> None:
    manifest = copy_library(tmp_path)
    (manifest.parent / "stories" / "forest" / "zh.txt").unlink()
    with pytest.raises(LibraryValidationError, match="Missing translation"):
        load_story_library(manifest)


def test_rejects_empty_story(tmp_path: Path) -> None:
    manifest = copy_library(tmp_path)
    (manifest.parent / "stories" / "forest" / "en.txt").write_text("\n")
    with pytest.raises(LibraryValidationError, match="empty"):
        load_story_library(manifest)


def test_rejects_story_id_that_does_not_match_directory(tmp_path: Path) -> None:
    manifest = copy_library(tmp_path)
    story_yaml = manifest.parent / "stories" / "forest" / "story.yaml"
    story_yaml.write_text(story_yaml.read_text().replace("forest", "other_story", 1))
    with pytest.raises(LibraryValidationError, match="must match directory"):
        load_story_library(manifest)


def test_rejects_missing_spoken_word_translation(tmp_path: Path) -> None:
    manifest = copy_library(tmp_path)
    story_yaml = manifest.parent / "stories" / "forest" / "story.yaml"
    metadata = yaml.safe_load(story_yaml.read_text())
    del metadata["spoken_word"]["zh"]
    story_yaml.write_text(yaml.safe_dump(metadata, allow_unicode=True))
    with pytest.raises(LibraryValidationError, match="spoken_word"):
        load_story_library(manifest)
