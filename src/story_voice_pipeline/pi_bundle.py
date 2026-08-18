"""Create self-contained, GPU-free bundles for the Raspberry Pi player."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import PipelineError
from .manifest import load_manifest
from .pipeline import utc_now

BUNDLE_SCHEMA_VERSION = 1
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


@dataclass(frozen=True)
class PiBundleResult:
    output_dir: Path
    selected_voice_id: str
    voice_count: int
    audio_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PipelineError(f"Cannot read pack audio {path}: {exc}") from exc
    return digest.hexdigest()


def _manifest_path(pack: Path) -> Path:
    candidate = pack / "manifest.json" if pack.is_dir() else pack
    if candidate.name != "manifest.json":
        raise PipelineError(f"Pack must be a directory or manifest.json path: {pack}")
    return candidate.resolve()


def _voice_details(manifest: dict[str, Any], path: Path) -> tuple[str, str, int]:
    voice = manifest["voice"]
    voice_id = voice.get("voice_id")
    voice_name = voice.get("name") or voice.get("voice_name")
    voice_version = voice.get("voice_version")
    if not isinstance(voice_id, str) or not SAFE_ID.fullmatch(voice_id):
        raise PipelineError(f"Manifest {path} has an unsafe or missing voice_id")
    if not isinstance(voice_name, str) or not voice_name.strip():
        raise PipelineError(f"Manifest {path} has a missing voice name")
    if isinstance(voice_version, bool) or not isinstance(voice_version, int) or voice_version < 1:
        raise PipelineError(f"Manifest {path} has an invalid voice_version")
    return voice_id, voice_name.strip(), voice_version


def _story_signature(manifest: dict[str, Any], path: Path) -> tuple[tuple[str, Any], ...]:
    signature: list[tuple[str, Any]] = []
    seen_triggers: dict[str, str] = {}
    for story in manifest["stories"]:
        triggers = story.get("trigger_objects")
        if not isinstance(triggers, list) or not triggers:
            raise PipelineError(f"Story {story['story_id']} in {path} has no trigger objects")
        for trigger in triggers:
            if not isinstance(trigger, str) or not SAFE_ID.fullmatch(trigger):
                raise PipelineError(f"Story {story['story_id']} in {path} has an unsafe trigger")
            normalized = trigger.casefold()
            previous = seen_triggers.get(normalized)
            if previous is not None and previous != story["story_id"]:
                raise PipelineError(
                    f"Trigger {trigger} selects both {previous} and {story['story_id']}"
                )
            seen_triggers[normalized] = story["story_id"]
        signature.append(
            (
                story["story_id"],
                (
                    story.get("story_version"),
                    tuple(trigger.casefold() for trigger in triggers),
                    tuple(sorted(story.get("spoken_word", {}).items())),
                    tuple(sorted(story["audio"])),
                    tuple(sorted(story.get("word_audio", {}))),
                ),
            )
        )
    return tuple(signature)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_pi_bundle(
    packs: list[Path],
    output_dir: Path,
    selected_voice_id: str | None = None,
    *,
    allow_review_ready: bool = False,
) -> PiBundleResult:
    """Export one compatible pack per voice into a new, self-contained directory."""
    if not packs:
        raise PipelineError("At least one --pack is required")
    if output_dir.exists():
        raise PipelineError(f"Output directory already exists: {output_dir}")

    loaded: list[tuple[Path, dict[str, Any], str, str, int]] = []
    seen_voice_ids: set[str] = set()
    expected_library: tuple[Any, Any] | None = None
    expected_stories: tuple[tuple[str, Any], ...] | None = None

    for pack in packs:
        manifest_path = _manifest_path(pack)
        manifest = load_manifest(manifest_path)
        if manifest["status"] != "approved" and not (
            allow_review_ready and manifest["status"] == "ready_for_review"
        ):
            raise PipelineError(
                f"Pack {manifest['pack_id']} is not approved; use --allow-review-ready "
                "only for prototypes"
            )
        voice_id, voice_name, voice_version = _voice_details(manifest, manifest_path)
        if voice_id in seen_voice_ids:
            raise PipelineError(f"More than one pack was supplied for voice {voice_id}")
        seen_voice_ids.add(voice_id)

        library = manifest["library"]
        library_key = (library.get("library_id"), library.get("library_version"))
        stories = _story_signature(manifest, manifest_path)
        if expected_library is None:
            expected_library = library_key
            expected_stories = stories
        elif library_key != expected_library or stories != expected_stories:
            raise PipelineError("All Pi packs must contain the same library, stories, and triggers")
        loaded.append((manifest_path, manifest, voice_id, voice_name, voice_version))

    selected = selected_voice_id or loaded[0][2]
    if selected not in seen_voice_ids:
        raise PipelineError(f"Selected voice is not present in the bundle: {selected}")

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    voices: list[dict[str, Any]] = []
    active_packs: dict[str, str] = {}
    audio_count = 0
    try:
        for manifest_path, manifest, voice_id, voice_name, voice_version in loaded:
            pack_id = manifest["pack_id"]
            source_root = manifest_path.parent
            target_root = staging / "packs" / voice_id / pack_id
            target_root.mkdir(parents=True)

            for story in manifest["stories"]:
                for group in (story["audio"], story.get("word_audio", {})):
                    for audio in group.values():
                        relative = Path(audio["path"])
                        source = source_root / relative
                        if not source.is_file():
                            raise PipelineError(f"Pack audio is missing: {source}")
                        actual_hash = _sha256(source)
                        if actual_hash != audio["sha256"]:
                            raise PipelineError(f"Pack audio checksum failed: {source}")
                        target = target_root / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                        if _sha256(target) != audio["sha256"]:
                            raise PipelineError(f"Copied audio checksum failed: {target}")
                        audio_count += 1

            shutil.copy2(manifest_path, target_root / "manifest.json")
            active_packs[voice_id] = pack_id
            voices.append(
                {
                    "voice_id": voice_id,
                    "voice_name": voice_name,
                    "voice_version": voice_version,
                    "pack_id": pack_id,
                    "pack_status": manifest["status"],
                    "manifest_path": f"packs/{voice_id}/{pack_id}/manifest.json",
                }
            )

        catalog = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at": utc_now(),
            "library": {
                "library_id": expected_library[0],
                "library_version": expected_library[1],
            },
            "voices": voices,
        }
        state = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "selected_voice_id": selected,
            "active_packs": active_packs,
        }
        _write_json(staging / "catalog.json", catalog)
        _write_json(staging / "player-state.json", state)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return PiBundleResult(output_dir, selected, len(voices), audio_count)
