from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from story_voice_pipeline.errors import PipelineError
from story_voice_pipeline.pi_bundle import export_pi_bundle


def _pack(root: Path, voice_id: str, status: str = "approved") -> Path:
    pack_id = f"pack_{voice_id}"
    pack = root / pack_id
    audio_path = pack / "apple_picnic" / "en.wav"
    audio_path.parent.mkdir(parents=True)
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(b"\x01\x00" * 24_000)
    word_path = pack / "apple_picnic" / "word" / "en.wav"
    word_path.parent.mkdir(parents=True)
    word_path.write_bytes(audio_path.read_bytes())
    digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "pack_id": pack_id,
        "status": status,
        "approved_at": "2026-07-28T00:00:00Z" if status == "approved" else None,
        "created_at": "2026-07-28T00:00:00Z",
        "voice": {"voice_id": voice_id, "name": voice_id.title(), "voice_version": 1},
        "library": {"library_id": "samples", "library_version": 1},
        "generation": {},
        "stories": [
            {
                "story_id": "apple_picnic",
                "story_version": 1,
                "trigger_objects": ["apple"],
                "audio": {
                    "en": {
                        "path": "apple_picnic/en.wav",
                        "sha256": digest,
                        "duration_seconds": 1.0,
                        "sample_rate": 24_000,
                    }
                },
                "word_audio": {
                    "en": {
                        "path": "apple_picnic/word/en.wav",
                        "sha256": digest,
                        "duration_seconds": 1.0,
                        "sample_rate": 24_000,
                    }
                },
            }
        ],
    }
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pack


def test_exports_multiple_voices_and_state(tmp_path: Path) -> None:
    mum = _pack(tmp_path / "source", "mum")
    dad = _pack(tmp_path / "source", "dad")
    output = tmp_path / "pi-bundle"

    result = export_pi_bundle([mum, dad], output, "dad")

    assert result.voice_count == 2
    assert result.audio_count == 4
    state = json.loads((output / "player-state.json").read_text())
    assert state["selected_voice_id"] == "dad"
    assert state["active_packs"] == {"mum": "pack_mum", "dad": "pack_dad"}
    catalog = json.loads((output / "catalog.json").read_text())
    assert [voice["voice_id"] for voice in catalog["voices"]] == ["mum", "dad"]
    assert (output / "packs/mum/pack_mum/apple_picnic/en.wav").is_file()
    assert (output / "packs/mum/pack_mum/apple_picnic/word/en.wav").is_file()


def test_requires_approval_by_default(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "source", "mum", "ready_for_review")
    with pytest.raises(PipelineError, match="not approved"):
        export_pi_bundle([pack], tmp_path / "output")

    export_pi_bundle([pack], tmp_path / "prototype", allow_review_ready=True)


def test_rejects_corrupt_audio_and_existing_output(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "source", "mum")
    (pack / "apple_picnic/en.wav").write_bytes(b"corrupt")
    with pytest.raises(PipelineError, match="checksum"):
        export_pi_bundle([pack], tmp_path / "output")

    valid = _pack(tmp_path / "second", "mum")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(PipelineError, match="already exists"):
        export_pi_bundle([valid], existing)
