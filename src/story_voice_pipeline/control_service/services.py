"""Local Phase 2 generation worker and pack persistence."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from ..chunking import chunk_text
from ..engines import FakeTTSEngine, QwenTTSEngine
from ..manifest import load_manifest
from ..pipeline import GenerateRequest, StoryPackGenerator
from ..story_library import load_story_library
from .config import ControlSettings
from .database import Database
from .models import GeneratedAudio, GenerationJob, StoryPack, Voice, now_utc


class JobCancellationRequested(RuntimeError):
    """Raised at a chunk boundary when a user cancels a running job."""


@dataclass(frozen=True)
class VoiceSnapshot:
    voice_id: str
    name: str
    version: int
    english_path: Path
    english_transcript: str
    mandarin_path: Path | None
    mandarin_transcript: str | None


class LocalJobProcessor:
    """Serial local processor; persistent remote workers are deferred to Phase 3."""

    def __init__(self, settings: ControlSettings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self._generation_lock = threading.Lock()

    def _data_path(self, stored_path: str) -> Path:
        return (self.settings.data_root / stored_path).resolve()

    def _engine(self):  # type: ignore[no-untyped-def]
        config = self.settings.pipeline_config
        if self.settings.tts_engine == "fake":
            return FakeTTSEngine(config.sample_rate, config.seed)
        return QwenTTSEngine(config)

    def _snapshot_job(self, job_id: str) -> tuple[VoiceSnapshot, list[str], list[str]]:
        with self.database.sessions() as session:
            job = session.get(GenerationJob, job_id)
            if job is None:
                raise RuntimeError(f"Job no longer exists: {job_id}")
            voice = session.get(Voice, job.voice_id)
            if voice is None:
                raise RuntimeError(f"Voice no longer exists: {job.voice_id}")
            mandarin_path = (
                self._data_path(voice.mandarin_reference_path)
                if voice.mandarin_reference_path
                else None
            )
            snapshot = VoiceSnapshot(
                voice_id=voice.id,
                name=voice.name,
                version=voice.voice_version,
                english_path=self._data_path(voice.english_reference_path),
                english_transcript=voice.english_reference_transcript,
                mandarin_path=mandarin_path,
                mandarin_transcript=voice.mandarin_reference_transcript,
            )
            return snapshot, json.loads(job.story_ids_json), json.loads(job.languages_json)

    def _set_initial_state(self, job_id: str, total_chunks: int) -> bool:
        with self.database.sessions() as session:
            job = session.get(GenerationJob, job_id)
            if job is None or job.status == "CANCELLED":
                return False
            job.status = "VALIDATING_REFERENCE"
            job.started_at = now_utc()
            job.completed_at = None
            job.total_chunks = total_chunks
            job.completed_chunks = 0
            job.current_chunk = 0
            job.current_story_id = None
            job.current_language = None
            job.error_message = None
            session.commit()
            return True

    def _event_callback(self, job_id: str):  # type: ignore[no-untyped-def]
        def callback(event: str, fields: dict[str, Any]) -> None:
            with self.database.sessions() as session:
                job = session.get(GenerationJob, job_id)
                if job is None:
                    raise RuntimeError(f"Job no longer exists: {job_id}")
                if job.cancel_requested:
                    raise JobCancellationRequested("Generation cancelled by user")
                if event in {"run_started", "run_resumed"}:
                    job.status = "PREPARING_VOICE"
                elif event in {"chunk_generated", "chunk_cache_hit"}:
                    job.status = "GENERATING"
                    job.completed_chunks += 1
                    job.current_story_id = fields.get("story_id")
                    job.current_language = fields.get("language")
                    job.current_chunk = int(fields.get("chunk_index", 0))
                session.commit()

        return callback

    def _persist_pack(self, job_id: str, pack_path: Path) -> None:
        manifest_path = pack_path / "manifest.json"
        manifest = load_manifest(manifest_path)
        with self.database.sessions() as session:
            job = session.get(GenerationJob, job_id)
            if job is None:
                raise RuntimeError(f"Job no longer exists: {job_id}")
            existing = session.scalar(select(StoryPack).where(StoryPack.job_id == job_id))
            if existing is not None:
                session.delete(existing)
                session.flush()
            pack = StoryPack(
                id=manifest["pack_id"],
                job_id=job.id,
                voice_id=job.voice_id,
                voice_version=manifest["voice"]["voice_version"],
                library_id=manifest["library"]["library_id"],
                library_version=manifest["library"]["library_version"],
                generation_version=manifest["generation"]["generation_version"],
                model_name=manifest["generation"]["model_name"],
                model_revision=manifest["generation"]["model_revision"],
                status="READY_FOR_REVIEW",
                manifest_path=str(
                    manifest_path.resolve().relative_to(self.settings.data_root.resolve())
                ),
                pack_path=str(pack_path.resolve().relative_to(self.settings.data_root.resolve())),
            )
            session.add(pack)
            for story in manifest["stories"]:
                for audio_group in (story["audio"], story.get("word_audio", {})):
                    for language, audio in audio_group.items():
                        session.add(
                            GeneratedAudio(
                                id=f"audio_{uuid.uuid4().hex}",
                                story_pack_id=pack.id,
                                story_id=story["story_id"],
                                story_version=story["story_version"],
                                language=language,
                                source_text_hash=audio["source_text_hash"],
                                audio_path=audio["path"],
                                audio_hash=audio["sha256"],
                                duration_seconds=audio["duration_seconds"],
                                sample_rate=audio["sample_rate"],
                                generation_seconds=audio["generation_seconds"],
                                real_time_factor=audio["real_time_factor"],
                                validation_status=("WARNING" if audio["warnings"] else "PASSED"),
                                warnings_json=json.dumps(audio["warnings"], ensure_ascii=False),
                            )
                        )
            job.status = "READY_FOR_REVIEW"
            job.pack_id = pack.id
            job.completed_at = now_utc()
            session.commit()

    def process(self, job_id: str) -> None:
        with self._generation_lock:
            try:
                voice, story_ids, languages = self._snapshot_job(job_id)
                library = load_story_library(self.settings.story_library_path)
                selected = {story.story_id: story for story in library.stories}
                total_chunks = sum(
                    len(
                        chunk_text(
                            selected[story_id].texts[language],
                            language,
                            self.settings.pipeline_config.max_chunk_characters,
                        )
                    )
                    + 1
                    for story_id in story_ids
                    for language in languages
                )
                if not self._set_initial_state(job_id, total_chunks):
                    return
                with self.database.sessions() as session:
                    job = session.get(GenerationJob, job_id)
                    if job is None:
                        return
                    run_id = job.run_id
                request_path = self.settings.data_root / "jobs" / run_id / "request.json"
                if request_path.is_file():
                    previous = json.loads(request_path.read_text(encoding="utf-8"))
                    if previous.get("status") == "completed":
                        pack_path = Path(previous["pack_path"])
                        if pack_path.is_dir():
                            self._persist_pack(job_id, pack_path)
                            return
                resume = request_path.is_file()
                generator = StoryPackGenerator(
                    self.settings.pipeline_config,
                    self._engine(),
                    event_callback=self._event_callback(job_id),
                )
                result = generator.generate(
                    GenerateRequest(
                        library=library,
                        reference_audio=voice.english_path,
                        reference_transcript=voice.english_transcript,
                        voice_id=voice.voice_id,
                        voice_name=voice.name,
                        voice_version=voice.version,
                        mandarin_reference_audio=voice.mandarin_path,
                        mandarin_reference_transcript=voice.mandarin_transcript,
                        story_ids=tuple(story_ids),
                        languages=tuple(languages),
                        run_id=run_id,
                        resume=resume,
                    )
                )
                self._persist_pack(job_id, result.pack_path)
            except JobCancellationRequested:
                with self.database.sessions() as session:
                    job = session.get(GenerationJob, job_id)
                    if job is not None:
                        job.status = "CANCELLED"
                        job.completed_at = now_utc()
                        job.error_message = "Generation cancelled by user"
                        session.commit()
            except Exception as exc:
                with self.database.sessions() as session:
                    job = session.get(GenerationJob, job_id)
                    if job is not None:
                        job.status = "FAILED"
                        job.completed_at = now_utc()
                        job.error_message = str(exc)
                        session.commit()
