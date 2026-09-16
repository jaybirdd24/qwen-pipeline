"""Cached end-to-end story-pack generation orchestration."""

from __future__ import annotations

import json
import os
import re
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import (
    assemble_wavs,
    normalize_reference,
    prepare_word_audio,
    real_time_factor,
    validate_wav,
)
from .cache import CacheResult, ChunkCache, chunk_cache_key
from .chunking import CHUNKING_VERSION, chunk_text
from .config import MANIFEST_SCHEMA_VERSION, PipelineConfig
from .engines.base import TTSEngine
from .errors import AudioValidationError, EngineError, PipelineError, ResumeError
from .hashing import sha256_text, stable_json_hash
from .manifest import write_manifest
from .models import SUPPORTED_LANGUAGES, Story, StoryLibrary, VoiceReference

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
WORD_GENERATION_VERSION = "word-prompt-v2"
WORD_MAX_DURATION_SECONDS = 2.0
WORD_MAX_NEW_TOKENS = 48
WORD_SILENCE_WINDOW_MS = 20
WORD_PADDING_MS = 50


def _word_synthesis_text(text: str, language: str) -> str:
    punctuation = "。" if language in {"zh", "ja"} else "."
    return text.rstrip(".!?。！？") + punctuation


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_event(run_root: Path, event: str, **fields: Any) -> None:
    log_path = run_root / "logs" / "generation.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": utc_now(), "event": event, **fields}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(frozen=True)
class GenerateRequest:
    library: StoryLibrary
    reference_audio: Path
    reference_transcript: str
    voice_id: str
    voice_name: str
    voice_version: int = 1
    mandarin_reference_audio: Path | None = None
    mandarin_reference_transcript: str | None = None
    story_ids: tuple[str, ...] = ()
    languages: tuple[str, ...] = ("en", "zh")
    run_id: str | None = None
    resume: bool = False


@dataclass(frozen=True)
class GenerationResult:
    run_id: str
    pack_id: str
    pack_path: Path
    manifest_path: Path
    cache_hits: int
    generated_chunks: int


@dataclass
class PlannedChunk:
    """One story destination; synthesis may be shared by identical cache keys."""

    story_id: str
    key: str
    job_path: Path
    payload: dict[str, Any]
    cached: CacheResult | None = None

    @property
    def generated_path(self) -> Path:
        return self.job_path.with_suffix(".generated.wav")

    @property
    def compatibility_key(self) -> str:
        # Includes language, reference audio/transcript, engine/model/revision and
        # all generation parameters (including dtype), but never story identity.
        return stable_json_hash(
            {
                key: value
                for key, value in self.payload.items()
                if key not in {"text", "chunk_index"}
            }
        )


@dataclass
class GenerationTiming:
    story_batch_seconds: float = 0.0
    word_generation_seconds: float = 0.0
    batch_attempts: int = 0
    fallback_count: int = 0

    def metadata(self, started: float) -> dict[str, Any]:
        return {
            "run_wall_seconds": time.perf_counter() - started,
            "generation_wall_seconds": self.story_batch_seconds + self.word_generation_seconds,
            "story_batch_seconds": self.story_batch_seconds,
            "word_generation_seconds": self.word_generation_seconds,
            "batch_attempts": self.batch_attempts,
            "fallback_count": self.fallback_count,
        }


class StoryPackGenerator:
    def __init__(
        self,
        config: PipelineConfig,
        engine: TTSEngine,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.cache = ChunkCache(config.output_root / "cache" / "chunks")
        self.event_callback = event_callback
        self._timing = GenerationTiming()

    def _record_event(self, run_root: Path, event: str, **fields: Any) -> None:
        _append_event(run_root, event, **fields)
        if self.event_callback is not None:
            self.event_callback(event, fields)

    def _validate_request(self, request: GenerateRequest) -> tuple[Story, ...]:
        if not SAFE_ID.fullmatch(request.voice_id):
            raise PipelineError(
                "voice-id must be 1–128 safe letters, digits, hyphens, or underscores"
            )
        if request.voice_version < 1:
            raise PipelineError("voice-version must be positive")
        if not request.voice_name.strip():
            raise PipelineError("voice-name must not be empty")
        if not request.languages or any(
            language not in SUPPORTED_LANGUAGES for language in request.languages
        ):
            raise PipelineError(
                f"languages must use supported codes: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        if len(request.languages) != len(set(request.languages)):
            raise PipelineError("languages contains duplicates")
        unavailable = [
            language
            for language in request.languages
            if language not in request.library.required_languages
        ]
        if unavailable:
            raise PipelineError(
                "Story library has no complete translation for: " + ", ".join(unavailable)
            )
        if (request.mandarin_reference_audio is None) != (
            request.mandarin_reference_transcript is None
        ):
            raise PipelineError("Mandarin reference audio and transcript must be supplied together")
        stories_by_id = {story.story_id: story for story in request.library.stories}
        selected_ids = request.story_ids or tuple(stories_by_id)
        if len(selected_ids) != len(set(selected_ids)):
            raise PipelineError("story selection contains duplicates")
        missing = [story_id for story_id in selected_ids if story_id not in stories_by_id]
        if missing:
            raise PipelineError(f"Unknown story IDs: {', '.join(missing)}")
        return tuple(stories_by_id[story_id] for story_id in selected_ids)

    def _prepare_references(self, request: GenerateRequest) -> dict[str, VoiceReference]:
        reference_root = (
            self.config.output_root / "references" / request.voice_id / f"v{request.voice_version}"
        )
        english = normalize_reference(
            request.reference_audio,
            request.reference_transcript,
            reference_root / "en.wav",
            "en",
            self.config.sample_rate,
        )
        (reference_root / "en.txt").write_text(english.transcript + "\n", encoding="utf-8")
        references = {"en": english}
        if request.mandarin_reference_audio is not None:
            mandarin = normalize_reference(
                request.mandarin_reference_audio,
                request.mandarin_reference_transcript or "",
                reference_root / "zh.wav",
                "zh",
                self.config.sample_rate,
            )
            (reference_root / "zh.txt").write_text(mandarin.transcript + "\n", encoding="utf-8")
            references["zh"] = mandarin
        return references

    def _generate_word_audio(
        self,
        run_root: Path,
        staging_root: Path,
        story: Story,
        language: str,
        reference_language: str,
        reference: VoiceReference,
    ) -> tuple[dict[str, Any], int, int]:
        """Generate the localized card word as a separate, cached audio asset."""
        text = story.spoken_word[language]
        synthesis_text = _word_synthesis_text(text, language)
        source_text_hash = sha256_text(text)
        word_profile = {
            "version": WORD_GENERATION_VERSION,
            "synthesis_text": synthesis_text,
            "max_new_tokens": WORD_MAX_NEW_TOKENS,
            "max_duration_seconds": WORD_MAX_DURATION_SECONDS,
            "silence_window_ms": WORD_SILENCE_WINDOW_MS,
            "padding_ms": WORD_PADDING_MS,
            "maximum_candidates": self.config.max_generation_attempts,
        }
        cache_payload = {
            "cache_schema_version": 1,
            "chunking_version": WORD_GENERATION_VERSION,
            "asset_type": "word",
            "engine": self.engine.name,
            "model_name": self.engine.model_name,
            "model_revision": self.engine.model_revision,
            "generation_parameters": self.config.generation_parameters(),
            "reference_language": reference_language,
            "reference_audio_hash": reference.audio_hash,
            "reference_transcript_hash": reference.transcript_hash,
            "language": language,
            "text": text,
            "word_generation_profile": word_profile,
            "chunk_index": 1,
        }
        key = chunk_cache_key(cache_payload)
        cached = self.cache.get(key)
        job_chunk = run_root / "chunks" / story.story_id / language / "word.wav"
        generated_chunks = 0
        cache_hits = 0
        if cached is None:
            generated_chunks = 1
            candidate = job_chunk.with_name("word.candidate.wav")
            prepared = job_chunk.with_name("word.prepared.wav")
            failures: list[str] = []
            preparation = None
            started = time.perf_counter()
            accepted_attempt = 0
            for attempt in range(self.config.max_generation_attempts):
                candidate.unlink(missing_ok=True)
                prepared.unlink(missing_ok=True)
                try:
                    self.engine.generate(
                        synthesis_text,
                        language,
                        reference_language,
                        candidate,
                        max_new_tokens=WORD_MAX_NEW_TOKENS,
                        seed_offset=attempt * self.config.max_generation_attempts,
                    )
                    preparation = prepare_word_audio(
                        candidate,
                        prepared,
                        self.config.sample_rate,
                        self.config.peak_dbfs,
                        max_duration_seconds=WORD_MAX_DURATION_SECONDS,
                        window_ms=WORD_SILENCE_WINDOW_MS,
                        padding_ms=WORD_PADDING_MS,
                    )
                    accepted_attempt = attempt + 1
                    break
                except (AudioValidationError, EngineError) as exc:
                    failures.append(str(exc))
                    self._record_event(
                        run_root,
                        "word_candidate_rejected",
                        story_id=story.story_id,
                        language=language,
                        candidate_attempt=attempt + 1,
                        reason=str(exc),
                    )
            generation_seconds = time.perf_counter() - started
            self._timing.word_generation_seconds += generation_seconds
            candidate.unlink(missing_ok=True)
            if preparation is None:
                prepared.unlink(missing_ok=True)
                reasons = "; ".join(failures)
                raise EngineError(
                    f"Could not generate a valid {language} word clip for {story.story_id} "
                    f"after {self.config.max_generation_attempts} candidates: {reasons}"
                )
            cached = self.cache.put(
                key,
                prepared,
                {
                    **cache_payload,
                    "text_length": len(text),
                    "generation_seconds": generation_seconds,
                    "accepted_candidate": accepted_attempt,
                    "rejected_candidates": failures,
                    "original_duration_seconds": preparation.original_duration_seconds,
                    "active_duration_seconds": preparation.active_duration_seconds,
                    "leading_trim_seconds": preparation.leading_trim_seconds,
                    "trailing_trim_seconds": preparation.trailing_trim_seconds,
                },
            )
            prepared.unlink(missing_ok=True)
            self._record_event(
                run_root,
                "chunk_generated",
                story_id=story.story_id,
                language=language,
                asset_type="word",
                chunk_index=1,
                cache_key=key,
                generation_seconds=cached.metadata["generation_seconds"],
                accepted_candidate=accepted_attempt,
            )
        else:
            cache_hits = 1
            self._record_event(
                run_root,
                "chunk_cache_hit",
                story_id=story.story_id,
                language=language,
                asset_type="word",
                chunk_index=1,
                cache_key=key,
            )
        self.cache.materialize(cached, job_chunk)
        output_path = staging_root / story.story_id / "word" / f"{language}.wav"
        final_validation = assemble_wavs(
            [job_chunk],
            output_path,
            self.config.chunk_silence_ms,
            self.config.peak_dbfs,
            self.config.sample_rate,
        )
        final_validation = validate_wav(output_path, text_length=len(text))
        generation_seconds = float(cached.metadata["generation_seconds"])
        metadata = {
            **cached.metadata,
            "text": text,
            "chunk_index": 1,
            "cache_hit": cached.hit,
            "job_path": str(job_chunk.relative_to(run_root)),
        }
        return (
            {
                "path": f"{story.story_id}/word/{language}.wav",
                "sha256": final_validation.sha256,
                "duration_seconds": round(final_validation.duration_seconds, 6),
                "sample_rate": final_validation.sample_rate,
                "source_text": text,
                "source_text_hash": source_text_hash,
                "reference_language": reference_language,
                "reference_audio_hash": reference.audio_hash,
                "generation_seconds": round(generation_seconds, 6),
                "real_time_factor": round(
                    real_time_factor(generation_seconds, final_validation.duration_seconds), 6
                ),
                "warnings": list(final_validation.warnings),
                "chunks": [metadata],
            },
            cache_hits,
            generated_chunks,
        )

    def _plan_story_chunks(
        self,
        run_root: Path,
        stories: tuple[Story, ...],
        languages: tuple[str, ...],
        references: dict[str, VoiceReference],
    ) -> dict[tuple[str, str], list[PlannedChunk]]:
        """Resolve all cache entries before any story synthesis; keep original boundaries."""
        planned: dict[tuple[str, str], list[PlannedChunk]] = {}
        cached_by_key: dict[str, CacheResult | None] = {}
        for story in stories:
            for language in languages:
                reference_language = "zh" if language == "zh" and "zh" in references else "en"
                reference = references[reference_language]
                jobs = []
                for index, text in enumerate(
                    chunk_text(story.texts[language], language, self.config.max_chunk_characters),
                    start=1,
                ):
                    # This payload is deliberately identical to the pre-batching key format.
                    payload = {
                        "cache_schema_version": 1,
                        "chunking_version": CHUNKING_VERSION,
                        "engine": self.engine.name,
                        "model_name": self.engine.model_name,
                        "model_revision": self.engine.model_revision,
                        "generation_parameters": self.config.generation_parameters(),
                        "reference_language": reference_language,
                        "reference_audio_hash": reference.audio_hash,
                        "reference_transcript_hash": reference.transcript_hash,
                        "language": language,
                        "text": text,
                        "chunk_index": index,
                    }
                    key = chunk_cache_key(payload)
                    if key not in cached_by_key:
                        cached_by_key[key] = self.cache.get(key)
                    jobs.append(
                        PlannedChunk(
                            story.story_id,
                            key,
                            run_root
                            / "chunks"
                            / story.story_id
                            / language
                            / f"chunk_{index:04d}.wav",
                            payload,
                            cached_by_key[key],
                        )
                    )
                planned[story.story_id, language] = jobs
        return planned

    def _materialize_story_chunk(
        self,
        run_root: Path,
        job: PlannedChunk,
        cached: CacheResult,
        *,
        cache_source: str | None = None,
    ) -> None:
        # Story identity belongs to the destination, not the shared cache key.
        job.cached = CacheResult(
            cached.hit,
            cached.wav_path,
            {
                **cached.metadata,
                "story_id": job.story_id,
            },
        )
        self.cache.materialize(job.cached, job.job_path)
        self._record_event(
            run_root,
            "chunk_cache_hit" if cached.hit else "chunk_generated",
            story_id=job.story_id,
            language=job.payload["language"],
            chunk_index=job.payload["chunk_index"],
            cache_key=job.key,
            generation_seconds=cached.metadata["generation_seconds"],
            cache_source=cache_source,
        )

    def _generate_chunk_batch(
        self,
        run_root: Path,
        jobs: list[PlannedChunk],
        results: dict[str, CacheResult],
        *,
        on_generated: Callable[[PlannedChunk, CacheResult], None] | None = None,
        parent_batch_id: str | None = None,
    ) -> None:
        """Commit chunks individually, splitting failures without losing successes."""
        if not jobs:
            return
        compatibility_key = jobs[0].compatibility_key
        if any(job.compatibility_key != compatibility_key for job in jobs):
            raise EngineError("Cannot batch chunks with different languages, prompts or settings")
        language = jobs[0].payload["language"]
        reference_language = jobs[0].payload["reference_language"]
        batch_id = uuid.uuid4().hex
        fields = {
            "batch_id": batch_id,
            "parent_batch_id": parent_batch_id,
            "configured_batch_size": self.config.qwen_batch_size,
            "batch_size": len(jobs),
            "actual_batch_size": len(jobs),
            "story_ids": [job.story_id for job in jobs],
            "chunk_indices": [job.payload.get("chunk_index") for job in jobs],
            "language": language,
            "reference_language": reference_language,
            "compatibility_key": compatibility_key,
            "cache_keys": [job.key for job in jobs],
        }
        # Callback/cancellation exceptions stay outside the engine retry scope.
        self._record_event(run_root, "batch_started", **fields)
        started = time.perf_counter()
        failure = None
        try:
            if len(jobs) == 1:
                job = jobs[0]
                self.engine.generate(
                    job.payload["text"], language, reference_language, job.generated_path
                )
            else:
                self.engine.generate_batch(
                    [job.payload["text"] for job in jobs],
                    language,
                    reference_language,
                    [job.generated_path for job in jobs],
                )
        except RuntimeError as exc:
            failure = str(exc)
        elapsed = time.perf_counter() - started
        self._timing.story_batch_seconds += elapsed
        self._timing.batch_attempts += 1
        self._record_event(
            run_root,
            "batch_finished",
            **fields,
            batch_wall_seconds=elapsed,
            error=failure,
        )
        retry = []
        for position, job in enumerate(jobs):
            cached = None
            try:
                if failure is not None:
                    raise EngineError(failure)
                cached = self.cache.put(
                    job.key,
                    job.generated_path,
                    {
                        **job.payload,
                        "text_length": len(job.payload["text"]),
                        "generation_seconds": elapsed / len(jobs),
                        "batch_id": batch_id,
                        "batch_size": len(jobs),
                        "batch_index": position,
                        "batch_wall_seconds": elapsed,
                        "generation_time_allocation": "equal_share",
                    },
                )
                results[job.key] = cached
            except (AudioValidationError, EngineError) as exc:
                retry.append(job)
                failure_reason = str(exc)
                self._record_event(
                    run_root,
                    "batch_chunk_rejected",
                    batch_id=batch_id,
                    story_id=job.story_id,
                    cache_key=job.key,
                    chunk_index=job.payload.get("chunk_index"),
                    language=language,
                    reason=failure_reason,
                )
            finally:
                job.generated_path.unlink(missing_ok=True)
            if cached is not None and on_generated is not None:
                on_generated(job, cached)
        if retry:
            if len(jobs) == 1:
                raise EngineError(failure_reason)
            midpoint = max(1, len(retry) // 2)
            smaller_batches = [part for part in (retry[:midpoint], retry[midpoint:]) if part]
            self._timing.fallback_count += 1
            self._record_event(
                run_root,
                "batch_fallback",
                **fields,
                reason=failure or "One or more chunks failed audio validation",
                retry_cache_keys=[job.key for job in retry],
                smaller_batch_sizes=[len(part) for part in smaller_batches],
            )
            # Retry every half even if a sibling remains irrecoverable.
            errors = []
            for smaller in smaller_batches:
                try:
                    self._generate_chunk_batch(
                        run_root,
                        smaller,
                        results,
                        on_generated=on_generated,
                        parent_batch_id=batch_id,
                    )
                except EngineError as exc:
                    errors.append(str(exc))
            if errors:
                raise EngineError("; ".join(errors))

    def generate(self, request: GenerateRequest) -> GenerationResult:
        run_started = time.perf_counter()
        self._timing = GenerationTiming()
        stories = self._validate_request(request)
        references = self._prepare_references(request)
        run_id = request.run_id or f"run_{uuid.uuid4().hex[:16]}"
        if not SAFE_ID.fullmatch(run_id):
            raise PipelineError(
                "run-id must be 1–128 safe letters, digits, hyphens, or underscores"
            )
        run_root = self.config.output_root / "jobs" / run_id
        request_path = run_root / "request.json"

        request_data: dict[str, Any] = {
            "voice": {
                "voice_id": request.voice_id,
                "voice_name": request.voice_name.strip(),
                "voice_version": request.voice_version,
            },
            "library": {
                "library_id": request.library.library_id,
                "library_version": request.library.version,
                "manifest_path": str(request.library.manifest_path),
            },
            "stories": [
                {
                    "story_id": story.story_id,
                    "story_version": story.version,
                    "word_text_hashes": {
                        language: sha256_text(story.spoken_word[language])
                        for language in request.languages
                    },
                    "source_hashes": {
                        language: story.source_hashes[language] for language in request.languages
                    },
                }
                for story in stories
            ],
            "languages": list(request.languages),
            "references": {
                language: {
                    "audio_hash": reference.audio_hash,
                    "transcript_hash": reference.transcript_hash,
                }
                for language, reference in references.items()
            },
            "engine": {
                "name": self.engine.name,
                "model_name": self.engine.model_name,
                "model_revision": self.engine.model_revision,
            },
            "parameters": self.config.generation_parameters(),
        }
        fingerprint = stable_json_hash(request_data)
        pack_id: str
        created_at: str
        if request_path.exists():
            if not request.resume:
                raise ResumeError(f"Run {run_id} already exists; use --resume to continue it")
            try:
                previous = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ResumeError(f"Cannot read previous run metadata: {exc}") from exc
            if previous.get("fingerprint") != fingerprint:
                raise ResumeError("Run inputs changed; start a new run instead of resuming")
            if previous.get("status") == "completed":
                raise ResumeError(f"Run {run_id} is already completed")
            pack_id = previous["pack_id"]
            created_at = previous["created_at"]
        else:
            if request.resume:
                raise ResumeError(f"Cannot resume missing run {run_id}")
            pack_id = f"pack_{uuid.uuid4().hex[:16]}"
            created_at = utc_now()

        state = {
            **request_data,
            "fingerprint": fingerprint,
            "run_id": run_id,
            "pack_id": pack_id,
            "created_at": created_at,
            "status": "generating",
            "qwen_batch_size": self.config.qwen_batch_size,
        }
        _write_json(request_path, state)
        self._record_event(
            run_root,
            "run_started" if not request.resume else "run_resumed",
            run_id=run_id,
            pack_id=pack_id,
            fingerprint=fingerprint,
            configured_batch_size=self.config.qwen_batch_size,
        )
        pack_root = self.config.output_root / "packs" / pack_id
        staging_root = self.config.output_root / "packs" / f".{pack_id}.staging"
        if pack_root.exists():
            raise ResumeError(f"Completed pack already exists: {pack_root}")
        staging_root.mkdir(parents=True, exist_ok=True)

        try:
            planned = self._plan_story_chunks(run_root, stories, request.languages, references)
            targets: dict[str, list[PlannedChunk]] = {}
            groups: dict[str, list[PlannedChunk]] = {}
            cache_hits = 0
            generated_chunks = 0
            for jobs in planned.values():
                for job in jobs:
                    if job.cached is not None:
                        self._materialize_story_chunk(
                            run_root, job, job.cached, cache_source="existing_cache"
                        )
                        cache_hits += 1
                    elif job.key in targets:
                        targets[job.key].append(job)
                    else:
                        targets[job.key] = [job]
                        groups.setdefault(job.compatibility_key, []).append(job)
            self._record_event(
                run_root,
                "chunks_planned",
                configured_batch_size=self.config.qwen_batch_size,
                total_story_chunks=sum(len(jobs) for jobs in planned.values()),
                cache_hits=cache_hits,
                uncached_unique_chunks=len(targets),
                shared_chunk_destinations=sum(len(jobs) - 1 for jobs in targets.values()),
                compatibility_groups=len(groups),
            )
            for reference in references.values():
                self.engine.prepare_reference(reference)

            def completed(job: PlannedChunk, cached: CacheResult) -> None:
                nonlocal cache_hits, generated_chunks
                for index, target in enumerate(targets[job.key]):
                    reused = index > 0
                    result = CacheResult(reused, cached.wav_path, cached.metadata)
                    self._materialize_story_chunk(
                        run_root, target, result, cache_source="job_reuse" if reused else None
                    )
                    cache_hits += int(reused)
                    generated_chunks += int(not reused)

            results: dict[str, CacheResult] = {}
            failures = []
            # Dict insertion order follows requested story/language/chunk order.
            for jobs in groups.values():
                for start in range(0, len(jobs), self.config.qwen_batch_size):
                    try:
                        self._generate_chunk_batch(
                            run_root,
                            jobs[start : start + self.config.qwen_batch_size],
                            results,
                            on_generated=completed,
                        )
                    except EngineError as exc:
                        failures.append(str(exc))
            self._record_event(
                run_root,
                "story_synthesis_finished",
                cache_hits=cache_hits,
                generated_chunks=generated_chunks,
                **self._timing.metadata(run_started),
            )
            if failures:
                raise EngineError("; ".join(failures))
            story_entries: list[dict[str, Any]] = []
            for story in stories:
                audio_entries: dict[str, Any] = {}
                word_audio_entries: dict[str, Any] = {}
                for language in request.languages:
                    reference_language = "zh" if language == "zh" and "zh" in references else "en"
                    reference = references[reference_language]
                    job_chunk_paths: list[Path] = []
                    chunk_metadata: list[dict[str, Any]] = []
                    for job in planned[story.story_id, language]:
                        cached = job.cached
                        if cached is None:
                            raise EngineError(f"Missing generated chunk: {job.job_path}")
                        job_chunk_paths.append(job.job_path)
                        chunk_metadata.append(
                            {
                                **cached.metadata,
                                "story_id": job.story_id,
                                "text": job.payload["text"],
                                "chunk_index": job.payload["chunk_index"],
                                "cache_hit": cached.hit,
                                "job_path": str(job.job_path.relative_to(run_root)),
                            }
                        )

                    output_path = staging_root / story.story_id / f"{language}.wav"
                    final_validation = assemble_wavs(
                        job_chunk_paths,
                        output_path,
                        self.config.chunk_silence_ms,
                        self.config.peak_dbfs,
                        self.config.sample_rate,
                    )
                    final_validation = validate_wav(
                        output_path, text_length=len(story.texts[language])
                    )
                    generation_seconds = sum(
                        float(item["generation_seconds"]) for item in chunk_metadata
                    )
                    audio_entries[language] = {
                        "path": f"{story.story_id}/{language}.wav",
                        "sha256": final_validation.sha256,
                        "duration_seconds": round(final_validation.duration_seconds, 6),
                        "sample_rate": final_validation.sample_rate,
                        "source_text_hash": story.source_hashes[language],
                        "reference_language": reference_language,
                        "reference_audio_hash": reference.audio_hash,
                        "generation_seconds": round(generation_seconds, 6),
                        "real_time_factor": round(
                            real_time_factor(generation_seconds, final_validation.duration_seconds),
                            6,
                        ),
                        "warnings": list(final_validation.warnings),
                        "chunks": chunk_metadata,
                    }
                    word_audio, word_cache_hits, word_generated_chunks = self._generate_word_audio(
                        run_root,
                        staging_root,
                        story,
                        language,
                        reference_language,
                        reference,
                    )
                    word_audio_entries[language] = word_audio
                    cache_hits += word_cache_hits
                    generated_chunks += word_generated_chunks
                story_entries.append(
                    {
                        "story_id": story.story_id,
                        "story_version": story.version,
                        "title": story.title,
                        "spoken_word": story.spoken_word,
                        "trigger_objects": list(story.trigger_objects),
                        "audio": audio_entries,
                        "word_audio": word_audio_entries,
                    }
                )

            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "pack_id": pack_id,
                "status": "ready_for_review",
                "created_at": created_at,
                "approved_at": None,
                "voice": {
                    "voice_id": request.voice_id,
                    "voice_version": request.voice_version,
                    "name": request.voice_name.strip(),
                    "references": {
                        language: {
                            "audio_sha256": reference.audio_hash,
                            "transcript_sha256": reference.transcript_hash,
                            "duration_seconds": round(reference.duration_seconds, 6),
                            "warnings": list(reference.warnings),
                        }
                        for language, reference in references.items()
                    },
                },
                "library": {
                    "library_id": request.library.library_id,
                    "library_version": request.library.version,
                },
                "generation": {
                    "generation_version": 2,
                    "qwen_batch_size": self.config.qwen_batch_size,
                    "engine": self.engine.name,
                    "model_name": self.engine.model_name,
                    "model_revision": self.engine.model_revision,
                    "seed": self.config.seed,
                    "parameters": self.config.generation_parameters(),
                    "runtime": self.engine.runtime_metadata(),
                    "timing": self._timing.metadata(run_started),
                },
                "stories": story_entries,
            }
            write_manifest(staging_root / "manifest.json", manifest)
            os.replace(staging_root, pack_root)
            state.update(
                {
                    "status": "completed",
                    "completed_at": utc_now(),
                    "cache_hits": cache_hits,
                    "generated_chunks": generated_chunks,
                    "pack_path": str(pack_root),
                    "timing": self._timing.metadata(run_started),
                }
            )
            _write_json(request_path, state)
            self._record_event(
                run_root,
                "run_completed",
                pack_id=pack_id,
                cache_hits=cache_hits,
                generated_chunks=generated_chunks,
                configured_batch_size=self.config.qwen_batch_size,
                **self._timing.metadata(run_started),
            )
            return GenerationResult(
                run_id=run_id,
                pack_id=pack_id,
                pack_path=pack_root,
                manifest_path=pack_root / "manifest.json",
                cache_hits=cache_hits,
                generated_chunks=generated_chunks,
            )
        except Exception as exc:
            state.update(
                {
                    "status": "failed",
                    "failed_at": utc_now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "timing": self._timing.metadata(run_started),
                }
            )
            _write_json(request_path, state)
            self._record_event(
                run_root,
                "run_failed",
                pack_id=pack_id,
                error=str(exc),
                exception_type=type(exc).__name__,
                **self._timing.metadata(run_started),
            )
            raise
