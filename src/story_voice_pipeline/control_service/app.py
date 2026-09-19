"""FastAPI control service, JSON API, and server-rendered researcher pages."""

import json
import shutil
import uuid
import zipfile
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from ..audio import normalize_reference
from ..errors import PipelineError
from ..manifest import load_manifest, write_manifest
from ..models import LANGUAGE_NAMES, SUPPORTED_LANGUAGES
from ..story_library import load_story_library
from .config import ControlSettings
from .database import Database
from .models import GeneratedAudio, GenerationJob, StoryPack, Voice, VoiceReference, now_utc
from .passages import RECORDING_PASSAGES
from .services import LocalJobProcessor

PACKAGE_ROOT = Path(__file__).parent


class JobCreate(BaseModel):
    voice_id: str
    story_ids: list[str] | None = None
    languages: list[str] | None = None


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _voice_json(voice: Voice, include_private: bool = False) -> dict[str, Any]:
    result = {
        "id": voice.id,
        "name": voice.name,
        "created_at": _iso(voice.created_at),
        "voice_version": voice.voice_version,
        "english_reference_duration": voice.english_reference_duration,
        "has_mandarin_reference": voice.mandarin_reference_path is not None,
        "mandarin_reference_duration": voice.mandarin_reference_duration,
        "consent_confirmed": voice.consent_confirmed,
        "notes": voice.notes,
        "languages": [ref.language_code for ref in voice.references],
        "references": {
            ref.language_code: {
                "duration_seconds": ref.duration_seconds,
                "audio_sha256": ref.audio_sha256,
                **({"transcript": ref.transcript} if include_private else {}),
            }
            for ref in voice.references
        },
    }
    if include_private:
        result.update(
            {
                "english_reference_transcript": voice.english_reference_transcript,
                "mandarin_reference_transcript": voice.mandarin_reference_transcript,
            }
        )
    return result


def _job_json(job: GenerationJob) -> dict[str, Any]:
    progress = job.completed_chunks / job.total_chunks if job.total_chunks else 0.0
    elapsed_seconds = None
    if job.started_at is not None:
        started_at = job.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        ended_at = job.completed_at or datetime.now(timezone.utc)
        if ended_at.tzinfo is None:
            ended_at = ended_at.replace(tzinfo=timezone.utc)
        elapsed_seconds = max(0.0, (ended_at - started_at).total_seconds())
    return {
        "id": job.id,
        "voice_id": job.voice_id,
        "library_id": job.library_id,
        "library_version": job.library_version,
        "story_ids": json.loads(job.story_ids_json),
        "languages": json.loads(job.languages_json),
        "status": job.status,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "completed_at": _iso(job.completed_at),
        "elapsed_seconds": elapsed_seconds,
        "current_story_id": job.current_story_id,
        "current_language": job.current_language,
        "current_chunk": job.current_chunk,
        "total_chunks": job.total_chunks,
        "completed_chunks": job.completed_chunks,
        "progress": progress,
        "error_message": job.error_message,
        "run_id": job.run_id,
        "pack_id": job.pack_id,
    }


def _pack_json(pack: StoryPack, audios: list[GeneratedAudio]) -> dict[str, Any]:
    return {
        "id": pack.id,
        "job_id": pack.job_id,
        "voice_id": pack.voice_id,
        "voice_version": pack.voice_version,
        "library_id": pack.library_id,
        "library_version": pack.library_version,
        "generation_version": pack.generation_version,
        "model_name": pack.model_name,
        "model_revision": pack.model_revision,
        "status": pack.status,
        "created_at": _iso(pack.created_at),
        "approved_at": _iso(pack.approved_at),
        "audio": [
            {
                "story_id": audio.story_id,
                "story_version": audio.story_version,
                "language": audio.language,
                "audio_type": "word" if "/word/" in audio.audio_path else "story",
                "path": audio.audio_path,
                "sha256": audio.audio_hash,
                "duration_seconds": audio.duration_seconds,
                "sample_rate": audio.sample_rate,
                "generation_seconds": audio.generation_seconds,
                "real_time_factor": audio.real_time_factor,
                "validation_status": audio.validation_status,
                "warnings": json.loads(audio.warnings_json),
            }
            for audio in audios
        ],
    }


async def _save_upload(upload: UploadFile, path: Path, maximum: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with path.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise PipelineError(f"Upload exceeds the {maximum}-byte limit")
                handle.write(chunk)
    finally:
        await upload.close()
    if size == 0:
        raise PipelineError("Uploaded audio is empty")


def create_app(settings: ControlSettings | None = None) -> FastAPI:
    settings = settings or ControlSettings.from_environment()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    database = Database(settings.resolved_database_url)
    processor = LocalJobProcessor(settings, database)
    templates = Jinja2Templates(directory=PACKAGE_ROOT / "templates")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.create_schema(settings.data_root)
        yield
        database.dispose()

    app = FastAPI(title="Story Voice Control Service", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.database = database
    app.state.processor = processor
    app.mount("/static", StaticFiles(directory=PACKAGE_ROOT / "static"), name="static")

    def get_session() -> Session:
        with database.sessions() as session:
            yield session

    def data_path(stored: str) -> Path:
        return (settings.data_root / stored).resolve()

    async def create_voice_record(
        session: Session,
        name: str,
        recordings: list[tuple[str, UploadFile, str]],
        consent_confirmed: bool,
        notes: str = "",
    ) -> Voice:
        if not name.strip() or len(name.strip()) > 200:
            raise HTTPException(422, "Voice name is required and must be at most 200 characters")
        if not consent_confirmed:
            raise HTTPException(422, "Consent confirmation is required")
        codes = [code for code, _, _ in recordings]
        if not codes or len(codes) != len(set(codes)):
            raise HTTPException(422, "Choose distinct reference languages")
        if any(code not in SUPPORTED_LANGUAGES for code in codes):
            raise HTTPException(422, "Invalid reference language")
        voice_id = f"voice_{uuid.uuid4().hex[:16]}"
        voice_root = settings.data_root / "references" / voice_id / "v1"
        upload_root = settings.data_root / "uploads" / voice_id
        voice = Voice(id=voice_id, name=name.strip(), consent_confirmed=True, notes=notes.strip())
        try:
            for code, upload, transcript in recordings:
                source = upload_root / f"{code}{Path(upload.filename or '').suffix.lower()}"
                await _save_upload(upload, source, settings.max_upload_bytes)
                reference = normalize_reference(
                    source,
                    transcript,
                    voice_root / code / "reference.wav",
                    code,
                    settings.pipeline_config.sample_rate,
                )
                (voice_root / code / "reference.txt").write_text(
                    reference.transcript + "\n", encoding="utf-8"
                )
                stored = str(
                    reference.audio_path.resolve().relative_to(settings.data_root.resolve())
                )
                voice.references.append(
                    VoiceReference(
                        language_code=code,
                        audio_path=stored,
                        transcript=reference.transcript,
                        duration_seconds=reference.duration_seconds,
                        audio_sha256=reference.audio_hash,
                    )
                )
                # Compatibility fields for existing API clients and databases.
                prefix = {"en": "english", "zh": "mandarin"}.get(code)
                if prefix:
                    setattr(voice, f"{prefix}_reference_path", stored)
                    setattr(voice, f"{prefix}_reference_transcript", reference.transcript)
                    setattr(voice, f"{prefix}_reference_duration", reference.duration_seconds)
            session.add(voice)
            session.commit()
        except Exception as exc:
            session.rollback()
            shutil.rmtree(voice_root.parent, ignore_errors=True)
            if isinstance(exc, PipelineError):
                raise HTTPException(422, str(exc)) from exc
            raise
        finally:
            shutil.rmtree(upload_root, ignore_errors=True)
        return voice

    def create_job_record(session: Session, payload: JobCreate) -> GenerationJob:
        voice = session.get(Voice, payload.voice_id)
        if voice is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Voice not found")
        library = load_story_library(settings.story_library_path)
        available = {story.story_id for story in library.stories}
        stories = (
            payload.story_ids
            if payload.story_ids is not None
            else [story.story_id for story in library.stories]
        )
        languages = (
            payload.languages
            if payload.languages is not None
            else [ref.language_code for ref in voice.references]
        )
        if (
            not stories
            or len(stories) != len(set(stories))
            or any(story not in available for story in stories)
        ):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid story selection")
        if (
            not languages
            or len(languages) != len(set(languages))
            or any(language not in SUPPORTED_LANGUAGES for language in languages)
        ):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid language selection")
        unavailable = [
            language for language in languages if language not in library.required_languages
        ]
        if unavailable:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "Story library has no complete translation for: " + ", ".join(unavailable),
            )
        missing = set(languages) - {ref.language_code for ref in voice.references}
        if missing:
            raise HTTPException(422, "Missing language reference: " + ", ".join(sorted(missing)))
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        job = GenerationJob(
            id=job_id,
            voice_id=voice.id,
            library_id=library.library_id,
            library_version=library.version,
            story_ids_json=json.dumps(stories),
            languages_json=json.dumps(languages),
            status="QUEUED",
            run_id=f"run_{uuid.uuid4().hex[:16]}",
        )
        session.add(job)
        session.commit()
        return job

    def get_pack_or_404(session: Session, pack_id: str) -> StoryPack:
        pack = session.get(StoryPack, pack_id)
        if pack is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Story pack not found")
        return pack

    def pack_file(pack: StoryPack, file_path: str, approved_only: bool) -> Path:
        if approved_only and pack.status != "APPROVED":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Approved story pack not found")
        manifest = load_manifest(data_path(pack.manifest_path))
        allowed = {
            audio["path"]
            for story in manifest["stories"]
            for group in (story["audio"], story.get("word_audio", {}))
            for audio in group.values()
        }
        requested = Path(file_path)
        if requested.is_absolute() or ".." in requested.parts or file_path not in allowed:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Pack file not found")
        path = data_path(pack.pack_path) / requested
        if not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Pack file not found")
        return path

    @app.get("/api/v1/voices")
    def list_voices(session: Annotated[Session, Depends(get_session)]) -> list[dict[str, Any]]:
        voices = session.scalars(select(Voice).order_by(Voice.created_at.desc())).all()
        return [_voice_json(voice) for voice in voices]

    @app.get("/api/v1/languages")
    def list_languages() -> list[dict[str, Any]]:
        library = load_story_library(settings.story_library_path)
        return [
            {
                "code": code,
                "name": LANGUAGE_NAMES[code],
                "available": code in library.required_languages,
                "experimental": False,
            }
            for code in SUPPORTED_LANGUAGES
        ]

    @app.get("/api/v1/voices/{voice_id}")
    def get_voice(voice_id: str, session: Annotated[Session, Depends(get_session)]):
        voice = session.get(Voice, voice_id)
        if voice is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Voice not found")
        return _voice_json(voice, include_private=True)

    @app.post("/api/v1/voices", status_code=status.HTTP_201_CREATED)
    async def create_voice_api(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, Any]:
        form = await request.form()
        recordings = []
        codes = form.getlist("languages")
        audios = form.getlist("audios")
        if codes or audios:
            if any(
                key in form
                for key in (
                    "english_audio",
                    "english_transcript",
                    "mandarin_audio",
                    "mandarin_transcript",
                )
            ):
                raise HTTPException(422, "Do not mix legacy and language-specific reference fields")
            if len(codes) != len(audios):
                raise HTTPException(422, "Supply one audio for each language")
            for code, audio in zip(codes, audios, strict=True):
                if code not in RECORDING_PASSAGES or not hasattr(audio, "read"):
                    raise HTTPException(422, "Invalid language or recording")
                recordings.append((str(code), audio, RECORDING_PASSAGES[code]))
        else:
            for code, prefix in (("en", "english"), ("zh", "mandarin")):
                audio = form.get(f"{prefix}_audio")
                transcript = str(form.get(f"{prefix}_transcript", "")).strip()
                if bool(audio and getattr(audio, "filename", "")) != bool(transcript):
                    raise HTTPException(422, "Audio and transcript must be supplied together")
                if audio and getattr(audio, "filename", ""):
                    recordings.append((code, audio, transcript))
        voice = await create_voice_record(
            session,
            str(form.get("name", "")),
            recordings,
            str(form.get("consent_confirmed", "")).lower() in {"true", "1", "on"},
            str(form.get("notes", "")),
        )
        return _voice_json(voice, include_private=True)

    @app.post("/api/v1/references/validate")
    async def validate_recording(
        language: Annotated[str, Form()],
        audio: Annotated[UploadFile, File()],
    ):
        if language not in RECORDING_PASSAGES:
            raise HTTPException(422, "Invalid reference language")
        work = settings.data_root / "uploads" / f"validation_{uuid.uuid4().hex}"
        try:
            source = work / f"source{Path(audio.filename or '').suffix.lower()}"
            await _save_upload(audio, source, settings.max_upload_bytes)
            reference = normalize_reference(
                source,
                RECORDING_PASSAGES[language],
                work / "reference.wav",
                language,
                settings.pipeline_config.sample_rate,
            )
            return {"duration_seconds": reference.duration_seconds, "warnings": reference.warnings}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @app.post("/api/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
    def create_job_api(
        payload: JobCreate,
        background_tasks: BackgroundTasks,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, Any]:
        job = create_job_record(session, payload)
        background_tasks.add_task(processor.process, job.id)
        return _job_json(job)

    @app.get("/api/v1/jobs/{job_id}")
    def get_job_api(job_id: str, session: Annotated[Session, Depends(get_session)]):
        job = session.get(GenerationJob, job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
        return _job_json(job)

    @app.post("/api/v1/jobs/{job_id}/retry", status_code=status.HTTP_202_ACCEPTED)
    def retry_job_api(
        job_id: str,
        background_tasks: BackgroundTasks,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, Any]:
        job = session.get(GenerationJob, job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
        if job.status not in {"FAILED", "CANCELLED"}:
            raise HTTPException(status.HTTP_409_CONFLICT, "Only failed or cancelled jobs can retry")
        job.status = "QUEUED"
        job.cancel_requested = False
        job.error_message = None
        session.commit()
        background_tasks.add_task(processor.process, job.id)
        return _job_json(job)

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job_api(job_id: str, session: Annotated[Session, Depends(get_session)]):
        job = session.get(GenerationJob, job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
        if job.status == "QUEUED":
            job.status = "CANCELLED"
            job.completed_at = now_utc()
        elif job.status in {"VALIDATING_REFERENCE", "PREPARING_VOICE", "GENERATING"}:
            job.cancel_requested = True
            job.status = "CANCELLING"
        else:
            raise HTTPException(status.HTTP_409_CONFLICT, "Job cannot be cancelled in this state")
        session.commit()
        return _job_json(job)

    @app.get("/api/v1/story-packs")
    def list_packs(session: Annotated[Session, Depends(get_session)]) -> list[dict[str, Any]]:
        packs = session.scalars(select(StoryPack).order_by(StoryPack.created_at.desc())).all()
        return [
            _pack_json(
                pack,
                list(
                    session.scalars(
                        select(GeneratedAudio).where(GeneratedAudio.story_pack_id == pack.id)
                    ).all()
                ),
            )
            for pack in packs
        ]

    @app.get("/api/v1/story-packs/{pack_id}")
    def get_pack_api(pack_id: str, session: Annotated[Session, Depends(get_session)]):
        pack = get_pack_or_404(session, pack_id)
        audios = list(
            session.scalars(
                select(GeneratedAudio).where(GeneratedAudio.story_pack_id == pack.id)
            ).all()
        )
        return _pack_json(pack, audios)

    @app.post("/api/v1/story-packs/{pack_id}/approve")
    def approve_pack_api(pack_id: str, session: Annotated[Session, Depends(get_session)]):
        pack = get_pack_or_404(session, pack_id)
        if pack.status != "READY_FOR_REVIEW":
            raise HTTPException(status.HTTP_409_CONFLICT, "Pack is not ready for review")
        approved_at = now_utc()
        manifest_path = data_path(pack.manifest_path)
        manifest = load_manifest(manifest_path)
        previous_manifest = deepcopy(manifest)
        manifest["status"] = "approved"
        manifest["approved_at"] = _iso(approved_at)
        write_manifest(manifest_path, manifest)
        pack.status = "APPROVED"
        pack.approved_at = approved_at
        job = session.get(GenerationJob, pack.job_id)
        if job is not None:
            job.status = "APPROVED"
        try:
            session.commit()
        except Exception:
            session.rollback()
            write_manifest(manifest_path, previous_manifest)
            raise
        audios = list(
            session.scalars(
                select(GeneratedAudio).where(GeneratedAudio.story_pack_id == pack.id)
            ).all()
        )
        return _pack_json(pack, audios)

    @app.post("/api/v1/story-packs/{pack_id}/reject")
    def reject_pack_api(pack_id: str, session: Annotated[Session, Depends(get_session)]):
        pack = get_pack_or_404(session, pack_id)
        if pack.status != "READY_FOR_REVIEW":
            raise HTTPException(status.HTTP_409_CONFLICT, "Pack is not ready for review")
        pack.status = "REJECTED"
        job = session.get(GenerationJob, pack.job_id)
        if job is not None:
            job.status = "REJECTED"
        session.commit()
        audios = list(
            session.scalars(
                select(GeneratedAudio).where(GeneratedAudio.story_pack_id == pack.id)
            ).all()
        )
        return _pack_json(pack, audios)

    @app.get("/api/v1/devices/{device_id}/latest-manifest")
    def latest_manifest(device_id: str, session: Annotated[Session, Depends(get_session)]):
        del device_id  # Device assignments are introduced with Pi integration.
        pack = session.scalar(
            select(StoryPack)
            .where(StoryPack.status == "APPROVED")
            .order_by(StoryPack.approved_at.desc())
        )
        if pack is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No approved pack is available")
        return FileResponse(data_path(pack.manifest_path), media_type="application/json")

    @app.get("/api/v1/packs/{pack_id}/files/{file_path:path}")
    def approved_pack_file(
        pack_id: str,
        file_path: str,
        session: Annotated[Session, Depends(get_session)],
    ):
        pack = get_pack_or_404(session, pack_id)
        return FileResponse(pack_file(pack, file_path, approved_only=True), media_type="audio/wav")

    @app.get("/review-files/{pack_id}/{file_path:path}")
    def review_pack_file(
        pack_id: str,
        file_path: str,
        session: Annotated[Session, Depends(get_session)],
    ):
        pack = get_pack_or_404(session, pack_id)
        return FileResponse(pack_file(pack, file_path, approved_only=False), media_type="audio/wav")

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, session: Annotated[Session, Depends(get_session)]):
        voices = session.scalars(select(Voice).order_by(Voice.created_at.desc())).all()
        jobs = session.scalars(
            select(GenerationJob).order_by(GenerationJob.created_at.desc()).limit(20)
        ).all()
        packs = session.scalars(
            select(StoryPack).order_by(StoryPack.created_at.desc()).limit(20)
        ).all()
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "voices": voices,
                "jobs": jobs,
                "packs": packs,
                "language_names": LANGUAGE_NAMES,
            },
        )

    @app.get("/voices/new", response_class=HTMLResponse)
    def new_voice_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="new_voice.html",
            context={"languages": LANGUAGE_NAMES, "passages": RECORDING_PASSAGES},
        )

    @app.get("/references/prepare", response_class=HTMLResponse)
    def prepare_reference_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="prepare_reference.html",
            context={"error": None, "languages": LANGUAGE_NAMES},
        )

    @app.post("/references/prepare")
    async def prepare_reference_form(
        request: Request,
        audio: Annotated[UploadFile, File()],
        transcript: Annotated[str, Form()],
        language: Annotated[str, Form()] = "en",
        consent_confirmed: Annotated[bool, Form()] = False,
    ):
        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid language")
        if not consent_confirmed:
            return templates.TemplateResponse(
                request=request,
                name="prepare_reference.html",
                context={
                    "error": "Please confirm that you have permission to use this voice.",
                    "transcript": transcript,
                    "language": language,
                    "languages": LANGUAGE_NAMES,
                },
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            )

        export_id = f"reference_{uuid.uuid4().hex}"
        work_root = settings.data_root / "reference_exports" / export_id
        source = work_root / f"source{Path(audio.filename or '').suffix.lower()}"
        normalized = work_root / "reference.wav"
        archive = work_root / "qwen-reference.zip"
        try:
            await _save_upload(audio, source, settings.max_upload_bytes)
            reference = normalize_reference(
                source,
                transcript,
                normalized,
                language,
                settings.pipeline_config.sample_rate,
            )
            transcript_path = work_root / "reference.txt"
            transcript_path.write_text(reference.transcript + "\n", encoding="utf-8")
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(normalized, "reference.wav")
                bundle.write(transcript_path, "reference.txt")
        except PipelineError as exc:
            shutil.rmtree(work_root, ignore_errors=True)
            return templates.TemplateResponse(
                request=request,
                name="prepare_reference.html",
                context={
                    "error": str(exc),
                    "transcript": transcript,
                    "language": language,
                    "languages": LANGUAGE_NAMES,
                },
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            )

        return FileResponse(
            archive,
            media_type="application/zip",
            filename="qwen-reference.zip",
            background=BackgroundTask(shutil.rmtree, work_root, ignore_errors=True),
        )

    @app.post("/voices/new")
    async def new_voice_form(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
    ):
        form = await request.form()
        if len(form.getlist("languages")) != 2 or len(form.getlist("audios")) != 2:
            raise HTTPException(422, "Supply recordings for two distinct languages")
        voice = await create_voice_api(request, session)
        return RedirectResponse(
            f"/jobs/new?voice_id={voice['id']}", status_code=status.HTTP_303_SEE_OTHER
        )

    @app.get("/jobs/new", response_class=HTMLResponse)
    def new_job_page(request: Request, session: Annotated[Session, Depends(get_session)]):
        voices = session.scalars(select(Voice).order_by(Voice.name)).all()
        library = load_story_library(settings.story_library_path)
        return templates.TemplateResponse(
            request=request,
            name="new_job.html",
            context={
                "voices": voices,
                "library": library,
                "language_names": LANGUAGE_NAMES,
                "selected_voice_id": request.query_params.get("voice_id"),
                "voice_languages": {
                    voice.id: [ref.language_code for ref in voice.references] for voice in voices
                },
            },
        )

    @app.post("/jobs/new")
    async def new_job_form(
        request: Request,
        background_tasks: BackgroundTasks,
        session: Annotated[Session, Depends(get_session)],
    ):
        form = await request.form()
        payload = JobCreate(
            voice_id=str(form.get("voice_id", "")),
            story_ids=[str(item) for item in form.getlist("story")],
            languages=None,
        )
        job = create_job_record(session, payload)
        background_tasks.add_task(processor.process, job.id)
        return RedirectResponse(f"/jobs/{job.id}", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(
        job_id: str,
        request: Request,
        session: Annotated[Session, Depends(get_session)],
    ):
        job = session.get(GenerationJob, job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
        return templates.TemplateResponse(
            request=request,
            name="job_status.html",
            context={"job": job, "job_data": _job_json(job)},
        )

    @app.post("/jobs/{job_id}/retry")
    def retry_job_form(
        job_id: str,
        background_tasks: BackgroundTasks,
        session: Annotated[Session, Depends(get_session)],
    ):
        retry_job_api(job_id, background_tasks, session)
        return RedirectResponse(f"/jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job_form(job_id: str, session: Annotated[Session, Depends(get_session)]):
        cancel_job_api(job_id, session)
        return RedirectResponse(f"/jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/packs/{pack_id}/review", response_class=HTMLResponse)
    def review_page(
        pack_id: str,
        request: Request,
        session: Annotated[Session, Depends(get_session)],
    ):
        pack = get_pack_or_404(session, pack_id)
        audios = list(
            session.scalars(
                select(GeneratedAudio)
                .where(GeneratedAudio.story_pack_id == pack.id)
                .order_by(GeneratedAudio.story_id, GeneratedAudio.language)
            ).all()
        )
        return templates.TemplateResponse(
            request=request,
            name="review_pack.html",
            context={"pack": pack, "audios": audios, "language_names": LANGUAGE_NAMES},
        )

    @app.post("/packs/{pack_id}/approve")
    def approve_pack_form(pack_id: str, session: Annotated[Session, Depends(get_session)]):
        approve_pack_api(pack_id, session)
        return RedirectResponse(f"/packs/{pack_id}/review", status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/packs/{pack_id}/reject")
    def reject_pack_form(pack_id: str, session: Annotated[Session, Depends(get_session)]):
        reject_pack_api(pack_id, session)
        return RedirectResponse(f"/packs/{pack_id}/review", status_code=status.HTTP_303_SEE_OTHER)

    @app.exception_handler(PipelineError)
    def pipeline_error_handler(_: Request, exc: PipelineError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return app


app = create_app()


def main() -> None:
    settings = ControlSettings.from_environment()
    uvicorn.run(
        "story_voice_pipeline.control_service.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
