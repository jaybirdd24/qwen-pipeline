"""Phase 2 persistence models."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Voice(Base):
    __tablename__ = "voices"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    voice_version: Mapped[int] = mapped_column(Integer, default=1)
    english_reference_path: Mapped[str] = mapped_column(Text)
    english_reference_transcript: Mapped[str] = mapped_column(Text)
    english_reference_duration: Mapped[float] = mapped_column(Float)
    mandarin_reference_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    mandarin_reference_transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    mandarin_reference_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    consent_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")


class GenerationJob(Base):
    __tablename__ = "generation_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    voice_id: Mapped[str] = mapped_column(ForeignKey("voices.id"), index=True)
    library_id: Mapped[str] = mapped_column(String(128))
    library_version: Mapped[int] = mapped_column(Integer)
    story_ids_json: Mapped[str] = mapped_column(Text)
    languages_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    current_story_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    current_language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    current_chunk: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    completed_chunks: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    pack_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)


class StoryPack(Base):
    __tablename__ = "story_packs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("generation_jobs.id"), unique=True)
    voice_id: Mapped[str] = mapped_column(ForeignKey("voices.id"), index=True)
    voice_version: Mapped[int] = mapped_column(Integer)
    library_id: Mapped[str] = mapped_column(String(128))
    library_version: Mapped[int] = mapped_column(Integer)
    generation_version: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(Text)
    model_revision: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    manifest_path: Mapped[str] = mapped_column(Text)
    pack_path: Mapped[str] = mapped_column(Text)


class GeneratedAudio(Base):
    __tablename__ = "generated_audio"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    story_pack_id: Mapped[str] = mapped_column(ForeignKey("story_packs.id"), index=True)
    story_id: Mapped[str] = mapped_column(String(128))
    story_version: Mapped[int] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(8))
    source_text_hash: Mapped[str] = mapped_column(String(64))
    audio_path: Mapped[str] = mapped_column(Text)
    audio_hash: Mapped[str] = mapped_column(String(64))
    duration_seconds: Mapped[float] = mapped_column(Float)
    sample_rate: Mapped[int] = mapped_column(Integer)
    generation_seconds: Mapped[float] = mapped_column(Float)
    real_time_factor: Mapped[float] = mapped_column(Float)
    validation_status: Mapped[str] = mapped_column(String(40))
    warnings_json: Mapped[str] = mapped_column(Text, default="[]")
