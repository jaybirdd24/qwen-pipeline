"""SQLAlchemy database setup."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from ..hashing import sha256_file
from .models import Base, Voice, VoiceReference


class Database:
    def __init__(self, url: str) -> None:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self, data_root: Path | None = None) -> None:
        Base.metadata.create_all(self.engine)
        columns = {column["name"] for column in inspect(self.engine).get_columns("generation_jobs")}
        if "fallback_reference_language" not in columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE generation_jobs ADD COLUMN "
                        "fallback_reference_language VARCHAR(16)"
                    )
                )
        # Additive, repeatable migration: keep legacy columns for older clients.
        if data_root is None:
            return
        with self.sessions() as session:
            for voice in session.scalars(select(Voice)).all():
                existing = {reference.language_code for reference in voice.references}
                for code, prefix in (("en", "english"), ("zh", "mandarin")):
                    stored = getattr(voice, f"{prefix}_reference_path")
                    if not stored or code in existing:
                        continue
                    path = data_root / stored
                    voice.references.append(
                        VoiceReference(
                            language_code=code,
                            audio_path=stored,
                            transcript=getattr(voice, f"{prefix}_reference_transcript"),
                            duration_seconds=getattr(voice, f"{prefix}_reference_duration"),
                            audio_sha256=sha256_file(path) if path.is_file() else "",
                        )
                    )
            session.commit()

    def session(self) -> Iterator[Session]:
        with self.sessions() as session:
            yield session

    def dispose(self) -> None:
        self.engine.dispose()
