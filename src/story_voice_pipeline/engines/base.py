"""TTS engine protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import VoiceReference


class TTSEngine(Protocol):
    name: str
    model_name: str
    model_revision: str

    def prepare_reference(self, reference: VoiceReference) -> None:
        """Prepare and retain reusable state for one reference language."""

    def generate(
        self,
        text: str,
        language: str,
        reference_language: str,
        output: Path,
        *,
        max_new_tokens: int | None = None,
        seed_offset: int = 0,
    ) -> float:
        """Generate a WAV and return generation time in seconds."""

    def runtime_metadata(self) -> dict[str, object]:
        """Return reproducibility and hardware metadata."""
