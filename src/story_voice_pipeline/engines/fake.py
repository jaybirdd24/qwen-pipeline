"""Deterministic fake engine used by automated tests."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..models import VoiceReference


class FakeTTSEngine:
    name = "fake"
    model_name = "fake-deterministic-tone"
    model_revision = "1"

    def __init__(self, sample_rate: int = 24_000, seed: int = 42) -> None:
        self.sample_rate = sample_rate
        self.seed = seed
        self._references: set[str] = set()

    def prepare_reference(self, reference: VoiceReference) -> None:
        self._references.add(reference.language)

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
        if reference_language not in self._references:
            raise RuntimeError(f"Reference {reference_language} was not prepared")
        started = time.perf_counter()
        digest = hashlib.sha256(
            f"{self.seed + seed_offset}:{language}:{text}:{max_new_tokens}".encode()
        ).digest()
        frequency = 180 + int.from_bytes(digest[:2], "big") % 260
        chars_per_second = 5 if language == "zh" else 12
        duration = max(0.6, min(8.0, len(text) / chars_per_second))
        times = np.arange(round(duration * self.sample_rate), dtype=np.float32) / self.sample_rate
        envelope = np.minimum(1.0, np.minimum(times * 20, (duration - times) * 20))
        samples = (0.15 * envelope * np.sin(2 * np.pi * frequency * times)).astype(np.float32)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, samples, self.sample_rate, subtype="PCM_16")
        return time.perf_counter() - started

    def generate_batch(
        self,
        texts: list[str],
        language: str,
        reference_language: str,
        outputs: list[Path],
        *,
        max_new_tokens: int | None = None,
        seed_offset: int = 0,
    ) -> float:
        """Write ordered WAVs for one language/reference pair; return total wall seconds."""
        if not texts or len(texts) != len(outputs):
            raise ValueError("Batch texts and outputs must have equal, nonzero length")
        started = time.perf_counter()
        for text, output in zip(texts, outputs, strict=True):
            self.generate(
                text,
                language,
                reference_language,
                output,
                max_new_tokens=max_new_tokens,
                seed_offset=seed_offset,
            )
        return time.perf_counter() - started

    def runtime_metadata(self) -> dict[str, object]:
        return {"engine": self.name, "sample_rate": self.sample_rate, "seed": self.seed}
