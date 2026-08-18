from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from story_voice_pipeline.config import PipelineConfig
from story_voice_pipeline.engines.qwen import QwenTTSEngine, _reached_generation_limit
from story_voice_pipeline.errors import EngineError


class FakeCuda:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed_all(self, seed: int) -> None:
        self.seeds.append(seed)


class FakeTorch:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.cuda = FakeCuda()

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)


class FakeModel:
    def __init__(self, capped_attempts: int, successful_sample_count: int = 1_000) -> None:
        self.capped_attempts = capped_attempts
        self.successful_sample_count = successful_sample_count
        self.calls = 0
        self.languages: list[object] = []
        self.max_new_tokens: list[object] = []

    def generate_voice_clone(self, **kwargs: object) -> tuple[list[np.ndarray], int]:
        self.calls += 1
        self.languages.append(kwargs.get("language"))
        self.max_new_tokens.append(kwargs.get("max_new_tokens"))
        sample_count = 4_792 if self.calls <= self.capped_attempts else self.successful_sample_count
        return [np.full(sample_count, 0.1, dtype=np.float32)], 100


def engine(capped_attempts: int, successful_sample_count: int = 1_000) -> QwenTTSEngine:
    result = QwenTTSEngine.__new__(QwenTTSEngine)
    result.config = PipelineConfig(max_new_tokens=600, max_generation_attempts=3)
    result._prompts = {"en": object()}
    result._torch = FakeTorch()
    result._model = FakeModel(capped_attempts, successful_sample_count)
    return result


def test_generation_limit_detection() -> None:
    assert _reached_generation_limit(4_792, 100, 600)
    assert not _reached_generation_limit(3_000, 100, 600)
    assert not _reached_generation_limit(4_792, 100, None)


def test_qwen_retries_capped_output_with_alternate_seed(tmp_path: Path) -> None:
    tts = engine(capped_attempts=2)
    output = tmp_path / "result.wav"

    elapsed = tts.generate("Test text.", "en", "en", output)

    assert elapsed >= 0
    assert tts._model.calls == 3
    assert tts._torch.seeds == [42, 43, 44]
    samples, sample_rate = sf.read(output)
    assert sample_rate == 100
    assert len(samples) == 1_000


def test_qwen_rejects_output_when_every_attempt_reaches_cap(tmp_path: Path) -> None:
    tts = engine(capped_attempts=3)

    with pytest.raises(EngineError, match="rejecting noisy output"):
        tts.generate("Test text.", "en", "en", tmp_path / "result.wav")

    assert not (tmp_path / "result.wav").exists()


def test_qwen_accepts_word_token_limit_and_seed_offset(tmp_path: Path) -> None:
    tts = engine(capped_attempts=0, successful_sample_count=100)

    tts.generate(
        "Word.",
        "en",
        "en",
        tmp_path / "word.wav",
        max_new_tokens=48,
        seed_offset=10,
    )

    assert tts._model.max_new_tokens == [48]
    assert tts._torch.seeds == [52]


@pytest.mark.parametrize(
    ("language", "qwen_language"),
    [("es", "Spanish"), ("fr", "French"), ("de", "German")],
)
def test_qwen_maps_control_language_codes(
    tmp_path: Path, language: str, qwen_language: str
) -> None:
    tts = engine(capped_attempts=0)

    tts.generate("Text", language, "en", tmp_path / f"{language}.wav")

    assert tts._model.languages == [qwen_language]
