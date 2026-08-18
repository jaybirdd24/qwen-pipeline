from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import write_tone

from story_voice_pipeline.audio import (
    assemble_wavs,
    normalize_reference,
    prepare_word_audio,
    validate_wav,
)
from story_voice_pipeline.errors import AudioValidationError


def test_normalizes_stereo_reference_and_warns_when_short(tmp_path: Path) -> None:
    source = tmp_path / "stereo.wav"
    mono_path = write_tone(tmp_path / "mono.wav", duration=1.0, sample_rate=16_000)
    mono, rate = sf.read(mono_path)
    sf.write(source, np.column_stack([mono, mono * 0.8]), rate)
    reference = normalize_reference(source, " exact transcript ", tmp_path / "normalized.wav", "en")
    samples, output_rate = sf.read(reference.audio_path, always_2d=True)
    assert samples.shape[1] == 1
    assert output_rate == 24_000
    assert reference.transcript == "exact transcript"
    assert reference.warnings


def test_rejects_silent_audio(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(24_000, dtype=np.float32), 24_000)
    with pytest.raises(AudioValidationError, match="silent"):
        validate_wav(path)


def test_assembles_silence_and_normalizes_peak(tmp_path: Path) -> None:
    first = write_tone(tmp_path / "one.wav", duration=0.5)
    second = write_tone(tmp_path / "two.wav", duration=0.5)
    validation = assemble_wavs([first, second], tmp_path / "final.wav", 350, -1.0, 24_000)
    assert validation.duration_seconds == pytest.approx(1.35, abs=0.01)
    assert validation.peak_amplitude == pytest.approx(10 ** (-1 / 20), abs=0.002)


def test_duration_heuristic_is_warning_not_failure(tmp_path: Path) -> None:
    path = write_tone(tmp_path / "short.wav", duration=0.5)
    validation = validate_wav(path, text_length=500)
    assert any("plausible" in warning for warning in validation.warnings)


def test_prepares_word_by_trimming_long_leading_silence(tmp_path: Path) -> None:
    sample_rate = 24_000
    times = np.arange(sample_rate, dtype=np.float32) / sample_rate
    speech = 0.2 * np.sin(2 * np.pi * 220 * times)
    source = tmp_path / "source.wav"
    sf.write(source, np.concatenate([np.zeros(sample_rate * 4), speech]), sample_rate)

    result = prepare_word_audio(source, tmp_path / "word.wav", sample_rate, -1.0)

    assert result.original_duration_seconds == pytest.approx(5.0)
    assert result.leading_trim_seconds == pytest.approx(3.95, abs=0.03)
    assert result.active_duration_seconds == pytest.approx(1.0, abs=0.03)
    assert result.validation.duration_seconds <= 2.0
    assert result.validation.peak_amplitude == pytest.approx(10 ** (-1 / 20), abs=0.002)


def test_rejects_word_with_more_than_two_seconds_active_audio(tmp_path: Path) -> None:
    source = write_tone(tmp_path / "too-long.wav", duration=3.0)
    with pytest.raises(AudioValidationError, match="active audio"):
        prepare_word_audio(source, tmp_path / "word.wav", 24_000, -1.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_normalizes_m4a_reference_with_ffmpeg_fallback(tmp_path: Path) -> None:
    source_wav = write_tone(tmp_path / "source.wav", duration=1.0, sample_rate=44_100)
    source_m4a = tmp_path / "source.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source_wav),
            "-c:a",
            "aac",
            str(source_m4a),
        ],
        check=True,
    )

    reference = normalize_reference(
        source_m4a,
        "exact transcript",
        tmp_path / "normalized.wav",
        "en",
    )

    info = sf.info(reference.audio_path)
    assert info.samplerate == 24_000
    assert info.channels == 1
