"""Reference normalization, WAV assembly, and validation."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .errors import AudioValidationError
from .hashing import sha256_file, sha256_text
from .models import AudioValidation, VoiceReference


@dataclass(frozen=True)
class WordAudioPreparation:
    validation: AudioValidation
    original_duration_seconds: float
    active_duration_seconds: float
    leading_trim_seconds: float
    trailing_trim_seconds: float


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        samples, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    except (OSError, RuntimeError) as exc:
        raise AudioValidationError(f"Cannot read audio file {path}: {exc}") from exc
    if samples.size == 0 or sample_rate <= 0:
        raise AudioValidationError(f"Audio file is empty or has an invalid sample rate: {path}")
    if not np.isfinite(samples).all():
        raise AudioValidationError(f"Audio contains NaN or infinite samples: {path}")
    return samples, int(sample_rate)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        samples,
        up=target_rate // divisor,
        down=source_rate // divisor,
    ).astype(np.float32)


def _read_reference_audio(path: Path, target_rate: int) -> tuple[np.ndarray, int]:
    try:
        return _read_audio(path)
    except AudioValidationError as direct_error:
        with tempfile.TemporaryDirectory(prefix="story-pipeline-audio-") as directory:
            converted = Path(directory) / "reference.wav"
            try:
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-nostdin",
                        "-y",
                        "-i",
                        str(path),
                        "-ac",
                        "1",
                        "-ar",
                        str(target_rate),
                        "-c:a",
                        "pcm_s16le",
                        str(converted),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise AudioValidationError(
                    f"Cannot decode {path} with libsndfile and FFmpeg is not installed"
                ) from exc
            if result.returncode != 0 or not converted.is_file():
                detail = result.stderr.strip() or str(direct_error)
                raise AudioValidationError(
                    f"Cannot convert reference audio {path}: {detail}"
                ) from direct_error
            return _read_audio(converted)


def normalize_reference(
    source_path: Path,
    transcript: str,
    destination_path: Path,
    language: str,
    sample_rate: int = 24_000,
) -> VoiceReference:
    transcript = transcript.strip()
    if not transcript:
        raise AudioValidationError(f"{language} reference transcript is empty")
    if not source_path.is_file():
        raise AudioValidationError(f"Reference audio does not exist: {source_path}")
    samples, source_rate = _read_reference_audio(source_path, sample_rate)
    mono = samples.mean(axis=1)
    mono = _resample(mono, source_rate, sample_rate)
    if not np.isfinite(mono).all() or not np.any(np.abs(mono) > 1e-5):
        raise AudioValidationError(f"Reference audio is silent or invalid: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(".tmp.wav")
    sf.write(temporary, mono, sample_rate, subtype="PCM_16", format="WAV")
    os.replace(temporary, destination_path)
    duration = len(mono) / sample_rate
    warnings: list[str] = []
    if duration < 10:
        warnings.append(f"Reference is shorter than the preferred 10 seconds ({duration:.2f}s)")
    if duration > 40:
        warnings.append(f"Reference is longer than the preferred 40 seconds ({duration:.2f}s)")
    return VoiceReference(
        language=language,
        audio_path=destination_path,
        transcript=transcript,
        audio_hash=sha256_file(destination_path),
        transcript_hash=sha256_text(transcript),
        duration_seconds=duration,
        warnings=tuple(warnings),
    )


def validate_wav(path: Path, text_length: int | None = None) -> AudioValidation:
    samples, sample_rate = _read_audio(path)
    mono = samples.mean(axis=1)
    duration = len(mono) / sample_rate
    if duration <= 0:
        raise AudioValidationError(f"Audio duration is zero: {path}")
    absolute = np.abs(mono)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    if rms <= 1e-5:
        raise AudioValidationError(f"Audio is entirely silent: {path}")

    warnings: list[str] = []
    clipped_fraction = float(np.mean(absolute >= 0.999))
    if clipped_fraction > 0.01:
        warnings.append(f"Audio may be severely clipped ({clipped_fraction:.1%} samples at peak)")
    if text_length is not None and text_length > 0:
        minimum = max(0.25, text_length / 40.0)
        maximum = max(5.0, text_length / 1.5)
        if duration < minimum or duration > maximum:
            warnings.append(
                f"Duration {duration:.2f}s is outside the plausible "
                f"{minimum:.2f}–{maximum:.2f}s range"
            )
    return AudioValidation(
        path=path,
        sha256=sha256_file(path),
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=int(samples.shape[1]),
        peak_amplitude=peak,
        rms_amplitude=rms,
        warnings=tuple(warnings),
    )


def assemble_wavs(
    chunk_paths: list[Path],
    output_path: Path,
    silence_ms: int,
    peak_dbfs: float,
    expected_sample_rate: int,
) -> AudioValidation:
    if not chunk_paths:
        raise AudioValidationError("Cannot assemble audio without chunks")
    chunks: list[np.ndarray] = []
    for path in chunk_paths:
        samples, sample_rate = _read_audio(path)
        if sample_rate != expected_sample_rate:
            raise AudioValidationError(
                f"Chunk {path} has sample rate {sample_rate}; expected {expected_sample_rate}"
            )
        chunks.append(samples.mean(axis=1))
    silence = np.zeros(round(expected_sample_rate * silence_ms / 1000), dtype=np.float32)
    combined_parts: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        if index:
            combined_parts.append(silence)
        combined_parts.append(chunk)
    combined = np.concatenate(combined_parts).astype(np.float32)
    peak = float(np.max(np.abs(combined)))
    if peak <= 1e-5:
        raise AudioValidationError("Assembled audio is silent")
    target_peak = 10 ** (peak_dbfs / 20.0)
    combined *= target_peak / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.wav")
    sf.write(temporary, combined, expected_sample_rate, subtype="PCM_16", format="WAV")
    os.replace(temporary, output_path)
    return validate_wav(output_path)


def prepare_word_audio(
    source_path: Path,
    output_path: Path,
    expected_sample_rate: int,
    peak_dbfs: float,
    *,
    max_duration_seconds: float = 2.0,
    minimum_active_seconds: float = 0.15,
    window_ms: int = 20,
    padding_ms: int = 50,
) -> WordAudioPreparation:
    """Trim a generated word to active speech, normalize it, and enforce a hard duration cap."""
    samples, sample_rate = _read_audio(source_path)
    if sample_rate != expected_sample_rate:
        raise AudioValidationError(
            f"Word audio has sample rate {sample_rate}; expected {expected_sample_rate}"
        )
    mono = samples.mean(axis=1)
    original_duration = len(mono) / sample_rate
    peak = float(np.max(np.abs(mono)))
    if peak <= 1e-5:
        raise AudioValidationError("Generated word audio is silent")

    window = max(1, round(sample_rate * window_ms / 1000))
    smoothed_power = np.convolve(
        np.square(mono, dtype=np.float64),
        np.ones(window, dtype=np.float64) / window,
        mode="same",
    )
    envelope = np.sqrt(smoothed_power)
    threshold = max(peak * 0.01, 10 ** (-50 / 20))
    active = np.flatnonzero(envelope >= threshold)
    if not active.size:
        raise AudioValidationError("Generated word audio contains no detectable speech")

    active_start = int(active[0])
    active_end = int(active[-1]) + 1
    active_duration = (active_end - active_start) / sample_rate
    if active_duration < minimum_active_seconds:
        raise AudioValidationError(
            f"Generated word has only {active_duration:.2f}s of detectable speech"
        )
    if active_duration > max_duration_seconds:
        raise AudioValidationError(
            f"Generated word has {active_duration:.2f}s of active audio; "
            f"maximum is {max_duration_seconds:.2f}s"
        )

    requested_padding = round(sample_rate * padding_ms / 1000)
    available_padding = max(0, round((max_duration_seconds - active_duration) * sample_rate))
    leading_padding = min(requested_padding, available_padding // 2, active_start)
    trailing_padding = min(
        requested_padding,
        available_padding - leading_padding,
        len(mono) - active_end,
    )
    start = active_start - leading_padding
    end = active_end + trailing_padding
    trimmed = mono[start:end].astype(np.float32, copy=True)
    trimmed_peak = float(np.max(np.abs(trimmed)))
    if trimmed_peak <= 1e-5:
        raise AudioValidationError("Trimmed word audio is silent")
    trimmed *= (10 ** (peak_dbfs / 20.0)) / trimmed_peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.wav")
    sf.write(temporary, trimmed, sample_rate, subtype="PCM_16", format="WAV")
    os.replace(temporary, output_path)
    validation = validate_wav(output_path)
    if validation.duration_seconds > max_duration_seconds + (1 / sample_rate):
        raise AudioValidationError(
            f"Prepared word is {validation.duration_seconds:.2f}s; "
            f"maximum is {max_duration_seconds:.2f}s"
        )
    return WordAudioPreparation(
        validation=validation,
        original_duration_seconds=original_duration,
        active_duration_seconds=active_duration,
        leading_trim_seconds=start / sample_rate,
        trailing_trim_seconds=(len(mono) - end) / sample_rate,
    )


def real_time_factor(generation_seconds: float, audio_seconds: float) -> float:
    if audio_seconds <= 0 or not math.isfinite(audio_seconds):
        raise AudioValidationError("Cannot calculate real-time factor for invalid duration")
    return generation_seconds / audio_seconds
