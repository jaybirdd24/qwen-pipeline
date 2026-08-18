from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def write_tone(path: Path, duration: float = 1.0, sample_rate: int = 24_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(round(duration * sample_rate), dtype=np.float32) / sample_rate
    samples = 0.1 * np.sin(2 * np.pi * 220 * times)
    sf.write(path, samples, sample_rate, subtype="PCM_16")
    return path
