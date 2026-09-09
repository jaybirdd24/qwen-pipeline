"""Uncached, sequential batch-size comparisons using a single loaded Qwen model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .audio import validate_wav
from .engines.qwen import QwenTTSEngine
from .errors import PipelineError


def benchmark_batches(
    engine: QwenTTSEngine, texts: list[str], language: str, output_dir: Path
) -> list[dict[str, Any]]:
    if not texts:
        raise PipelineError("Benchmark requires at least one text chunk")
    torch = engine._torch
    device = engine.config.device
    if not device.startswith("cuda"):
        raise PipelineError("The batch benchmark requires a CUDA device")
    rows = []
    for size in (1, 2, 4):
        row: dict[str, Any] = {"batch_size": size, "chunk_count": len(texts)}
        paths = [output_dir / f"batch_{size}" / f"chunk_{i:04d}.wav" for i in range(len(texts))]
        duration = 0.0
        started = None
        try:
            # Warm up each shape; model loading and prompt preparation are excluded.
            warm_texts = [texts[i % len(texts)] for i in range(size)]
            engine.generate_batch(
                warm_texts,
                language,
                "en",
                [output_dir / "warmup" / f"{i}.wav" for i in range(size)],
            )
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for start in range(0, len(texts), size):
                engine.generate_batch(
                    texts[start : start + size], language, "en", paths[start : start + size]
                )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            for text, path in zip(texts, paths, strict=True):
                duration += validate_wav(path, len(text)).duration_seconds
            row.update(
                status="ok",
                wall_seconds=elapsed,
                total_audio_seconds=duration,
                effective_rtf=elapsed / duration,
                peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device),
            )
        except RuntimeError as exc:
            row.update(
                status="failed",
                error=str(exc),
                wall_seconds=None if started is None else time.perf_counter() - started,
                total_audio_seconds=None,
                effective_rtf=None,
                peak_cuda_memory_bytes=(
                    None if started is None else torch.cuda.max_memory_allocated(device)
                ),
                failure_phase="warmup" if started is None else "measurement",
            )
        rows.append(row)
    return rows
