"""Command-line interface for story-pack generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from .audio import normalize_reference, validate_wav
from .config import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION, PipelineConfig
from .engines import FakeTTSEngine, QwenTTSEngine
from .errors import PipelineError
from .models import SUPPORTED_LANGUAGES
from .pi_bundle import export_pi_bundle
from .pipeline import GenerateRequest, StoryPackGenerator, utc_now
from .story_library import load_story_library


def _read_text(path: Path, label: str) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise PipelineError(f"Cannot read {label} {path}: {exc}") from exc
    if not text:
        raise PipelineError(f"{label} is empty: {path}")
    return text


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=os.getenv("QWEN_DTYPE", "bfloat16"),
    )
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--qwen-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-generation-attempts", type=int, default=3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="story-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-library", help="Validate and summarize a library")
    validate.add_argument("--library", required=True, type=Path)

    generate = subparsers.add_parser("generate", help="Generate a review-ready story pack")
    generate.add_argument("--library", required=True, type=Path)
    generate.add_argument("--reference-audio", required=True, type=Path)
    generate.add_argument("--reference-transcript-file", required=True, type=Path)
    generate.add_argument("--mandarin-reference-audio", type=Path)
    generate.add_argument("--mandarin-reference-transcript-file", type=Path)
    generate.add_argument("--voice-id", required=True)
    generate.add_argument("--voice-name", required=True)
    generate.add_argument("--voice-version", type=int, default=1)
    generate.add_argument("--story", action="append", default=[])
    generate.add_argument("--language", action="append", choices=SUPPORTED_LANGUAGES, default=[])
    generate.add_argument("--run-id")
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--engine", choices=("qwen", "fake"), default="qwen")
    generate.add_argument("--output-root", type=Path, default=Path("data"))
    generate.add_argument("--max-chunk-characters", type=int, default=400)
    generate.add_argument("--chunk-silence-ms", type=int, default=350)
    generate.add_argument("--peak-dbfs", type=float, default=-1.0)
    _add_runtime_options(generate)

    smoke = subparsers.add_parser("smoke-test", help="Run an opt-in bilingual Qwen GPU test")
    smoke.add_argument("--reference-audio", required=True, type=Path)
    smoke.add_argument("--reference-transcript-file", required=True, type=Path)
    smoke.add_argument("--output-dir", required=True, type=Path)
    _add_runtime_options(smoke)

    benchmark = subparsers.add_parser("benchmark", help="Compare uncached Qwen batches 1, 2, 4")
    benchmark.add_argument("--reference-audio", required=True, type=Path)
    benchmark.add_argument("--reference-transcript-file", required=True, type=Path)
    benchmark.add_argument("--text-file", required=True, type=Path)
    benchmark.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="en")
    benchmark.add_argument("--max-chunk-characters", type=int, default=400)
    benchmark.add_argument("--output-dir", required=True, type=Path)
    _add_runtime_options(benchmark)

    export_pi = subparsers.add_parser(
        "export-pi", help="Export approved story packs as an offline Pi bundle"
    )
    export_pi.add_argument("--pack", action="append", required=True, type=Path)
    export_pi.add_argument("--output-dir", required=True, type=Path)
    export_pi.add_argument("--selected-voice")
    export_pi.add_argument(
        "--allow-review-ready",
        action="store_true",
        help="Allow reviewed but unapproved packs in a prototype bundle",
    )
    return parser


def _config_from_args(args: argparse.Namespace, output_root: Path) -> PipelineConfig:
    config = PipelineConfig(output_root=output_root)
    changes = {
        key: getattr(args, key)
        for key in (
            "model_name",
            "model_revision",
            "device",
            "dtype",
            "attention_implementation",
            "seed",
            "max_new_tokens",
            "max_generation_attempts",
        )
    }
    for optional in ("max_chunk_characters", "chunk_silence_ms", "peak_dbfs"):
        if hasattr(args, optional):
            changes[optional] = getattr(args, optional)
    if args.qwen_batch_size is not None:
        changes["qwen_batch_size"] = args.qwen_batch_size
    return replace(config, **changes)


def _engine(name: str, config: PipelineConfig):  # type: ignore[no-untyped-def]
    if name == "fake":
        return FakeTTSEngine(config.sample_rate, config.seed)
    return QwenTTSEngine(config)


def _validate_command(args: argparse.Namespace) -> int:
    library = load_story_library(args.library)
    print(
        json.dumps(
            {
                "library_id": library.library_id,
                "version": library.version,
                "required_languages": library.required_languages,
                "stories": [story.story_id for story in library.stories],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _generate_command(args: argparse.Namespace) -> int:
    if (args.mandarin_reference_audio is None) != (args.mandarin_reference_transcript_file is None):
        raise PipelineError(
            "--mandarin-reference-audio and --mandarin-reference-transcript-file are a pair"
        )
    config = _config_from_args(args, args.output_root)
    library = load_story_library(args.library)
    engine = _engine(args.engine, config)
    request = GenerateRequest(
        library=library,
        reference_audio=args.reference_audio,
        reference_transcript=_read_text(args.reference_transcript_file, "reference transcript"),
        voice_id=args.voice_id,
        voice_name=args.voice_name,
        voice_version=args.voice_version,
        mandarin_reference_audio=args.mandarin_reference_audio,
        mandarin_reference_transcript=(
            _read_text(args.mandarin_reference_transcript_file, "Mandarin reference transcript")
            if args.mandarin_reference_transcript_file
            else None
        ),
        story_ids=tuple(args.story),
        languages=tuple(args.language or library.required_languages),
        run_id=args.run_id,
        resume=args.resume,
    )
    result = StoryPackGenerator(config, engine).generate(request)
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "pack_id": result.pack_id,
                "manifest": str(result.manifest_path),
                "cache_hits": result.cache_hits,
                "generated_chunks": result.generated_chunks,
            },
            indent=2,
        )
    )
    return 0


def _smoke_command(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _config_from_args(args, output_dir)
    reference = normalize_reference(
        args.reference_audio,
        _read_text(args.reference_transcript_file, "reference transcript"),
        output_dir / "reference.wav",
        "en",
        config.sample_rate,
    )
    engine = QwenTTSEngine(config)
    engine.prepare_reference(reference)
    cases = {
        "en": "A small apple waited beside the garden path.",
        "zh": "一颗小苹果在花园的小路旁等待。",
    }
    outputs: dict[str, object] = {}
    started = time.perf_counter()
    for language, text in cases.items():
        path = output_dir / f"smoke_{language}.wav"
        generation_seconds = engine.generate(text, language, "en", path)
        validation = validate_wav(path, len(text))
        outputs[language] = {
            "path": str(path),
            "sha256": validation.sha256,
            "duration_seconds": validation.duration_seconds,
            "sample_rate": validation.sample_rate,
            "generation_seconds": generation_seconds,
            "warnings": validation.warnings,
        }
    report = {
        "created_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "reference_audio_hash": reference.audio_hash,
        "reference_warnings": reference.warnings,
        "runtime": engine.runtime_metadata(),
        "outputs": outputs,
    }
    report_path = output_dir / "smoke_test.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _benchmark_command(args: argparse.Namespace) -> int:
    from .benchmark import benchmark_batches
    from .chunking import chunk_text

    config = _config_from_args(args, args.output_dir)
    texts = chunk_text(
        _read_text(args.text_file, "benchmark text"), args.language, config.max_chunk_characters
    )
    reference = normalize_reference(
        args.reference_audio,
        _read_text(args.reference_transcript_file, "reference transcript"),
        args.output_dir / "reference.wav",
        "en",
        config.sample_rate,
    )
    engine = QwenTTSEngine(config)
    engine.prepare_reference(reference)
    rows = benchmark_batches(engine, texts, args.language, args.output_dir)
    report = {
        "created_at": utc_now(),
        "parameters": config.generation_parameters(),
        "runtime": engine.runtime_metadata(),
        "results": rows,
        "timing_scope": "generation and WAV writing; excludes warmup, loading and validation",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(row["status"] == "ok" for row in rows) else 2


def _export_pi_command(args: argparse.Namespace) -> int:
    result = export_pi_bundle(
        args.pack,
        args.output_dir,
        args.selected_voice,
        allow_review_ready=args.allow_review_ready,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "selected_voice_id": result.selected_voice_id,
                "voice_count": result.voice_count,
                "audio_count": result.audio_count,
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-library":
            return _validate_command(args)
        if args.command == "generate":
            return _generate_command(args)
        if args.command == "smoke-test":
            return _smoke_command(args)
        if args.command == "benchmark":
            return _benchmark_command(args)
        if args.command == "export-pi":
            return _export_pi_command(args)
        raise PipelineError(f"Unknown command: {args.command}")
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
