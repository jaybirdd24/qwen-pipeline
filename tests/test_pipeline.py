from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import write_tone

from story_voice_pipeline.config import PipelineConfig
from story_voice_pipeline.engines.fake import FakeTTSEngine
from story_voice_pipeline.errors import ResumeError
from story_voice_pipeline.manifest import load_manifest
from story_voice_pipeline.pipeline import GenerateRequest, StoryPackGenerator
from story_voice_pipeline.story_library import load_story_library

SAMPLE_LIBRARY = Path(__file__).parents[1] / "story_library"


class InterruptingFakeEngine(FakeTTSEngine):
    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.calls = 0

    def generate(
        self,
        text: str,
        language: str,
        reference_language: str,
        output: Path,
        **kwargs: object,
    ) -> float:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        return super().generate(text, language, reference_language, output, **kwargs)


class RetryingWordEngine(FakeTTSEngine):
    def __init__(self) -> None:
        super().__init__()
        self.word_calls: list[tuple[str, int]] = []

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
        if max_new_tokens is None:
            return super().generate(text, language, reference_language, output)
        self.word_calls.append((text, seed_offset))
        sample_rate = 24_000
        duration = 3.0 if seed_offset == 0 else 0.6
        times = np.arange(round(duration * sample_rate), dtype=np.float32) / sample_rate
        speech = (0.15 * np.sin(2 * np.pi * 220 * times)).astype(np.float32)
        samples = (
            speech if seed_offset == 0 else np.concatenate([np.zeros(sample_rate * 4), speech])
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, samples, sample_rate, subtype="PCM_16")
        return 0.01


def request(tmp_path: Path, run_id: str | None = None, resume: bool = False) -> GenerateRequest:
    reference = write_tone(tmp_path / "input" / "reference.wav", duration=1.0)
    return GenerateRequest(
        library=load_story_library(tmp_path / "library" / "library.yaml"),
        reference_audio=reference,
        reference_transcript="This is the exact reference transcript.",
        voice_id="voice_001",
        voice_name="Test Caregiver",
        run_id=run_id,
        resume=resume,
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    shutil.copytree(SAMPLE_LIBRARY, tmp_path / "library")
    return tmp_path


def test_fake_pipeline_generates_bilingual_pack_and_reuses_cache(project: Path) -> None:
    config = PipelineConfig(output_root=project / "data")
    first = StoryPackGenerator(config, FakeTTSEngine()).generate(request(project))
    manifest = load_manifest(first.manifest_path)
    assert manifest["status"] == "ready_for_review"
    assert len(manifest["stories"]) == 7
    assert all(set(story["audio"]) == {"en", "zh"} for story in manifest["stories"])
    assert all(set(story["word_audio"]) == {"en", "zh"} for story in manifest["stories"])
    assert manifest["stories"][0]["word_audio"]["en"]["source_text"] == "Forest"
    dog = next(story for story in manifest["stories"] if story["story_id"] == "dog")
    assert dog["word_audio"]["zh"]["source_text"] == "狗"
    assert (first.pack_path / "forest/word/en.wav").is_file()
    assert first.generated_chunks > 0
    assert first.cache_hits == 0

    second = StoryPackGenerator(config, FakeTTSEngine()).generate(request(project))
    assert second.generated_chunks == 0
    assert second.cache_hits == first.generated_chunks
    assert second.pack_id != first.pack_id


def test_word_generation_retries_trims_and_enforces_two_seconds(project: Path) -> None:
    config = PipelineConfig(output_root=project / "data")
    engine = RetryingWordEngine()

    result = StoryPackGenerator(config, engine).generate(request(project))
    manifest = load_manifest(result.manifest_path)

    word_entries = [
        audio for story in manifest["stories"] for audio in story["word_audio"].values()
    ]
    assert word_entries
    assert all(entry["duration_seconds"] <= 2.0 for entry in word_entries)
    assert all(entry["chunks"][0]["accepted_candidate"] == 2 for entry in word_entries)
    assert all(entry["chunks"][0]["chunking_version"] == "word-prompt-v2" for entry in word_entries)
    assert all(entry["chunks"][0]["leading_trim_seconds"] > 3.9 for entry in word_entries)
    assert all(text.endswith((".", "。")) for text, _ in engine.word_calls)
    assert {seed_offset for _, seed_offset in engine.word_calls} == {0, 3}


def test_changing_one_chunk_invalidates_only_that_cache_entry(project: Path) -> None:
    config = PipelineConfig(output_root=project / "data")
    first = StoryPackGenerator(config, FakeTTSEngine()).generate(request(project))
    story_path = project / "library" / "stories" / "forest" / "en.txt"
    story_path.write_text(
        story_path.read_text().rstrip() + " One new sentence.\n", encoding="utf-8"
    )

    second = StoryPackGenerator(config, FakeTTSEngine()).generate(request(project))

    assert second.generated_chunks == 1
    assert second.cache_hits == first.generated_chunks - 1


def test_failed_run_preserves_state_and_resumes_cached_chunks(project: Path) -> None:
    config = PipelineConfig(output_root=project / "data")
    run_id = "run_interrupted"
    with pytest.raises(RuntimeError, match="simulated"):
        StoryPackGenerator(config, InterruptingFakeEngine(fail_on_call=2)).generate(
            request(project, run_id=run_id)
        )
    state_path = config.output_root / "jobs" / run_id / "request.json"
    assert json.loads(state_path.read_text())["status"] == "failed"
    assert list((config.output_root / "jobs" / run_id / "chunks").rglob("*.wav"))

    result = StoryPackGenerator(config, FakeTTSEngine()).generate(
        request(project, run_id=run_id, resume=True)
    )
    assert result.cache_hits >= 1
    assert result.manifest_path.is_file()


def test_resume_rejects_changed_inputs(project: Path) -> None:
    config = PipelineConfig(output_root=project / "data")
    run_id = "run_changed"
    with pytest.raises(RuntimeError):
        StoryPackGenerator(config, InterruptingFakeEngine(fail_on_call=1)).generate(
            request(project, run_id=run_id)
        )
    story_path = project / "library" / "stories" / "forest" / "en.txt"
    story_path.write_text(story_path.read_text() + " Changed.", encoding="utf-8")
    with pytest.raises(ResumeError, match="inputs changed"):
        StoryPackGenerator(config, FakeTTSEngine()).generate(
            request(project, run_id=run_id, resume=True)
        )


def test_completed_run_cannot_be_overwritten(project: Path) -> None:
    config = PipelineConfig(output_root=project / "data")
    run_id = "run_complete"
    StoryPackGenerator(config, FakeTTSEngine()).generate(request(project, run_id=run_id))
    with pytest.raises(ResumeError, match="already completed"):
        StoryPackGenerator(config, FakeTTSEngine()).generate(
            request(project, run_id=run_id, resume=True)
        )


@pytest.mark.parametrize("size", [1, 2, 4])
def test_batch_pipeline_cache_order_and_language_groups(project: Path, size: int) -> None:
    from dataclasses import replace

    class RecordingEngine(FakeTTSEngine):
        def __init__(self):
            super().__init__()
            self.batches = []

        def generate_batch(self, texts, language, reference_language, outputs, **kwargs):
            self.batches.append((list(texts), language, reference_language))
            return super().generate_batch(texts, language, reference_language, outputs, **kwargs)

    config = PipelineConfig(
        output_root=project / "data", max_chunk_characters=60, qwen_batch_size=size
    )
    engine = RecordingEngine()
    first = StoryPackGenerator(config, engine).generate(request(project))
    manifest = load_manifest(first.manifest_path)
    from story_voice_pipeline.chunking import chunk_text

    library = request(project).library
    for entry, story in zip(manifest["stories"], library.stories, strict=True):
        for language, audio in entry["audio"].items():
            assert [c["text"] for c in audio["chunks"]] == chunk_text(
                story.texts[language], language, 60
            )
            assert all(c["batch_size"] <= size for c in audio["chunks"])
            assert all(c["reference_language"] == "en" for c in audio["chunks"])
    if size > 1:
        assert any(len(texts) == size for texts, _, _ in engine.batches)
    else:
        assert engine.batches == []
    assert all(ref == "en" and lang in {"en", "zh"} for _, lang, ref in engine.batches)
    # Changing scheduling must reuse every original cache key and hash.
    second = StoryPackGenerator(replace(config, qwen_batch_size=4), RecordingEngine()).generate(
        request(project)
    )
    assert second.generated_chunks == 0
    assert second.cache_hits == first.generated_chunks
    second_manifest = load_manifest(second.manifest_path)
    for left, right in zip(manifest["stories"], second_manifest["stories"], strict=True):
        for language in left["audio"]:
            assert left["audio"][language]["sha256"] == right["audio"][language]["sha256"]
    # One missing middle chunk must be generated alone, preserving surrounding hits.
    chunk = manifest["stories"][0]["audio"]["en"]["chunks"][1]
    cache = StoryPackGenerator(config, engine).cache
    cache.paths(chunk["cache_key"])[0].unlink()
    engine.batches.clear()
    third = StoryPackGenerator(config, engine).generate(request(project))
    assert third.generated_chunks == 1
    assert third.cache_hits == first.generated_chunks - 1
    assert engine.batches == []


def test_batch_failure_splits_and_keeps_valid_siblings(project: Path) -> None:
    from story_voice_pipeline.errors import EngineError

    class LimitedEngine(FakeTTSEngine):
        def __init__(self):
            super().__init__()
            self.sizes = []

        def generate_batch(self, texts, language, reference_language, outputs, **kwargs):
            self.sizes.append(len(texts))
            if len(texts) > 2:
                raise EngineError("simulated CUDA out of memory")
            return super().generate_batch(texts, language, reference_language, outputs, **kwargs)

    config = PipelineConfig(
        output_root=project / "data", max_chunk_characters=60, qwen_batch_size=4
    )
    engine = LimitedEngine()
    result = StoryPackGenerator(config, engine).generate(request(project))
    assert 4 in engine.sizes and 2 in engine.sizes
    events = [
        json.loads(line)
        for line in (config.output_root / "jobs" / result.run_id / "logs/generation.jsonl")
        .read_text()
        .splitlines()
    ]
    batches = [event for event in events if event["event"] == "batch_finished"]
    assert any(event["error"] for event in batches)
    assert all(event["batch_wall_seconds"] >= 0 for event in batches)
    assert result.manifest_path.exists()


def test_invalid_batch_chunk_retries_only_bad_output(project: Path) -> None:
    from story_voice_pipeline.cache import chunk_cache_key

    class InvalidOutputEngine(FakeTTSEngine):
        def __init__(self):
            super().__init__()
            self._references.add("en")
            self.singles = []

        def generate(self, text, language, reference_language, output, **kwargs):
            self.singles.append(text)
            return super().generate(text, language, reference_language, output, **kwargs)

        def generate_batch(self, texts, language, reference_language, outputs, **kwargs):
            for text, output in zip(texts, outputs, strict=True):
                FakeTTSEngine.generate(self, text, language, reference_language, output)
            sf.write(outputs[1], np.zeros(24000), 24000)
            return 0.1

    engine = InvalidOutputEngine()
    generator = StoryPackGenerator(PipelineConfig(output_root=project / "data"), engine)
    jobs = []
    for i in range(4):
        payload = {"text": f"Text {i}.", "language": "en", "reference_language": "en"}
        jobs.append((chunk_cache_key(payload), project / f"{i}.wav", payload))
    results = {}
    generator._generate_chunk_batch(project / "run", jobs, results)
    assert len(results) == 4
    assert engine.singles == ["Text 1."]
    assert all(generator.cache.get(key) is not None for key, _, _ in jobs)


def test_irrecoverable_chunk_still_caches_siblings(project: Path) -> None:
    from story_voice_pipeline.errors import EngineError

    class FailingEngine(FakeTTSEngine):
        def __init__(self):
            super().__init__()
            self._references.add("en")

        def generate_batch(self, *args, **kwargs):
            raise EngineError("batch failed")

        def generate(self, text, *args, **kwargs):
            if text == "bad":
                raise EngineError("bad chunk")
            return super().generate(text, *args, **kwargs)

    generator = StoryPackGenerator(PipelineConfig(output_root=project / "data"), FailingEngine())
    jobs = [
        (
            str(i) * 64,
            project / f"{i}.wav",
            {"text": text, "language": "en", "reference_language": "en"},
        )
        for i, text in enumerate(["bad", "Good two.", "Good three.", "Good four."])
    ]
    results = {}
    with pytest.raises(EngineError, match="bad chunk"):
        generator._generate_chunk_batch(project / "run", jobs, results)
    assert len(results) == 3
    assert all(generator.cache.get(key) is not None for key, _, _ in jobs[1:])


def test_all_seven_languages_generate_cache_and_export(project: Path) -> None:
    from dataclasses import replace

    from story_voice_pipeline.chunking import chunk_text
    from story_voice_pipeline.pi_bundle import export_pi_bundle
    from story_voice_pipeline.pipeline import _word_synthesis_text

    config = PipelineConfig(
        output_root=project / "multilingual", max_chunk_characters=60, qwen_batch_size=4
    )
    req = request(project)
    req = replace(req, story_ids=("forest",), languages=req.library.required_languages)
    generator = StoryPackGenerator(config, FakeTTSEngine())
    first = generator.generate(req)
    manifest = load_manifest(first.manifest_path)
    story = manifest["stories"][0]
    expected = {"en", "zh", "ja", "ko", "de", "pt", "es"}
    assert set(story["audio"]) == set(story["word_audio"]) == expected
    for language in expected:
        audio = story["audio"][language]
        assert audio["reference_language"] == "en"
        source = req.library.stories[0].texts[language]
        assert [c["text"] for c in audio["chunks"]] == chunk_text(source, language, 60)
        assert (first.pack_path / audio["path"]).is_file()
        assert (first.pack_path / story["word_audio"][language]["path"]).is_file()
    assert _word_synthesis_text("森", "ja") == "森。"
    assert _word_synthesis_text("숲", "ko") == "숲."
    assert _word_synthesis_text("Floresta", "pt") == "Floresta."
    second = generator.generate(req)
    assert second.generated_chunks == 0
    assert second.cache_hits == first.generated_chunks
    exported = export_pi_bundle(
        [first.pack_path], project / "seven-language-bundle", allow_review_ready=True
    )
    assert exported.audio_count == 14
