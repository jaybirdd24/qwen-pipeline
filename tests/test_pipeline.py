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
