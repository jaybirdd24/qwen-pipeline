from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "validation/batch4/trials.py"
spec = importlib.util.spec_from_file_location("batch4_trials", SCRIPT)
assert spec and spec.loader
trials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trials)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def trial_data(root: Path, number: int = 1) -> Path:
    destination = root / f"trial-{number}"
    groups = trials.preflight()
    write_json(destination / "trial.json", {"groups": groups})
    run = destination / "jobs/run_test"
    state = {
        "status": "completed",
        "run_id": "run_test",
        "qwen_batch_size": 4,
        "engine": {"name": "qwen"},
        "parameters": {"dtype": "float32", "max_chunk_characters": 400},
        "references": {"en": {"audio_hash": "same-reference"}},
        "stories": [
            {
                "story_id": "lantern",
                "source_hashes": {
                    language: groups[f"lantern/{language}"]["source_hash"]
                    for language in ("en", "zh")
                },
            }
        ],
    }
    write_json(run / "request.json", state)
    events = []
    for language in ("en", "zh"):
        for offset in (0, 4):
            keys = [f"{language}-{i}" for i in range(offset, offset + 4)]
            events.append(
                {
                    "event": "batch_finished",
                    "batch_size": 4,
                    "error": None,
                    "cache_keys": keys,
                    "language": language,
                    "batch_wall_seconds": 10,
                }
            )
            events.extend(
                {
                    "event": "chunk_generated",
                    "cache_key": key,
                    "story_id": "lantern",
                    "language": language,
                }
                for key in keys
            )
    log = run / "logs/generation.jsonl"
    log.parent.mkdir()
    log.write_text("\n".join(json.dumps(event) for event in events))
    with sqlite3.connect(destination / "control.db") as connection:
        connection.execute("CREATE TABLE generation_jobs (run_id, started_at, completed_at)")
        connection.execute(
            "INSERT INTO generation_jobs VALUES (?, ?, ?)",
            ("run_test", "2026-09-15 01:00:00", "2026-09-15 01:02:00"),
        )
    return destination


def test_long_library_has_eight_distinct_chunks_per_language():
    from story_voice_pipeline.chunking import chunk_text
    from story_voice_pipeline.story_library import load_story_library

    groups = trials.preflight()
    assert set(groups) == {"lantern/en", "lantern/zh"}
    assert all(group["chunks"] == 8 for group in groups.values())
    assert all(max(group["lengths"]) <= 400 for group in groups.values())
    library = load_story_library(SCRIPT.parent / "library/library.yaml")
    for language, text in library.stories[0].texts.items():
        assert len(set(chunk_text(text, language, 400))) == 8


def test_summary_success_and_wall_times(tmp_path):
    result = trials.summarize_trial(trial_data(tmp_path))
    assert result["stability_pass"]
    assert result["successful_four_chunk_batches"] == 4
    assert result["end_to_end_wall_seconds"] == 120
    assert result["batch_wall_seconds"] == 40
    assert result["audio_review"] == "pending"
    assert not result["fallback_recovered"]


@pytest.mark.parametrize("failure", ["oom", "cache", "single", "fake", "partial", "source"])
def test_invalid_trials_cannot_pass(tmp_path, failure):
    destination = trial_data(tmp_path)
    log = destination / "jobs/run_test/logs/generation.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    state_path = destination / "jobs/run_test/request.json"
    state = trials.read_json(state_path)
    if failure == "oom":
        events.append(
            {
                "event": "batch_finished",
                "batch_size": 4,
                "language": "en",
                "error": "CUDA out of memory",
                "batch_wall_seconds": 7,
                "cache_keys": [],
            }
        )
    elif failure == "cache":
        events.append({"event": "chunk_cache_hit"})
    elif failure == "single":
        for event in events:
            if event["event"] == "batch_finished":
                event["batch_size"] = 1
    elif failure == "fake":
        state["engine"]["name"] = "fake"
    elif failure == "partial":
        state["status"] = "failed"
    else:
        state["stories"][0]["source_hashes"]["zh"] = "changed"
    write_json(state_path, state)
    log.write_text("\n".join(json.dumps(event) for event in events))
    result = trials.summarize_trial(destination)
    assert not result["stability_pass"]
    if failure == "oom":
        assert result["fallback_recovered"]
        assert result["oom_batches"] == 1
        assert result["batch_wall_seconds"] == 47


def test_launch_isolates_database_and_preserves_existing_trials(tmp_path, monkeypatch):
    calls = []

    class Process:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def wait(self):
            return 0

    monkeypatch.setenv("DATABASE_URL", "sqlite:////production/control.db")
    monkeypatch.setenv("QWEN_DTYPE", "bfloat16")
    monkeypatch.setenv("QWEN_MAX_NEW_TOKENS", "500")
    monkeypatch.setattr(trials.subprocess, "Popen", Process)
    assert trials.start(tmp_path, 1) == 0
    env = calls[0]["env"]
    assert env["DATABASE_URL"] == f"sqlite:///{tmp_path / 'trial-1/control.db'}"
    assert env["CONTROL_DATA_ROOT"] == str(tmp_path / "trial-1")
    assert env["QWEN_DTYPE"] == "float32"
    assert env["QWEN_MAX_NEW_TOKENS"] == "500"
    assert env["QWEN_BATCH_SIZE"] == "4"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    with pytest.raises(FileExistsError):
        trials.start(tmp_path, 1)
    assert len(calls) == 1


def test_three_trials_require_matching_references(tmp_path, monkeypatch, capsys):
    for number in (1, 2, 3):
        trial_data(tmp_path, number)
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "summarize", "--root", str(tmp_path)])
    assert trials.main() == 0
    assert json.loads(capsys.readouterr().out)["stability_pass"]
    path = tmp_path / "trial-3/jobs/run_test/request.json"
    state = trials.read_json(path)
    state["references"]["en"]["audio_hash"] = "different-reference"
    write_json(path, state)
    assert trials.main() == 1
    assert not json.loads(capsys.readouterr().out)["same_inputs"]


def test_missing_trials_do_not_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "summarize", "--root", str(tmp_path)])
    assert trials.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["stability_pass"]
    assert len(report["trials"]) == 3


def test_real_service_logs_are_summarized_without_claiming_gpu_success(tmp_path):
    from conftest import write_tone
    from fastapi.testclient import TestClient

    from story_voice_pipeline.config import PipelineConfig
    from story_voice_pipeline.control_service.app import create_app
    from story_voice_pipeline.control_service.config import ControlSettings

    destination = tmp_path / "trial-1"
    write_json(destination / "trial.json", {"groups": trials.preflight()})
    app = create_app(
        ControlSettings(
            data_root=destination,
            story_library_path=SCRIPT.parent / "library/library.yaml",
            tts_engine="fake",
            pipeline=PipelineConfig(output_root=destination, dtype="float32", qwen_batch_size=4),
        )
    )
    reference = write_tone(tmp_path / "reference.wav").read_bytes()
    with TestClient(app) as client:
        voice = client.post(
            "/api/v1/voices",
            data={
                "name": "Fixture",
                "english_transcript": "Test reference.",
                "mandarin_transcript": "中文参考。",
                "consent_confirmed": "true",
            },
            files={
                "english_audio": ("reference.wav", reference, "audio/wav"),
                "mandarin_audio": ("reference.wav", reference, "audio/wav"),
            },
        )
        assert voice.status_code == 201, voice.text
        job = client.post(
            "/api/v1/jobs",
            json={
                "voice_id": voice.json()["id"],
                "story_ids": ["lantern"],
                "languages": ["en", "zh"],
            },
        )
        assert job.status_code == 202, job.text
    result = trials.summarize_trial(destination)
    assert result["status"] == "completed"
    assert result["successful_four_chunk_batches"] == 4
    assert result["end_to_end_wall_seconds"] > 0
    assert result["batch_wall_seconds"] > 0
    assert result["cache_hits"] == 0
    assert not result["stability_pass"]  # Fake execution must never count as GPU validation.
