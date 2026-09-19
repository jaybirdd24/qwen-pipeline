"""Launch isolated service trials and summarize their recorded evidence."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preflight() -> dict:
    from story_voice_pipeline.chunking import chunk_text
    from story_voice_pipeline.story_library import load_story_library

    library = load_story_library(HERE / "library/library.yaml")
    groups = {}
    for story in library.stories:
        for language, text in story.texts.items():
            chunks = chunk_text(text, language, 400)
            if len(chunks) < 4:
                raise ValueError(f"{story.story_id}/{language} has fewer than four chunks")
            groups[f"{story.story_id}/{language}"] = {
                "chunks": len(chunks),
                "lengths": [len(chunk) for chunk in chunks],
                "source_hash": story.source_hashes[language],
            }
    return groups


def start(root: Path, trial: int) -> int:
    groups = preflight()
    root = root.resolve()
    destination = root / f"trial-{trial}"
    root.mkdir(parents=True, exist_ok=True)
    # Never reuse caches or a database, even after an interrupted launch.
    destination.mkdir()
    env = dict(os.environ)
    env.update(
        CONTROL_DATA_ROOT=str(destination),
        DATABASE_URL=f"sqlite:///{destination / 'control.db'}",
        CONTROL_STORY_LIBRARY=str(HERE / "library/library.yaml"),
        CONTROL_TTS_ENGINE="qwen",
        QWEN_DTYPE="float32",
        QWEN_BATCH_SIZE="4",
        MAX_CHUNK_CHARACTERS="400",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        CONTROL_HOST="127.0.0.1",
        CONTROL_PORT="8001",
    )
    # Keep the usual model, token budget, seed, and reference handling.
    settings = {
        key: value
        for key, value in env.items()
        if key.startswith("QWEN_")
        or key
        in {
            "CONTROL_DATA_ROOT",
            "CONTROL_STORY_LIBRARY",
            "DATABASE_URL",
            "MAX_CHUNK_CHARACTERS",
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_ALLOC_CONF",
        }
    }
    (destination / "trial.json").write_text(
        json.dumps({"trial": trial, "groups": groups, "environment": settings}, indent=2) + "\n"
    )
    print(f"Trial {trial}: http://127.0.0.1:8001", flush=True)
    print(
        f"Upload English and Mandarin references, generate ALL stories in BOTH languages once.\n"
        f"Stop with Ctrl-C after completion. Service log: {destination / 'service.log'}",
        flush=True,
    )
    with (destination / "service.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from story_voice_pipeline.control_service.app import main; main()",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return 0


def summarize_trial(destination: Path) -> dict:
    result = {
        "trial": destination.name,
        "stability_pass": False,
        "fallback_recovered": False,
        "audio_review": "pending",
    }
    setup = read_json(destination / "trial.json")
    requests = list((destination / "jobs").glob("*/request.json"))
    if len(requests) != 1:
        return {**result, "reason": "Exactly one generation per trial is required"}
    request_path = requests[0]
    state = read_json(request_path)
    events = [
        json.loads(line)
        for line in (request_path.parent / "logs/generation.jsonl").read_text().splitlines()
    ]
    batches = [e for e in events if e["event"] == "batch_finished"]
    failures = [e for e in batches if e.get("error")]
    rejections = [e for e in events if e["event"] == "batch_chunk_rejected"]
    successful_four = [e for e in batches if e["batch_size"] == 4 and not e.get("error")]
    # Service job timestamps include model loading, reference preparation and assembly.
    database = destination / "control.db"
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as connection:
        timing = connection.execute(
            "SELECT started_at, completed_at FROM generation_jobs WHERE run_id = ?",
            (state["run_id"],),
        ).fetchone()
    wall = None
    if timing and all(timing):
        wall = (
            datetime.fromisoformat(timing[1]) - datetime.fromisoformat(timing[0])
        ).total_seconds()
    completed = state.get("status") == "completed"
    expected_chunks = sum(group["chunks"] for group in setup["groups"].values())
    generated = [
        e for e in events if e["event"] == "chunk_generated" and e.get("asset_type") != "word"
    ]
    generated_groups = {e["cache_key"]: f"{e['story_id']}/{e['language']}" for e in generated}
    observed = {
        generated_groups[key]
        for batch in successful_four
        for key in batch["cache_keys"]
        if key in generated_groups
    }
    source_hashes = {
        f"{story['story_id']}/{language}": digest
        for story in state["stories"]
        for language, digest in story["source_hashes"].items()
    }
    correct_settings = (
        state["engine"]["name"] == "qwen"
        and state["parameters"]["dtype"] == "float32"
        and state.get("qwen_batch_size") == 4
        and state["parameters"]["max_chunk_characters"] == 400
        and source_hashes == {key: group["source_hash"] for key, group in setup["groups"].items()}
    )
    eligible = (
        completed
        and correct_settings
        and wall is not None
        and len(generated) == expected_chunks
        and not any(e["event"] == "chunk_cache_hit" for e in events)
        and set(setup["groups"]) <= observed
    )
    result.update(
        status=state.get("status"),
        end_to_end_wall_seconds=wall,
        batch_wall_seconds=sum(e["batch_wall_seconds"] for e in batches),
        successful_four_chunk_batches=len(successful_four),
        failed_batches=len(failures),
        rejected_chunk_events=len(rejections),
        oom_batches=sum("out of memory" in e["error"].lower() for e in failures),
        batch_sizes=[e["batch_size"] for e in batches],
        cache_hits=sum(e["event"] == "chunk_cache_hit" for e in events),
        stability_pass=eligible and not failures and not rejections,
        fallback_recovered=completed and bool(failures or rejections),
        pack_path=state.get("pack_path"),
        comparison_inputs={
            "references": state["references"],
            "parameters": state["parameters"],
            "engine": state["engine"],
            "source_hashes": source_hashes,
        },
    )
    if not eligible:
        result["reason"] = (
            "Incomplete run, wrong settings/workload, cache hits, or missing batch-4 evidence"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "start", "summarize"])
    parser.add_argument("--root", type=Path, default=Path("data/batch4-validation"))
    parser.add_argument("--trial", type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2))
        return 0
    if args.command == "start":
        if args.trial is None:
            parser.error("start requires --trial")
        return start(args.root, args.trial)
    rows = []
    for trial in (1, 2, 3):
        try:
            rows.append(summarize_trial(args.root / f"trial-{trial}"))
        except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
            rows.append({"trial": f"trial-{trial}", "stability_pass": False, "reason": str(exc)})
    same_inputs = (
        len({json.dumps(row.get("comparison_inputs"), sort_keys=True) for row in rows}) == 1
    )
    passed = all(row["stability_pass"] for row in rows) and same_inputs
    print(
        json.dumps(
            {
                "trials": rows,
                "same_inputs": same_inputs,
                "stability_pass": passed,
                "production_decision": "Audio review required"
                if passed
                else "Keep batch 4 experimental; use batch 2 if OOM recurs",
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
