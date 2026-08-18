from __future__ import annotations

from pathlib import Path

from conftest import write_tone

from story_voice_pipeline.cache import ChunkCache, chunk_cache_key


def test_key_is_stable_and_changes_with_inputs() -> None:
    payload = {"text": "hello", "seed": 1, "reference_audio_hash": "abc"}
    assert chunk_cache_key(payload) == chunk_cache_key(dict(reversed(list(payload.items()))))
    assert chunk_cache_key(payload) != chunk_cache_key({**payload, "seed": 2})
    assert chunk_cache_key(payload) != chunk_cache_key({**payload, "text": "changed"})


def test_corrupt_cache_is_a_miss(tmp_path: Path) -> None:
    cache = ChunkCache(tmp_path / "cache")
    key = chunk_cache_key({"text": "hello"})
    source = write_tone(tmp_path / "source.wav")
    stored = cache.put(key, source, {"text_length": 5, "generation_seconds": 0.1})
    assert cache.get(key) is not None
    stored.wav_path.write_bytes(b"not a wav")
    assert cache.get(key) is None
