from pathlib import Path

from story_voice_pipeline.benchmark import benchmark_batches
from story_voice_pipeline.config import PipelineConfig
from story_voice_pipeline.engines.fake import FakeTTSEngine


def test_benchmark_measures_each_size_without_cache(tmp_path: Path):
    class Cuda:
        def __init__(self):
            self.resets = 0
            self.syncs = 0

        def synchronize(self, device):
            self.syncs += 1

        def reset_peak_memory_stats(self, device):
            self.resets += 1

        def max_memory_allocated(self, device):
            return 123456

    class Torch:
        cuda = Cuda()

    class Engine(FakeTTSEngine):
        config = PipelineConfig()
        _torch = Torch()

        def __init__(self):
            super().__init__()
            self._references.add("en")
            self.calls = []

        def generate_batch(self, texts, language, reference_language, outputs, **kwargs):
            self.calls.append(len(texts))
            return super().generate_batch(texts, language, reference_language, outputs, **kwargs)

    engine = Engine()
    rows = benchmark_batches(engine, ["One.", "Two.", "Three.", "Four."], "en", tmp_path)
    assert [row["batch_size"] for row in rows] == [1, 2, 4]
    assert engine.calls == [1, 1, 1, 1, 1, 2, 2, 2, 4, 4]
    assert engine._torch.cuda.resets == 3
    assert engine._torch.cuda.syncs == 6
    for row in rows:
        assert row["status"] == "ok"
        assert row["wall_seconds"] > 0
        assert row["total_audio_seconds"] > 0
        assert row["effective_rtf"] == row["wall_seconds"] / row["total_audio_seconds"]
        assert row["peak_cuda_memory_bytes"] == 123456


def test_benchmark_reports_failed_size_and_continues(tmp_path: Path):
    from story_voice_pipeline.errors import EngineError

    class Cuda:
        def synchronize(self, device):
            pass

        def reset_peak_memory_stats(self, device):
            pass

        def max_memory_allocated(self, device):
            return 100

    class Torch:
        cuda = Cuda()

    class Engine(FakeTTSEngine):
        config = PipelineConfig()
        _torch = Torch()

        def generate_batch(self, texts, language, reference_language, outputs, **kwargs):
            if len(texts) == 2:
                raise EngineError("simulated OOM")
            return super().generate_batch(texts, language, reference_language, outputs, **kwargs)

    engine = Engine()
    engine._references.add("en")
    rows = benchmark_batches(engine, ["One."] * 4, "en", tmp_path)
    assert [row["status"] for row in rows] == ["ok", "failed", "ok"]
    assert rows[1]["failure_phase"] == "warmup"
    assert rows[1]["effective_rtf"] is None
    assert rows[1]["peak_cuda_memory_bytes"] is None
