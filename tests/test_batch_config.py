from dataclasses import replace
from pathlib import Path

import pytest

from story_voice_pipeline.cli import _config_from_args, build_parser
from story_voice_pipeline.config import PipelineConfig
from story_voice_pipeline.control_service.config import ControlSettings
from story_voice_pipeline.errors import PipelineError


def test_batch_environment_and_cli_override(monkeypatch):
    monkeypatch.delenv("QWEN_BATCH_SIZE", raising=False)
    assert PipelineConfig().qwen_batch_size == 1
    monkeypatch.setenv("QWEN_BATCH_SIZE", "4")
    assert PipelineConfig().qwen_batch_size == 4
    assert ControlSettings.from_environment().pipeline_config.qwen_batch_size == 4
    args = build_parser().parse_args(
        [
            "smoke-test",
            "--reference-audio",
            "a.wav",
            "--reference-transcript-file",
            "a.txt",
            "--output-dir",
            "out",
            "--qwen-batch-size",
            "2",
            "--dtype",
            "float32",
        ]
    )
    config = _config_from_args(args, Path("out"))
    assert config.qwen_batch_size == 2
    assert config.dtype == "float32"
    assert (
        config.generation_parameters() == replace(config, qwen_batch_size=1).generation_parameters()
    )
    assert "qwen_batch_size" not in config.generation_parameters()


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_batch_size(value):
    with pytest.raises(PipelineError, match="positive integer"):
        PipelineConfig(qwen_batch_size=value)


def test_invalid_batch_environment(monkeypatch):
    monkeypatch.setenv("QWEN_BATCH_SIZE", "oops")
    with pytest.raises(PipelineError, match="QWEN_BATCH_SIZE"):
        PipelineConfig()
