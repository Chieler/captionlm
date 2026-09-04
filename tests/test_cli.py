from parakeet_mlx.alignment import AlignedResult, AlignedSentence, AlignedToken

from captionlm.cli import (
    configure_document_bias,
    format_timestamp,
    get_audio_duration,
    result_to_srt,
)
from captionlm.config import SpotterConfig


def test_get_audio_duration():
    duration = get_audio_duration("data/read_aloud/distributed_systems.wav")
    assert 348.0 < duration < 350.0


def test_configure_document_bias_applies_and_resets_the_duration_cap(monkeypatch):
    """One cached model must not carry a long-recording cap into the next file."""
    class Model:
        spotter_config = SpotterConfig(cb_weight=3.0)

    model = Model()
    monkeypatch.setattr(
        "captionlm.cli.get_audio_duration",
        lambda path: 301.0 if path == "long.wav" else 120.0,
    )

    assert configure_document_bias(model, "long.wav", base_weight=3.0) == 1.0
    assert model.spotter_config.cb_weight == 1.0
    assert configure_document_bias(model, "short.wav", base_weight=3.0) == 3.0
    assert model.spotter_config.cb_weight == 3.0


def test_format_timestamp():
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(65.25) == "00:01:05,250"
    assert format_timestamp(3661.001) == "01:01:01,001"


def test_result_to_srt():
    token = AlignedToken(id=0, text=" hello world", start=0.0, duration=1.5)
    sentence = AlignedSentence(text=" hello world", tokens=[token])
    result = AlignedResult(text=" hello world", sentences=[sentence])

    srt = result_to_srt(result)

    assert srt.splitlines() == [
        "1",
        "00:00:00,000 --> 00:00:01,500",
        "hello world",
        "",
    ]
