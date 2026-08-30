from parakeet_mlx.alignment import AlignedResult, AlignedSentence, AlignedToken

from captionlm.cli import format_timestamp, result_to_srt


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
