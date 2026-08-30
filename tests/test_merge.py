from parakeet_mlx.alignment import AlignedToken

from captionlm.merge import merge_spans
from captionlm.vendor.ctc_word_spotter import WSHyp

TIME_RATIO = 0.08  # seconds per encoder frame, arbitrary for these tests


def _tok(text, start, duration):
    return AlignedToken(id=0, text=text, start=start, duration=duration)


def test_single_word_replacement():
    tokens = [
        _tok(" hello", 0.0, 0.16),
        _tok(" kubernetees", 0.16, 0.40),  # mis-transcribed
        _tok(" cluster", 0.56, 0.24),
    ]
    # word 2 spans frames [2, 6] at 0.08s/frame (0.16s to 0.56s)
    hyps = [WSHyp(word="Kubernetes", score=10.0, start_frame=2, end_frame=6)]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert [t.text for t in merged] == [" hello", " Kubernetes", " cluster"]
    assert merged[1].id == -1
    assert merged[1].start == 0.16
    assert abs(merged[1].end - 0.56) < 1e-9


def test_multiword_span_replacement():
    tokens = [
        _tok(" hello", 0.0, 0.16),
        _tok(" san", 0.16, 0.16),
        _tok(" francisco", 0.32, 0.32),
        _tok(" today", 0.64, 0.16),
    ]
    # "san francisco" spans frames [2, 7] (0.16s to 0.64s)
    hyps = [WSHyp(word="San Francisco", score=8.0, start_frame=2, end_frame=7)]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert [t.text for t in merged] == [" hello", " San Francisco", " today"]
    assert merged[1].id == -1
    assert merged[1].start == 0.16
    assert abs(merged[1].end - 0.64) < 1e-9


def test_no_hyps_returns_tokens_unchanged():
    tokens = [_tok(" hello", 0.0, 0.16)]
    assert merge_spans(tokens, [], TIME_RATIO) == tokens


def test_low_overlap_hyp_is_ignored():
    tokens = [_tok(" hello", 0.0, 1.0)]  # 1s-long word
    # hyp only covers the first 0.08s of a 1s-long word -> 8% overlap, below default 30%
    hyps = [WSHyp(word="Wrong", score=10.0, start_frame=0, end_frame=0)]
    merged = merge_spans(tokens, hyps, TIME_RATIO)
    assert [t.text for t in merged] == [" hello"]
