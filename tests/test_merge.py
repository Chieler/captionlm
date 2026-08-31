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


def test_trailing_punctuation_is_preserved():
    tokens = [
        _tok(" hello", 0.0, 0.16),
        _tok(" kubernetees", 0.16, 0.32),  # mis-transcribed
        _tok(".", 0.48, 0.08),  # sentence-ending punctuation, no leading space
    ]
    # word + period together span frames [2, 6] at 0.08s/frame (0.16s to 0.56s)
    hyps = [WSHyp(word="Kubernetes", score=10.0, start_frame=2, end_frame=6)]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert [t.text for t in merged] == [" hello", " Kubernetes."]
    assert merged[1].id == -1


def test_leading_article_is_not_consumed_by_a_shorter_keyword():
    # Measured failure: 'the hard parts' -> 'hard parts'. The article's
    # frames overlap the keyword enough to clear intersection_threshold,
    # and the whole range was replaced by the keyword alone.
    tokens = [
        _tok(" the", 0.16, 0.16),
        _tok(" hard", 0.32, 0.16),
        _tok(" parts", 0.48, 0.16),
    ]
    # hyp covers frames [3, 7] = 0.24s..0.64s, which clips half of " the"
    hyps = [WSHyp(word="hard parts", score=9.0, start_frame=3, end_frame=7)]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert "".join(t.text for t in merged).split() == ["the", "hard", "parts"]


def test_trims_the_lower_overlap_edge_first():
    # Both edges clear the threshold, so the range holds three words for a
    # two-word keyword. " a" overlaps the hypothesis 50%, " token" 100%,
    # so " a" is the one that is not part of the keyword.
    tokens = [
        _tok(" a", 0.00, 0.16),
        _tok(" fencing", 0.16, 0.24),
        _tok(" token", 0.40, 0.24),
        _tok(" now", 0.64, 0.16),
    ]
    # frames [1, 7] = 0.08s..0.64s, which clips half of " a"
    hyps = [WSHyp(word="fencing token", score=9.0, start_frame=1, end_frame=7)]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert "".join(t.text for t in merged).split() == [
        "a", "fencing", "token", "now",
    ]


def test_two_hyps_do_not_clobber_each_other():
    tokens = [
        _tok(" alpha", 0.0, 0.16),
        _tok(" beta", 0.16, 0.16),
        _tok(" gamma", 0.32, 0.16),
    ]
    hyps = [
        WSHyp(word="Beta-Term", score=10.0, start_frame=2, end_frame=3),   # covers only " beta" (0.16-0.32)
        WSHyp(word="Gamma-Term", score=8.0, start_frame=4, end_frame=5),   # covers only " gamma" (0.32-0.48)
    ]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert [t.text for t in merged] == [" alpha", " Beta-Term", " Gamma-Term"]
    assert merged[1].id == -1
    assert merged[2].id == -1
