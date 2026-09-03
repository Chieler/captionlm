"""headroom.py against hand-built logprobs -- no model, no GPU, no audio.

Every fixture here was verified by direct execution against the real
vendored functions before being written down (not hand-traced), because
run_word_spotter's beam-search interactions are easy to get subtly wrong
by inspection alone.
"""
import pytest
import numpy as np

from captionlm.headroom import argmax_span_score, raw_spot, unbiased_headroom
from captionlm.vendor.context_graph_ctc import ContextGraphCTC


class FakeModel:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary


def test_raw_spot_finds_strong_single_token_term():
    # vocab: 0="▁hi" (word-start piece), 1="lo" (continuation piece,
    # irrelevant to this graph), blank=2
    logprobs = np.full((4, 3), -50.0, dtype=np.float32)
    logprobs[0, 0] = -0.1
    logprobs[1, 2] = -0.1
    logprobs[2, 1] = -0.1
    logprobs[3, 2] = -0.1
    graph = ContextGraphCTC(blank_id=2)
    graph.add_to_graph([("hi", [[0]])])

    hyps = raw_spot(logprobs, graph, blank_idx=2, cb_weight=0.0)

    assert len(hyps) == 1
    assert hyps[0].word == "hi"
    assert hyps[0].start_frame == 0
    assert hyps[0].end_frame == 0
    assert hyps[0].score == pytest.approx(-0.1, abs=1e-4)


def test_raw_spot_returns_empty_when_blank_dominates_everywhere():
    logprobs = np.full((2, 3), -50.0, dtype=np.float32)
    logprobs[:, 2] = -0.01  # blank near-certain every frame
    graph = ContextGraphCTC(blank_id=2)
    graph.add_to_graph([("hi", [[0]])])

    assert raw_spot(logprobs, graph, blank_idx=2, cb_weight=0.0) == []


def test_argmax_span_score_single_full_overlap_returns_that_words_score():
    word_alignment = [("cat", 5, 9, 2.5)]
    assert argmax_span_score(word_alignment, 5, 9) == 2.5


def test_argmax_span_score_two_word_overlap_is_weighted_not_summed():
    # First overlapping word counts in full; the second is weighted by
    # its overlap percentage. A plain sum would give 3.0, not ~2.33.
    word_alignment = [("the", 0, 2, 1.0), ("cat", 3, 5, 2.0)]
    result = argmax_span_score(word_alignment, 2, 4)
    assert result == pytest.approx(1.0 + (2 / 3) * 2.0)
    assert result != pytest.approx(3.0)


def test_unbiased_headroom_positive_gap_when_spotted_term_beats_a_weak_competing_word():
    # vocab: 0="▁spot" (the term), 1="▁weak1", 2="weak2" (weak1+weak2
    # merge into one low-confidence argmax word spanning frames 0-1).
    # At frame0, "spot" is a real second-best acoustic candidate (-4.0,
    # clears every pruning threshold) even though "weak1" (-3.0) wins
    # argmax there. The merged "weak1weak2" argmax word's own accumulated
    # score (-5.0, two weak frames) is worse than "spot"'s single-frame
    # score, because raw_spot never has to pay for weak2's bad frame.
    model = FakeModel(["▁spot", "▁weak1", "weak2"])
    logprobs = np.full((2, 4), -30.0, dtype=np.float32)
    logprobs[0] = [-4.0, -3.0, -30.0, -5.0]
    logprobs[1] = [-30.0, -30.0, -3.0, -5.0]
    graph = ContextGraphCTC(blank_id=3)
    graph.add_to_graph([("spot", [[0]])])

    result = unbiased_headroom(context_graph=graph, logprobs=logprobs, asr_model=model,
                                blank_idx=3, ctc_ali_token_weight=0.5)

    assert result == {"spot": pytest.approx(1.0)}


def test_unbiased_headroom_has_no_entry_for_a_term_with_no_acoustic_support():
    model = FakeModel(["▁spot"])
    logprobs = np.full((2, 2), -50.0, dtype=np.float32)
    logprobs[:, 1] = -0.01  # blank near-certain -- spot never clears the gate
    graph = ContextGraphCTC(blank_id=1)
    graph.add_to_graph([("spot", [[0]])])

    result = unbiased_headroom(graph, logprobs, model, blank_idx=1)

    assert "spot" not in result


def test_unbiased_headroom_keeps_the_max_across_two_occurrences():
    # "spot" is spotted twice: at frame0 with a +1.0 gap (beats a weak
    # competing word, same construction as the positive-gap test above)
    # and at frame4 with a -0.5 gap (matches argmax's own reading there
    # exactly -- always exactly -ctc_ali_token_weight when spotter and
    # argmax agree, since argmax's score there is spot's own logprob plus
    # that constant). The max must be +1.0, not the average or the last.
    model = FakeModel(["▁spot", "▁weak1", "weak2"])
    logprobs = np.full((5, 4), -30.0, dtype=np.float32)
    logprobs[0] = [-4.0, -3.0, -30.0, -5.0]
    logprobs[1] = [-30.0, -30.0, -3.0, -5.0]
    logprobs[2] = [-30.0, -30.0, -30.0, -0.01]
    logprobs[3] = [-30.0, -30.0, -30.0, -0.01]
    logprobs[4] = [-0.5, -30.0, -30.0, -5.0]
    graph = ContextGraphCTC(blank_id=3)
    graph.add_to_graph([("spot", [[0]])])

    result = unbiased_headroom(graph, logprobs, model, blank_idx=3, ctc_ali_token_weight=0.5)

    assert result == {"spot": pytest.approx(1.0)}
