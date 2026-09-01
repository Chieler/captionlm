import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from serve import SCORES, _cues, _diff  # noqa: E402

SRT = """1
00:00:01,840 --> 00:00:08,240
Let me walk through where agent tooling stands.

2
00:00:08,560 --> 00:00:10,480
Start with a narrow definition.
"""


def test_cues_keep_the_start_time_without_milliseconds():
    assert _cues(SRT) == [
        {"time": "00:00:01", "text": "Let me walk through where agent tooling stands."},
        {"time": "00:00:08", "text": "Start with a narrow definition."},
    ]


def test_a_cue_wrapped_over_two_lines_stays_one_cue():
    wrapped = "1\n00:00:01,840 --> 00:00:08,240\nfirst line\nsecond line\n"
    assert _cues(wrapped) == [{"time": "00:00:01", "text": "first line second line"}]


def test_no_captions_is_no_cues_rather_than_a_crash():
    assert _cues("") == []


def test_diff_reports_only_what_the_second_model_changed():
    # Real span from the read-aloud set at 110m.
    assert _diff(
        "Let me walk through where agent's coding today stands.",
        "Let me walk through where agent tooling today stands.",
    ) == [{"from": "agent's coding", "to": "agent tooling"}]


def test_diff_of_an_untouched_transcript_is_empty():
    text = "the storage engine makes durability cheap"
    assert _diff(text, text) == []


def test_diff_records_a_deletion_with_an_empty_replacement():
    # The second model dropping the reader's false start. The UI skips these
    # rather than rendering an arrow pointing at nothing.
    assert _diff("the strategy is always part is almost always", "the strategy is almost always") == [
        {"from": "always part is", "to": ""}
    ]


def test_every_toggle_combination_has_a_measured_score():
    # The UI reads these straight out of this table; a missing pair would be a
    # KeyError on /api/state the moment someone flipped a switch.
    for preset in ("110m", "1.1b"):
        for second in (False, True):
            row = SCORES[(preset, second)]
            assert set(row) == {"recall", "wer", "speed"}


def test_the_second_opinion_is_scored_better_than_going_without():
    for preset in ("110m", "1.1b"):
        assert float(SCORES[(preset, True)]["recall"]) > float(SCORES[(preset, False)]["recall"])
        assert float(SCORES[(preset, True)]["wer"]) < float(SCORES[(preset, False)]["wer"])
