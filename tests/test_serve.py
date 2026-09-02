import os
import shutil
import sys
import tempfile
import urllib.parse

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


def test_a_job_that_throws_stops_the_row_it_died_on():
    # Otherwise the row keeps status "working" and its progress bar animates
    # for the rest of the session, under an error banner, with nothing behind it.
    from serve import run_job, state

    state["files"] = [{"name": "a.wav", "status": "working", "stage": "transcribing", "progress": 40}]
    run_job("no-such-preset", False)

    assert state["files"][0]["status"] == "failed"
    assert state["files"][0]["progress"] == 0
    assert state["error"] and not state["running"]


def test_a_document_with_no_recording_is_still_listed(tmp_path, monkeypatch):
    # find_pairs is keyed on audio, so a lone document produces no row. Without
    # this the file lands on disk and appears nowhere, which reads as a failed
    # upload.
    import serve

    (tmp_path / "notes.txt").write_text("quorum, Paxos")
    (tmp_path / "standup.txt").write_text("paired with the recording")
    (tmp_path / "README.md").write_text("the drop-off's own instructions")
    monkeypatch.setattr(serve, "DROPOFF", str(tmp_path))

    rows = [{"name": "standup.m4a", "docs": ["standup.txt"], "shared": []}]
    assert serve.strays(rows) == ["notes.txt"]


def test_deleting_a_recording_takes_its_generated_files_with_it():
    # Leaving them behind means re-dropping a recording of the same name shows
    # the previous run's captions.
    import serve

    d = tempfile.mkdtemp()
    for suffix in (".wav", ".srt", ".terms.txt", ".converted.wav"):
        open(os.path.join(d, "standup" + suffix), "w").write("x")
    open(os.path.join(d, "keep.wav"), "w").write("x")

    serve.DROPOFF = d
    try:
        code, body = _delete(serve, "standup.wav")
        assert code == 200 and body["ok"]
        assert sorted(os.listdir(d)) == ["keep.wav"]

        assert _delete(serve, "../../SETUP.md")[0] == 404      # traversal
        assert _delete(serve, "evil.sh")[0] == 400             # not an audio or document
        assert _delete(serve, "gone.wav")[0] == 404
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _delete(serve, name):
    """Drive Handler.do_DELETE without a socket."""
    captured = {}

    class Fake(serve.Handler):
        def __init__(self):
            self.path = "/api/file?name=" + urllib.parse.quote(name)

        def _json(self, obj, code=200):
            captured["code"], captured["body"] = code, obj

    Fake().do_DELETE()
    return captured["code"], captured["body"]


def test_captions_left_by_a_recording_removed_outside_the_ui_are_listed():
    # Generated files are excluded from pairing, so without this they sit in
    # the drop-off invisible and unremovable.
    import serve

    d = tempfile.mkdtemp()
    for name in ("ghost.srt", "ghost.terms.txt", "keep.srt", "keep.terms.txt"):
        open(os.path.join(d, name), "w").write("x")
    open(os.path.join(d, "keep.wav"), "w").write("x")

    serve.DROPOFF = d
    try:
        rows = [{"name": "keep.wav", "docs": [], "shared": []}]
        assert serve.strays(rows) == ["ghost.srt", "ghost.terms.txt"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_decoding_cache_of_a_gone_recording_is_swept_not_listed():
    # It is the size of the recording and rebuilt on demand, unlike the .srt,
    # which is what the user came for and is never deleted for them.
    import serve

    d = tempfile.mkdtemp()
    for name in ("ghost.converted.wav", "keep.converted.wav", "keep.m4a"):
        open(os.path.join(d, name), "w").write("x")

    serve.DROPOFF = d
    try:
        assert serve.strays([{"name": "keep.m4a", "docs": [], "shared": []}]) == []
        assert sorted(os.listdir(d)) == ["keep.converted.wav", "keep.m4a"]
    finally:
        shutil.rmtree(d, ignore_errors=True)
