"""The rolling-commit logic, with the model stubbed out.

What matters here is not the acoustics -- that is the library's -- but that a
word is shown once, never revised, and never silently dropped.
"""
import numpy as np
import pytest
from parakeet_mlx.alignment import AlignedToken, sentences_to_result, tokens_to_sentences

from captionlm import live

RATE = 16000


class FakeModel:
    """Emits one token per second of buffered audio, at predictable times.

    Token i covers [i, i+1) seconds of the buffer, so a test can reason about
    exactly which tokens should commit for a given horizon.
    """

    class preprocessor_config:
        sample_rate = RATE

    def __init__(self):
        self.calls = 0

    def generate(self, mel, *, decoding_config=None):
        self.calls += 1
        seconds = int(mel)  # the stubbed get_logmel passes buffer seconds through
        tokens = []
        for i in range(seconds):
            tokens.append(AlignedToken(id=i, text=f" w{i}", start=float(i), duration=1.0))
        return [sentences_to_result(tokens_to_sentences(tokens))]


@pytest.fixture
def session(monkeypatch):
    # The stub hands generate() the buffer length in seconds instead of a mel.
    monkeypatch.setattr(live, "get_logmel", lambda audio, cfg: len(audio) // RATE)
    return live.LiveSession(FakeModel(), window=30.0, horizon=4.0, interval=2.0)


def _silence(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def test_nothing_commits_before_the_first_interval(session):
    assert session.add_audio(_silence(1.0)) == []
    assert session.committed_until == 0.0


def test_only_words_older_than_the_horizon_commit(session):
    # 10s of audio, 4s horizon: tokens ending at or before 6.0 commit.
    # add_audio decodes on the spot here -- 10s is well past the interval.
    committed = session.add_audio(_silence(10.0))
    assert [t.text.strip() for t in committed] == ["w0", "w1", "w2", "w3", "w4", "w5"]
    assert session.committed_until == 6.0


def test_a_word_is_never_committed_twice(session):
    first = session.add_audio(_silence(10.0))
    later = session.add_audio(_silence(4.0))
    assert first and later
    assert {t.text for t in later}.isdisjoint({t.text for t in first})


def test_commits_are_monotonic_and_never_repeat(session):
    seen = []
    for _ in range(6):
        seen += session.add_audio(_silence(3.0))
    starts = [t.start for t in seen]
    assert starts == sorted(starts)
    assert len(starts) == len(set(starts))


def test_a_token_straddling_the_commit_point_is_dropped_not_duplicated(session):
    session.add_audio(_silence(10.0))     # commits through 6.0
    session.committed_until = 5.5         # pretend it landed mid-token
    again = session.add_audio(_silence(4.0))
    assert again                          # it still emits
    assert all(t.start >= 5.5 for t in again)


def test_finish_flushes_the_tail(session):
    session.add_audio(_silence(10.0))     # commits w0..w5
    tail = session.finish()
    assert [t.text.strip() for t in tail] == ["w6", "w7", "w8", "w9"]


def test_a_long_delivery_is_decoded_before_it_is_trimmed(session):
    # 40s arriving at once into a 30s window. The oldest 10s must be decoded
    # rather than trimmed away unheard.
    committed = session.add_audio(_silence(40.0))
    assert committed[0].text.strip() == "w0"
    assert len(session._buffer) <= 30 * RATE


def test_a_gap_in_the_buffer_is_reported_rather_than_hidden(session):
    # The guard for audio discarded before any pass committed it. Reaching it
    # requires forcing the state, which is the point -- it should not happen.
    session.add_audio(_silence(10.0))
    session._buffer_start = 20.0
    session.committed_until = 5.0
    session._decode()
    assert session.degraded is True
    assert session.committed_until >= 20.0


def test_a_slow_pass_slows_the_cadence_and_says_so(monkeypatch):
    monkeypatch.setattr(live, "get_logmel", lambda audio, cfg: len(audio) // RATE)
    ticks = [0.0, 3.0]                    # one pass costing 3s of wall time
    monkeypatch.setattr(live, "_now", lambda: ticks.pop(0) if ticks else 3.0)
    s = live.LiveSession(FakeModel(), window=30.0, horizon=8.0, interval=2.0)

    s.add_audio(_silence(10.0))
    assert s.interval == 3.0               # raised to the decode cost, under the 4.0 ceiling
    assert s.degraded is True


def test_a_recovered_session_returns_to_its_configured_cadence(monkeypatch):
    monkeypatch.setattr(live, "get_logmel", lambda audio, cfg: len(audio) // RATE)
    ticks = [0.0, 3.0]                     # slow pass, then instant ones
    monkeypatch.setattr(live, "_now", lambda: ticks.pop(0) if ticks else 3.0)
    s = live.LiveSession(FakeModel(), window=30.0, horizon=8.0, interval=2.0)

    s.add_audio(_silence(10.0))
    for _ in range(3):
        s.add_audio(_silence(5.0))
    assert s.interval == 2.0
    assert s.degraded is False
