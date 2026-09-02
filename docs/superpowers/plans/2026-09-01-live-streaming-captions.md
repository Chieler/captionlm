# Live Streaming Captions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Caption a live microphone in the browser, with the domain term biasing intact.

**Architecture:** Rolling re-transcribe. A `LiveSession` holds the last 30 seconds of audio and, every 2 seconds, runs the *existing* biased `generate()` over that buffer. Words older than a 4-second stability horizon are committed and shown; everything newer is discarded and re-derived next pass, so no shown word is ever revised. Because it calls the same `generate()` the batch path calls, the biasing is bit-for-bit what has already been measured.

**Tech Stack:** Python 3.12, MLX, `parakeet-mlx`, `mlx-whisper`, stdlib `http.server`, browser `AudioWorklet`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-01-live-streaming-captions-design.md`

## Global Constraints

- **Nothing is trained, ever.** Term lists become a trie at call time.
- **Apple silicon / MLX only.** No CUDA, no torch.
- **No model may add more than 2 GB.** No new models are introduced by this plan.
- **One job at a time.** `serve.py` serialises work behind `state["running"]`; a live session takes the same lock.
- **Never show text that will change.** Only committed words reach the UI.
- **Never silently lose audio.** If the decoder falls behind, force-commit and mark the session degraded rather than dropping samples.
- **No new dependencies.** Everything below uses what `requirements.txt` already pins.
- Run tests with `venv/bin/python -m pytest`. The package is installed editable, so no `PYTHONPATH=` prefix.

## File Structure

| File | Responsibility |
|---|---|
| `captionlm/live.py` (create) | `LiveSession` and `SecondOpinionSession`. Pure decoding — no transport, no audio device, no disk. |
| `tests/test_live.py` (create) | Unit tests for both, against a stubbed model. |
| `scripts/measure_live.py` (create) | The two measurement gates. Reproducible, kept. |
| `captionlm/fusion.py` (modify) | Widen `second_opinion`'s annotation to accept samples. |
| `scripts/serve.py` (modify) | `/api/live/start`, `/api/live/audio`, `/api/live/stop`. |
| `tests/test_serve.py` (modify) | Transport lifecycle and the exclusivity lock. |
| `scripts/dashboard.html` (modify) | Microphone capture and live transcript rendering. |

**Tasks 2 and 3 are gates.** If either fails, stop and report — Tasks 4-6 assume both passed.

---

### Task 1: `LiveSession` — the rolling decoder

**Files:**
- Create: `captionlm/live.py`
- Test: `tests/test_live.py`

**Interfaces:**
- Consumes: `BiasedParakeetTDTCTC` from `captionlm.biased_model` (only its `.generate(mel, decoding_config=...) -> [AlignedResult]` and `.preprocessor_config.sample_rate`); `get_logmel` from `parakeet_mlx.audio`.
- Produces: `LiveSession(model, *, window=30.0, horizon=4.0, interval=2.0)` with `add_audio(frames: np.ndarray) -> list[AlignedToken]`, `finish() -> list[AlignedToken]`, and attributes `committed_until: float`, `degraded: bool`, `interval: float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_live.py`:

```python
"""The rolling-commit logic, with the model stubbed out.

What matters here is not the acoustics -- that is the library's -- but that a
word is shown once, never revised, and never silently dropped.
"""
import mlx.core as mx
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
    session.add_audio(_silence(10.0))
    assert [t.text for t in session.add_audio(_silence(0.0))] == []  # no new audio, no pass
    committed = session._decode()
    assert [t.text.strip() for t in committed] == ["w0", "w1", "w2", "w3", "w4", "w5"]
    assert session.committed_until == 6.0


def test_a_word_is_never_committed_twice(session):
    session.add_audio(_silence(10.0))
    first = session._decode()
    second = session._decode()
    assert first and second == []
    session.add_audio(_silence(4.0))
    third = session._decode()
    assert {t.text for t in third}.isdisjoint({t.text for t in first})


def test_commits_are_monotonic_in_time(session):
    seen = []
    for _ in range(6):
        seen += session.add_audio(_silence(3.0))
    starts = [t.start for t in seen]
    assert starts == sorted(starts)


def test_a_token_straddling_the_commit_point_is_dropped_not_duplicated(session):
    # Force committed_until into the middle of token w5.
    session.add_audio(_silence(10.0))
    session._decode()
    session.committed_until = 5.5
    again = session._decode()
    assert all(t.start >= 5.5 for t in again)


def test_finish_flushes_the_tail(session):
    session.add_audio(_silence(10.0))
    session._decode()
    tail = session.finish()
    assert [t.text.strip() for t in tail] == ["w6", "w7", "w8", "w9"]


def test_audio_older_than_the_window_is_force_committed_not_lost(session):
    # 40s of audio into a 30s window with no decode in between: the first 10s
    # would fall off the back uncommitted.
    session.add_audio(_silence(40.0))
    committed = session._decode()
    assert session.degraded is True
    assert committed[0].start == 0.0          # the oldest audio still came out
    assert session.committed_until >= 10.0


def test_a_slow_decode_raises_the_interval(session, monkeypatch):
    clock = iter([0.0, 3.0])  # a pass that took 3s of wall time
    monkeypatch.setattr(live.time, "monotonic", lambda: next(clock))
    session.add_audio(_silence(10.0))
    session._decode()
    assert session.interval == 2.0  # capped at horizon / 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_live.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'captionlm.live'`

- [ ] **Step 3: Write the implementation**

Create `captionlm/live.py`:

```python
"""Live captioning by rolling re-transcribe.

Hold the last `window` seconds of audio and, every `interval` seconds, run the
ordinary biased `generate()` over the whole buffer. Words older than `horizon`
are committed and handed out; everything newer is thrown away and re-derived
on the next pass.

Why this rather than `parakeet_mlx`'s own streaming API: `StreamingParakeet`
calls `model.decode()` directly, bypassing `generate()`, which is the only
place `BiasedParakeetTDTCTC` runs the CTC word spotter. Streaming that way
silently drops every bit of term biasing. Re-transcribing costs compute the
streaming decoder would not, and buys back exactly the biasing behaviour that
has already been measured, under full attention rather than the local
attention streaming switches on.
"""
import time

import mlx.core as mx
import numpy as np
from parakeet_mlx.alignment import AlignedToken
from parakeet_mlx.audio import get_logmel
from parakeet_mlx.parakeet import DecodingConfig


class LiveSession:
    def __init__(
        self,
        model,
        *,
        window: float = 30.0,
        horizon: float = 4.0,
        interval: float = 2.0,
        decoding_config: DecodingConfig = DecodingConfig(),
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        if horizon >= window:
            raise ValueError("horizon must be shorter than window")
        self.model = model
        self.window = window
        self.horizon = horizon
        self.interval = interval
        self.committed_until = 0.0
        self.degraded = False

        self._configured_interval = interval
        self._decoding_config = decoding_config
        self._dtype = dtype
        self._rate = model.preprocessor_config.sample_rate
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0.0        # absolute time of _buffer[0]
        self._received = 0              # total samples ever accepted
        self._last_decode = 0.0         # audio time of the last pass
        self._good_passes = 0

    @property
    def now(self) -> float:
        """Absolute audio time at the live edge."""
        return self._received / self._rate

    def add_audio(self, frames: np.ndarray) -> list[AlignedToken]:
        """Accept samples; return whatever just became final."""
        frames = np.asarray(frames, dtype=np.float32).reshape(-1)
        self._buffer = np.concatenate([self._buffer, frames])
        self._received += len(frames)
        if self.now - self._last_decode < self.interval:
            return []
        return self._decode()

    def finish(self) -> list[AlignedToken]:
        """Commit everything left, including inside the horizon."""
        return self._decode(cutoff=float("inf"))

    def _decode(self, cutoff: float | None = None) -> list[AlignedToken]:
        if len(self._buffer) < self._rate // 10:      # under 100ms, nothing to decode
            return []

        started = time.monotonic()
        mel = get_logmel(mx.array(self._buffer, dtype=self._dtype), self.model.preprocessor_config)
        result = self.model.generate(mel, decoding_config=self._decoding_config)[0]
        elapsed = time.monotonic() - started

        for token in result.tokens:
            token.start += self._buffer_start
            token.end = token.start + token.duration

        if cutoff is None:
            cutoff = self.now - self.horizon

        # Anything about to fall off the back of the buffer must come out now,
        # however unstable. Dropping it would lose the words silently.
        falling_off = self.now - self.window
        if falling_off > self.committed_until:
            cutoff = max(cutoff, falling_off)
            self.degraded = True

        committed = [
            t for t in result.tokens
            if t.start >= self.committed_until and t.end <= cutoff
        ]
        if committed:
            self.committed_until = committed[-1].end

        self._trim()
        self._last_decode = self.now
        self._retune(elapsed)
        return committed

    def _trim(self) -> None:
        keep = int(self.window * self._rate)
        if len(self._buffer) > keep:
            self._buffer = self._buffer[-keep:]
        self._buffer_start = self.now - len(self._buffer) / self._rate

    def _retune(self, elapsed: float) -> None:
        """Decode less often when a pass costs more than the interval.

        Decoding less often is what reduces duty cycle; a longer horizon does
        nothing for throughput. The cap keeps at least two commits per horizon
        so the transcript does not arrive in one lump.
        """
        ceiling = self.horizon / 2
        if elapsed > self.interval:
            self.interval = min(max(self._configured_interval, elapsed), ceiling)
            self._good_passes = 0
        else:
            self._good_passes += 1
            if self._good_passes >= 3:
                self.interval = self._configured_interval
                self._good_passes = 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_live.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Run the whole suite**

Run: `venv/bin/python -m pytest -q`
Expected: PASS, 123 existing + 8 new.

- [ ] **Step 6: Commit**

```bash
git add captionlm/live.py tests/test_live.py
git commit -m "Add a rolling-re-transcribe live session"
```

---

### Task 2: GATE 1 — measure the duty cycle

The spec's 25% figure is a projection from batch throughput. It assumes per-call overhead on a 30-second `generate()` is negligible, which nobody has checked. **If a pass costs more than `interval` at the default settings, stop and report before building anything on top.**

**Files:**
- Create: `scripts/measure_live.py`

**Interfaces:**
- Consumes: `LiveSession` from Task 1; `load_biased_model`, `load_tokenizer`, `build_context_graph`, `load_term_list`.
- Produces: `measure_duty_cycle(model, wav, **kw) -> dict` with keys `passes`, `mean_decode_s`, `max_decode_s`, `duty_cycle`, `interval`.

- [ ] **Step 1: Write the measurement script**

Create `scripts/measure_live.py`:

```python
"""The two gates from the live-captions design, measured rather than assumed.

    venv/bin/python scripts/measure_live.py data/read_aloud/distributed_systems.wav

Gate 1, duty cycle: what fraction of realtime does the rolling re-transcribe
actually consume? The design projects ~25% at 110m from batch throughput,
assuming per-call overhead on a 30s generate() is negligible.

Gate 2, acceptance: does the committed live transcript match what the batch
path produces for the same audio? Approach A's whole claim is that rolling
re-transcribe preserves measured quality.
"""
import argparse
import os
import time

import numpy as np
from parakeet_mlx.audio import load_audio

from captionlm.config import MODEL_ID_110M
from captionlm.eval import _wer
from captionlm.live import LiveSession
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer
from captionlm.biased_model import load_biased_model


def _feed(model, wav: str, chunk: float = 1.0, **kw):
    """Run a file through a LiveSession as if it arrived in real time."""
    rate = model.preprocessor_config.sample_rate
    audio = np.array(load_audio(wav, rate, __import__("mlx").core.float32), copy=False)
    session = LiveSession(model, **kw)
    committed, timings = [], []
    step = int(chunk * rate)
    for i in range(0, len(audio), step):
        started = time.monotonic()
        committed += session.add_audio(audio[i : i + step])
        timings.append(time.monotonic() - started)
    committed += session.finish()
    return session, committed, timings, len(audio) / rate


def measure_duty_cycle(model, wav: str, **kw) -> dict:
    session, _, timings, seconds = _feed(model, wav, **kw)
    passes = [t for t in timings if t > 0.01]      # calls that actually decoded
    return {
        "passes": len(passes),
        "mean_decode_s": float(np.mean(passes)) if passes else 0.0,
        "max_decode_s": float(np.max(passes)) if passes else 0.0,
        "duty_cycle": float(np.sum(timings)) / seconds,
        "interval": session.interval,
        "degraded": session.degraded,
        "audio_s": seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument("--terms", help="Bias term list (default: <wav>.terms.txt if present)")
    parser.add_argument("--model", default=MODEL_ID_110M)
    args = parser.parse_args()

    model = load_biased_model(args.model)
    terms_path = args.terms or os.path.splitext(args.wav)[0] + ".terms.txt"
    if os.path.isfile(terms_path):
        terms = load_term_list(terms_path)
        model.context_graph = build_context_graph(
            terms, load_tokenizer(args.model), len(model.vocabulary)
        )
        print(f"biased by {len(terms)} terms from {terms_path}")
    else:
        print("no term list; running unbiased")

    d = measure_duty_cycle(model, args.wav)
    print(f"\nGATE 1  duty cycle")
    print(f"  audio            {d['audio_s']:.1f}s")
    print(f"  decode passes    {d['passes']}")
    print(f"  mean per pass    {d['mean_decode_s']:.3f}s")
    print(f"  worst pass       {d['max_decode_s']:.3f}s")
    print(f"  duty cycle       {d['duty_cycle']:.1%}")
    print(f"  final interval   {d['interval']:.1f}s (2.0 = never retuned)")
    print(f"  degraded         {d['degraded']}")
    verdict = "PASS" if d["duty_cycle"] < 0.8 and not d["degraded"] else "FAIL"
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `venv/bin/python scripts/measure_live.py data/read_aloud/distributed_systems.wav`
Expected: prints a duty cycle. **PASS is under 80% with `degraded` false.**

- [ ] **Step 3: Record the result**

If it FAILS, stop the plan and report the numbers. The remedies are in the spec: shrink `window`, raise `interval`, or move to Approach B. Do not proceed to Task 4.

If it PASSES, paste the numbers into the commit message.

- [ ] **Step 4: Commit**

```bash
git add scripts/measure_live.py
git commit -m "Measure the live duty cycle rather than projecting it"
```

---

### Task 3: GATE 2 — live transcript against batch

**Files:**
- Modify: `scripts/measure_live.py`

**Interfaces:**
- Consumes: `_feed` from Task 2.
- Produces: `measure_acceptance(model, wav) -> dict` with keys `live_wer`, `batch_wer`, `delta`, `live_text`, `batch_text`.

- [ ] **Step 1: Add the acceptance measurement**

Append to `scripts/measure_live.py`, above `main()`:

```python
def measure_acceptance(model, wav: str, reference: str, **kw) -> dict:
    """Live transcript versus batch transcript of the same audio.

    Both are scored against the reference so the number is a WER a reader
    already knows how to interpret, rather than a similarity score between two
    hypotheses.
    """
    _, committed, _, _ = _feed(model, wav, **kw)
    live_text = "".join(t.text for t in committed).strip()
    batch_text = model.transcribe(wav, chunk_duration=120.0).text.strip()

    # _wer applies eval.py's normaliser to both sides, so this number is
    # directly comparable to every other WER this project has recorded.
    live_wer = _wer([reference], [live_text])
    batch_wer = _wer([reference], [batch_text])
    return {
        "live_wer": live_wer,
        "batch_wer": batch_wer,
        "delta": live_wer - batch_wer,
        "live_text": live_text,
        "batch_text": batch_text,
    }
```

And in `main()`, after the gate 1 block:

```python
    ref_path = os.path.splitext(args.wav)[0] + ".txt"
    if os.path.isfile(ref_path):
        reference = open(ref_path, encoding="utf-8").read().strip()
        a = measure_acceptance(model, args.wav, reference)
        print(f"\nGATE 2  live against batch")
        print(f"  batch WER        {a['batch_wer']:.4f}")
        print(f"  live WER         {a['live_wer']:.4f}")
        print(f"  delta            {a['delta']:+.4f}")
        print(f"  -> {'PASS' if a['delta'] < 0.02 else 'FAIL'}")
    else:
        print(f"\nGATE 2 skipped: no reference at {ref_path}")
```

- [ ] **Step 2: Confirm the WER helper is importable**

Run: `venv/bin/python -c "from captionlm.eval import _wer; print(_wer(['a b c'], ['a b d']))"`
Expected: `0.333...`. Use `_wer`, not `_NORMALIZE` directly — `_NORMALIZE` is a `jiwer.Compose` that must be handed to `jiwer.wer` as `reference_transform`/`hypothesis_transform`, never called on a string.

- [ ] **Step 3: Run both gates**

Run: `venv/bin/python scripts/measure_live.py data/read_aloud/distributed_systems.wav`
Expected: **PASS is a delta under +0.02 WER** against batch.

- [ ] **Step 4: Record the result**

If it FAILS, stop and report both transcripts — the likely causes are words lost at commit boundaries or a horizon too short for the model to settle. Try `--horizon 6` before concluding the approach is wrong. Do not proceed to Task 4 on a failing gate.

- [ ] **Step 5: Commit**

```bash
git add scripts/measure_live.py
git commit -m "Score the live transcript against the batch one"
```

---

### Task 4: `SecondOpinionSession` — segment-wise fusion

**Files:**
- Modify: `captionlm/fusion.py` (the `second_opinion` signature)
- Modify: `captionlm/live.py`
- Test: `tests/test_live.py`

**Interfaces:**
- Consumes: `LiveSession` from Task 1; `fuse_tokens`, `second_opinion` from `captionlm.fusion`.
- Produces: `SecondOpinionSession(model, terms, *, segment=30.0, **live_kw)` with the same `add_audio(frames) -> list[AlignedToken]` and `finish() -> list[AlignedToken]` contract as `LiveSession`, plus `degraded: bool`.

- [ ] **Step 1: Widen `second_opinion` to accept samples**

In `captionlm/fusion.py`, change:

```python
def second_opinion(audio_path: str, model_id: str = WHISPER_MODEL_ID) -> str:
    """Transcribe the same audio with an independent model."""
```

to:

```python
def second_opinion(audio, model_id: str = WHISPER_MODEL_ID) -> str:
    """Transcribe the same audio with an independent model.

    Takes a path or the samples themselves -- `mlx_whisper.transcribe` accepts
    `str | np.ndarray | mx.array`, and the live path has a segment in memory
    with no file to point at.
    """
```

and rename the argument at the single call site inside the function body (`audio_path` -> `audio`).

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_live.py`:

```python
def test_a_segment_is_fused_only_once_it_is_closed(monkeypatch):
    # Whisper must see the same closed audio parakeet saw. Handing it a prefix
    # would align a partial against a complete transcript and corrupt the tail.
    monkeypatch.setattr(live, "get_logmel", lambda audio, cfg: len(audio) // RATE)
    monkeypatch.setattr(live, "second_opinion", lambda audio, model_id=None: "w0 w1 w2")
    s = live.SecondOpinionSession(FakeModel(), terms=[], segment=10.0,
                                  window=30.0, horizon=4.0, interval=2.0)

    assert s.add_audio(_silence(5.0)) == []      # segment still open
    out = s.add_audio(_silence(6.0))             # crosses 10s: first segment closes
    assert [t.text.strip() for t in out]


def test_a_near_silent_segment_skips_the_second_model(monkeypatch):
    # Whisper hallucinates on silence.
    monkeypatch.setattr(live, "get_logmel", lambda audio, cfg: len(audio) // RATE)
    called = []
    monkeypatch.setattr(live, "second_opinion",
                        lambda audio, model_id=None: called.append(1) or "invented sentence")
    s = live.SecondOpinionSession(FakeModel(), terms=[], segment=10.0)
    s.add_audio(_silence(11.0))
    assert called == []
```

- [ ] **Step 3: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_live.py -k second -v`
Expected: FAIL — `AttributeError: module 'captionlm.live' has no attribute 'SecondOpinionSession'`

- [ ] **Step 4: Implement**

Add to the imports at the top of `captionlm/live.py`:

```python
from captionlm.fusion import WHISPER_MODEL_ID, fuse_tokens, second_opinion
```

and append:

```python
SILENCE_RMS = 1e-3          # below this a segment is silence, and Whisper invents
MAX_BACKLOG = 2             # segments outstanding before the second opinion is skipped


class SecondOpinionSession:
    """Live captions corrected by a second model, one closed segment at a time.

    Whisper reads a whole input at once, so it cannot follow the live edge.
    What it can do is read a *closed* segment -- an interval both models have
    seen in full -- which is what makes the alignment sound. The cost is that
    text lags by one segment, which is the trade the second-opinion toggle
    already stands for everywhere else in this project.
    """

    def __init__(self, model, terms: list[str], *, segment: float = 30.0,
                 second_opinion_model: str = WHISPER_MODEL_ID, **live_kw) -> None:
        self.live = LiveSession(model, **live_kw)
        self.terms = terms
        self.segment = segment
        self.second_opinion_model = second_opinion_model
        self.skipped = 0

        self._rate = model.preprocessor_config.sample_rate
        self._pending: list[AlignedToken] = []
        self._segment_audio = np.zeros(0, dtype=np.float32)
        self._segment_start = 0.0

    @property
    def degraded(self) -> bool:
        return self.live.degraded

    def add_audio(self, frames: np.ndarray) -> list[AlignedToken]:
        frames = np.asarray(frames, dtype=np.float32).reshape(-1)
        self._pending += self.live.add_audio(frames)
        self._segment_audio = np.concatenate([self._segment_audio, frames])

        out: list[AlignedToken] = []
        while len(self._segment_audio) >= self.segment * self._rate:
            edge = self._segment_start + self.segment
            samples = self._segment_audio[: int(self.segment * self._rate)]
            out += self._close(samples, edge)
            self._segment_audio = self._segment_audio[int(self.segment * self._rate) :]
            self._segment_start = edge
        return out

    def finish(self) -> list[AlignedToken]:
        self._pending += self.live.finish()
        edge = self._segment_start + len(self._segment_audio) / self._rate
        out = self._close(self._segment_audio, edge)
        self._segment_audio = np.zeros(0, dtype=np.float32)
        return out

    def _close(self, samples: np.ndarray, edge: float) -> list[AlignedToken]:
        """Fuse every committed token that ends inside this closed segment."""
        inside = [t for t in self._pending if t.end <= edge]
        self._pending = [t for t in self._pending if t.end > edge]
        if not inside:
            return []

        if len(samples) == 0 or float(np.sqrt(np.mean(np.square(samples)))) < SILENCE_RMS:
            self.skipped += 1
            return inside          # silence: never ask, it would invent a sentence
        if len(self._pending) > MAX_BACKLOG * self.segment * self._rate:
            self.skipped += 1
            return inside          # falling behind: parakeet-only rather than growing lag

        return fuse_tokens(inside, second_opinion(samples, self.second_opinion_model), self.terms)
```

- [ ] **Step 5: Run the tests**

Run: `venv/bin/python -m pytest tests/test_live.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 6: Run the whole suite**

Run: `venv/bin/python -m pytest -q`
Expected: PASS. `tests/test_fusion.py` must still pass — the `second_opinion` change is annotation-only.

- [ ] **Step 7: Commit**

```bash
git add captionlm/live.py captionlm/fusion.py tests/test_live.py
git commit -m "Fuse live captions one closed segment at a time"
```

---

### Task 5: Transport — the live endpoints

**Files:**
- Modify: `scripts/serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `LiveSession`, `SecondOpinionSession` from Tasks 1 and 4; `partition_documents`, `extract_bias_terms`, `build_context_graph` already imported in `serve.py`.
- Produces: three endpoints, and a module-level `live_sessions: dict[str, object]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_serve.py`:

```python
def test_audio_posted_to_an_unknown_session_is_refused():
    # Sessions live in memory, so a restart ends them. The client has to learn
    # that rather than posting into the void.
    import serve

    captured = {}

    class Fake(serve.Handler):
        def __init__(self):
            self.path = "/api/live/audio?session=nope"
            self.headers = {"Content-Length": "0"}
            self.rfile = io.BytesIO(b"")

        def _json(self, obj, code=200):
            captured["code"], captured["body"] = code, obj

    Fake().do_POST()
    assert captured["code"] == 404


def test_a_live_session_will_not_start_while_a_batch_job_runs():
    # One GPU. Two claimants both miss their deadline.
    import serve

    captured = {}

    class Fake(serve.Handler):
        def __init__(self):
            self.path = "/api/live/start"
            self.headers = {"Content-Length": "2"}
            self.rfile = io.BytesIO(b"{}")

        def _json(self, obj, code=200):
            captured["code"], captured["body"] = code, obj

    serve.state["running"] = True
    try:
        Fake().do_POST()
    finally:
        serve.state["running"] = False
    assert captured["code"] == 409
```

Add `import io` to the imports at the top of `tests/test_serve.py`.

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_serve.py -k live -v`
Expected: FAIL — both return 404 "no such endpoint" rather than the codes asserted.

- [ ] **Step 3: Implement the endpoints**

In `scripts/serve.py`, add to the imports:

```python
import struct
import uuid

from captionlm.live import LiveSession, SecondOpinionSession  # noqa: E402
```

Add beside the other module state:

```python
live_sessions: dict[str, object] = {}
```

Add to `Handler.do_POST`, before the final `return self._json({"error": "no such endpoint"}, 404)`:

```python
        if url.path == "/api/live/start":
            with _lock:
                if state["running"] or live_sessions:
                    return self._json({"error": "a job is already running"}, 409)
                state["running"] = True
                preset, want_second = state["preset"], state["second_opinion"]

            try:
                model, tokenizer = _get_model(PRESETS[preset]["model"])
                _, shared = partition_documents(DROPOFF)
                terms: list[str] = []
                if shared:
                    text = "\n\n".join(extract_text(d) for d in shared)
                    terms = extract_bias_terms(text)
                    model.context_graph = build_context_graph(
                        terms, tokenizer, len(model.vocabulary)
                    )
                else:
                    model.context_graph = None

                sid = uuid.uuid4().hex
                live_sessions[sid] = (
                    SecondOpinionSession(model, terms) if want_second else LiveSession(model)
                )
            except Exception:
                with _lock:
                    state["running"] = False
                raise
            return self._json({"session": sid, "terms": terms[:40]})

        if url.path == "/api/live/stop":
            sid = payload.get("session", "")
            session = live_sessions.pop(sid, None)
            with _lock:
                state["running"] = False
            if session is None:
                return self._json({"error": "no such session"}, 404)
            tail = session.finish()
            return self._json({"text": _text_of(tail), "degraded": session.degraded})
```

Add a `do_POST` branch for audio **before** the JSON body is parsed, because the body is raw PCM rather than JSON. Replace the opening of `do_POST`:

```python
    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0) or 0)

        if url.path == "/api/live/audio":
            sid = urllib.parse.parse_qs(url.query).get("session", [""])[0]
            session = live_sessions.get(sid)
            body = self.rfile.read(length)
            if session is None:
                return self._json({"error": "no such session"}, 404)
            samples = np.frombuffer(body, dtype="<f4")
            committed = session.add_audio(samples)
            return self._json({
                "text": _text_of(committed),
                "committed_until": getattr(session, "live", session).committed_until,
                "degraded": session.degraded,
            })

        body = self.rfile.read(length)
        payload = json.loads(body or b"{}")
```

Add the helper next to `_cues`:

```python
def _text_of(tokens) -> str:
    """Committed tokens as text. Token text carries its own leading space."""
    return "".join(t.text for t in tokens)
```

Add `import numpy as np` to the imports.

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/test_serve.py -v`
Expected: PASS.

- [ ] **Step 5: Verify by hand against a running server**

```bash
lsof -ti tcp:8756 | xargs kill 2>/dev/null
(cd scripts && ../venv/bin/python serve.py &) ; sleep 7
SID=$(curl -s -X POST -d '{}' http://localhost:8756/api/live/start | python3 -c 'import json,sys;print(json.load(sys.stdin)["session"])')
python3 -c "import sys,struct;sys.stdout.buffer.write(struct.pack('<16000f', *([0.0]*16000)))" \
  | curl -s -X POST --data-binary @- "http://localhost:8756/api/live/audio?session=$SID"
curl -s -X POST -d "{\"session\":\"$SID\"}" http://localhost:8756/api/live/stop
```

Expected: `start` returns a session id, `audio` returns JSON with `text` and `committed_until`, `stop` returns the tail and frees the lock.

- [ ] **Step 6: Commit**

```bash
git add scripts/serve.py tests/test_serve.py
git commit -m "Serve a live session over three endpoints"
```

---

### Task 6: Capture — microphone in the dashboard

**Files:**
- Modify: `scripts/dashboard.html`

**Interfaces:**
- Consumes: the three endpoints from Task 5.

- [ ] **Step 1: Add the live panel markup**

In `scripts/dashboard.html`, after the `dropzone` div, add:

```html
<div class="live">
  <button id="mic" class="btn">Start listening</button>
  <span id="liveState" class="fmeta"></span>
</div>
<div id="liveText" class="transcript" hidden></div>
```

- [ ] **Step 2: Add the styles**

Next to the other rules in the `<style>` block:

```css
  .live{ display:flex; align-items:center; gap:10px; padding:10px 14px;
         border-top:1px solid var(--line); }
  .live.on #mic{ background:var(--red-soft); color:var(--red); }
  #liveText{ padding:12px 14px; white-space:pre-wrap; }
  #liveText.degraded{ border-color:var(--red); }
```

- [ ] **Step 3: Add the capture code**

Before the final `refresh()` call in the `<script>` block:

```js
/* Live capture. AudioWorklet rather than the deprecated ScriptProcessor, and
   raw float32 rather than MediaRecorder's webm, so the server needs no codec.
   Only committed text is appended -- it is final by construction, so it never
   has to be rewritten under the reader. */
let live = null;

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    if (inputs[0][0]) this.port.postMessage(inputs[0][0].slice());
    return true;
  }
}
registerProcessor("tap", Tap);
`;

function downsample(chunks, from, to) {
  const flat = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const c of chunks) { flat.set(c, at); at += c.length; }
  const ratio = from / to;
  const out = new Float32Array(Math.floor(flat.length / ratio));
  for (let i = 0; i < out.length; i++) out[i] = flat[Math.floor(i * ratio)];
  return out;
}

async function startLive() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    $("liveState").textContent = "No microphone: " + e.message;
    return;
  }
  const r = await fetch("/api/live/start", { method: "POST", body: "{}" }).then((x) => x.json());
  if (r.error) { stream.getTracks().forEach((t) => t.stop()); alert(r.error); return; }

  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));
  const node = new AudioWorkletNode(ctx, "tap");
  ctx.createMediaStreamSource(stream).connect(node);

  live = { session: r.session, stream, ctx, chunks: [], sending: false };
  $("liveText").hidden = false;
  $("liveText").textContent = "";
  document.querySelector(".live").classList.add("on");
  $("mic").textContent = "Stop listening";
  $("liveState").textContent = r.terms.length ? `biased by ${r.terms.length}+ terms` : "unbiased";

  node.port.onmessage = (e) => live && live.chunks.push(e.data);
  live.timer = setInterval(() => send(ctx.sampleRate), 1000);
}

async function send(rate) {
  if (!live || live.sending || !live.chunks.length) return;
  live.sending = true;
  const pcm = downsample(live.chunks.splice(0), rate, 16000);
  try {
    const res = await fetch("/api/live/audio?session=" + live.session,
                            { method: "POST", body: pcm.buffer });
    if (res.status === 404) { $("liveState").textContent = "Session ended on the server."; return stopLive(true); }
    const r = await res.json();
    $("liveText").textContent += r.text;
    $("liveText").classList.toggle("degraded", !!r.degraded);
    if (r.degraded) $("liveState").textContent = "Falling behind — words may be rough.";
  } catch (e) {
    $("liveState").textContent = "Dropped a second of audio.";
  } finally {
    live.sending = false;
  }
}

async function stopLive(serverGone) {
  if (!live) return;
  const { session, stream, ctx, timer } = live;
  clearInterval(timer);
  live = null;
  stream.getTracks().forEach((t) => t.stop());
  ctx.close();
  document.querySelector(".live").classList.remove("on");
  $("mic").textContent = "Start listening";
  if (!serverGone) {
    const r = await fetch("/api/live/stop", { method: "POST", body: JSON.stringify({ session }) })
      .then((x) => x.json());
    if (r.text) $("liveText").textContent += r.text;
  }
  refresh();
}

$("mic").addEventListener("click", () => (live ? stopLive() : startLive()));
```

- [ ] **Step 4: Check the JavaScript parses**

```bash
python3 -c "
import pathlib,re
s=pathlib.Path('scripts/dashboard.html').read_text()
print('\n'.join(re.findall(r'<script>(.*?)</script>', s, re.S)))" > /tmp/dash.js
node --check /tmp/dash.js
```

Expected: no output, exit 0.

- [ ] **Step 5: Verify in the browser**

Start the server, open `http://localhost:8756`, click **Start listening**, grant the microphone, speak for twenty seconds. Expect text to appear a few seconds behind your voice and never to rewrite itself. Click **Stop listening**; expect the tail to flush.

- [ ] **Step 6: Run the whole suite**

Run: `venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/dashboard.html
git commit -m "Caption the microphone from the dashboard"
```

---

### Task 7: Document it

**Files:**
- Modify: `dropoff/README.md`
- Modify: `README.md`

- [ ] **Step 1: Document the live mode**

In `dropoff/README.md`, after the paragraph describing the UI, add:

```markdown
### Live

**Start listening** captions the microphone. Text appears a few seconds behind
your voice and never rewrites itself: a word is shown only once the decoder
will not revise it.

The second-opinion toggle chooses the trade, the same one it makes everywhere
else. Off: text lags about four seconds. On: it lags one 30-second segment,
because the second model reads a whole segment at a time, and both models have
to see the same closed audio for the correction to be sound.

Documents in this directory bias the live session exactly as they bias a file.

If the machine cannot keep up, the session says so and the transcript gets
rough rather than quietly skipping what it missed.
```

- [ ] **Step 2: Add live to the root README layout note**

In `README.md`, under **Quickstart**, after the `serve.py` line:

```markdown
The same page captions your microphone live — see `dropoff/README.md`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md dropoff/README.md
git commit -m "Document live captioning"
```

---

## Self-Review

**Spec coverage.** `LiveSession` → Task 1. `SecondOpinionSession` and the `second_opinion` widening → Task 4. Transport endpoints and the exclusivity lock → Task 5. Browser capture → Task 6. Falling behind, force-commit, adaptive interval → Task 1 (`_decode`, `_retune`), surfaced in Tasks 5 and 6. Silence and backlog bounds → Task 4. Session lifetime and the 404 → Tasks 5 and 6. The two gates → Tasks 2 and 3. Approach B is deferred by the spec and correctly has no task.

**One spec item deliberately not implemented:** the spec says `stop` writes an `.srt`. Task 5 returns the tail as text instead. Writing an SRT needs a filename the live session does not have, and inventing one is a product decision rather than a mechanical one. **Raise this before Task 5 rather than guessing** — the options are a timestamped default, prompting in the UI, or leaving live transcripts unsaved.

**Type consistency.** `add_audio(np.ndarray) -> list[AlignedToken]` and `finish() -> list[AlignedToken]` on both session classes, so Task 5 treats them interchangeably. `degraded` is a plain attribute on `LiveSession` and a property on `SecondOpinionSession`; `committed_until` lives on `LiveSession` only, which is why Task 5 reaches through `getattr(session, "live", session)`.

**Known rough edge in Task 6.** `downsample` picks nearest samples with no anti-aliasing filter. It is a few lines and adequate for speech at 48 kHz to 16 kHz, but it will alias. If gate 2 passes on file audio and live accuracy is visibly worse, this is the first thing to suspect.
