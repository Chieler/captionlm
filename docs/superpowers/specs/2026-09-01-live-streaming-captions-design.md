# Live streaming captions — design

Captions from a microphone, in the browser, with the domain biasing intact.

The existing second-opinion toggle selects the mode, because it already means
"faster or better" and now also means "live or lagging":

| toggle | behaviour | latency |
|---|---|---|
| off | live captions while speaking | ~4s |
| on | early results on a long recording | ~one 30s segment |

## The constraint that shapes everything

`parakeet_mlx`'s own streaming API cannot be used as-is.
`StreamingParakeet.add_audio` calls `self.model.decode()` directly, bypassing
`generate()` — and `generate()` is the only place `BiasedParakeetTDTCTC` runs
the CTC word spotter. Streaming through that API silently drops every bit of
term biasing, which is the entire product.

This is fixable rather than fatal (see **Approach B**, deferred), but the
first version does not attempt it.

## Approach: rolling re-transcribe

Keep a buffer of the last 30 seconds. Every 2 seconds, run the **existing**
biased `generate()` over that buffer. Commit words older than a stability
horizon; re-derive the rest.

The reason this is first: **biasing quality is exactly what has already been
measured.** No new spotter behaviour, no boundary splitting, no unmeasured
per-window beam cost, and full attention over the window rather than the
local attention `StreamingParakeet` switches on.

What it costs: repeated compute on the same audio, and "finalized" becomes a
time heuristic rather than the decoder's own notion. Both are acceptable at
this scale.

## Components

Four units, each testable alone.

### `LiveSession` (`captionlm/live.py`)

The decoder. No transport, no audio device.

```python
class LiveSession:
    def __init__(self, model, tokenizer, terms, *,
                 window=30.0, horizon=4.0, interval=2.0)
    def add_audio(self, frames: np.ndarray) -> list[AlignedToken]  # newly committed
    def finish(self) -> list[AlignedToken]                         # commit the tail
```

State: a float32 ring of the last `window` seconds, `committed_until` in
absolute session seconds, and the time of the last decode.

Each `add_audio` appends and, if `interval` has elapsed, runs one pass:
`get_logmel` over the buffer, `model.generate()`, offset the buffer-relative
token times to absolute. A token commits when

    token.start >= committed_until  and  token.end <= now - horizon

`committed_until` then advances to the last committed token's end. Everything
after the horizon is discarded and re-derived on the next pass, so a word is
emitted once and never revised.

The `start >= committed_until` test drops a token straddling the boundary
rather than emitting a second copy of a word already shown.

Trimming keeps the last `window` seconds. At `window` 30 and `horizon` 4
there are 26 seconds of slack between the commit point and the back of the
buffer, so uncommitted audio can only fall off if decoding stalls that long.
See **Falling behind**.

### `SecondOpinionSession` (`captionlm/live.py`)

Wraps `LiveSession`. Accumulates closed 30-second segments; when one closes,
runs Whisper on that segment's samples, calls `fuse_tokens` against the
parakeet tokens for the same span, and emits the fused result.

Both models see **identical closed audio**, which is what makes this sound.
Fusing a partial against a whole-file second transcript would align a prefix
to a complete transcript and corrupt its own tail — that is why the batch
path holds its transcript back to the end instead of streaming it.

One supporting change: `fusion.second_opinion` is annotated
`audio_path: str`, but `mlx_whisper.transcribe` accepts
`str | np.ndarray | mx.array`. Widening that annotation lets a segment pass
through in memory with no temporary wav.

### Transport (`scripts/serve.py`)

| endpoint | body | returns |
|---|---|---|
| `POST /api/live/start` | `{}` | `{"session": id}`, trie built from the drop-off documents |
| `POST /api/live/audio?session=<id>` | raw little-endian float32 PCM | `{"text": [...], "committed_until": s, "degraded": bool}` |
| `POST /api/live/stop` | `{"session": id}` | `finish()`, write `.srt`, drop the session |

No WebSocket. The 4-second horizon already exceeds any latency a socket would
save, so RFC 6455 framing on stdlib `http.server` — or a dependency to avoid
writing it — buys nothing this design can use.

**A live session takes the same exclusivity lock as a batch job.** `serve.py`
runs one job at a time; two sessions, or a live session alongside a file job,
would contend on one GPU and both would miss their deadline.

### Capture (`scripts/dashboard.html`)

`getUserMedia` → `AudioWorklet` → downsample to 16 kHz mono float32 → POST
roughly one second of samples at a time. This works on `http://localhost`
with no certificate, which is a secure context. Committed text appends to the
transcript panel that already exists.

## Data flow

    mic -> AudioWorklet -> 16kHz float32 -> POST -> LiveSession.add_audio
        -> committed tokens in the response -> transcript panel

## Error handling

**Falling behind** is the case that matters.

Detection: time each pass. Decode time exceeding `interval` means the session
is slipping; `now - committed_until` approaching `window` means uncommitted
audio is about to fall off the back.

Response, in order:

1. **Raise `interval`.** Decoding less often is what reduces duty cycle;
   raising the horizon does nothing for throughput. Concretely: set
   `interval` to the measured decode time, capped at `horizon / 2` so a
   commit still happens at least twice per horizon. Decay back toward the
   configured value after three passes that finish inside budget.
2. If audio would still fall off uncommitted, **force-commit it rather than
   drop it**, and mark the session degraded.

Losing words silently is ruled out. A live caption that quietly skips ten
seconds is worse than one that admits it fell behind. The degraded state
reaches the UI and the `.srt` written at `stop`.

**Second-opinion backlog.** At the measured 12x realtime a 30-second segment
costs about 2.5 seconds, so it should keep up — but bound the queue anyway:
more than two outstanding segments and that segment is emitted
parakeet-only, flagged.

**Silence.** Whisper hallucinates on silence. Near-silent segments skip the
second opinion rather than inventing a sentence.

**Session lifetime.** Sessions live in memory, so a server restart ends them.
The client gets a 404 on its session id, stops capture, and says so. Mic
permission denial and missing devices surface as UI state. A failed POST
retries once; if it still fails, the samples are gone and the session records
a gap rather than pretending continuity.

## Testing

**Unit, model stubbed** — the pattern in `tests/test_progressive.py`:

- commits are monotonic in time
- no word is emitted twice
- no committed word is revised
- a token straddling `committed_until` is dropped, not duplicated
- `finish()` flushes the tail
- a stalled decode force-commits rather than losing audio

**Acceptance test.** Feed a known file through `LiveSession` in simulated
real-time chunks; compare the committed transcript against the **batch**
transcript of the same file. Approach A's whole claim is that rolling
re-transcribe preserves measured quality, so on read-aloud the live arm
should land near the batch WER 0.0969 and term recall 0.9006. Two clips.

**Transport** — raw float32 round trip, session lifecycle, 404 on an unknown
session, exclusivity against a batch job.

## Order of work

Two measurements gate everything, and both are done with `LiveSession`
alone, before any transport or browser code exists:

1. **Duty cycle.** The projection is that 30 seconds of audio at 59x realtime
   costs about 0.5s of compute every 2s, roughly 25%. This is derived from
   batch throughput and assumes per-call overhead on a 30-second `generate()`
   is negligible. Nobody has checked that. If it is wrong, the horizon grows
   or the window shrinks — the design survives either, but the numbers above
   change.
2. **Acceptance WER.** If the live transcript does not track the batch
   transcript, approach A is wrong and we know it before building on it.

Transport and capture are built only if both pass.

## Approach B — biased streaming decoder (deferred)

The architecturally correct version, and the one to build if A's duty cycle
turns out too high or the acceptance WER disappoints.

Subclass `StreamingParakeet` and run the spotter inside `add_audio`. Both
inputs `run_word_spotter` needs are already present there: the model's
`ctc_decoder` (`ParakeetTDTCTC.ctc_decoder`, since `ParakeetTDTCTC` subclasses
`ParakeetTDT` and therefore takes the working TDT decode branch) and the
`features` computed in that method. Spotted terms merge into the finalized
tokens exactly as `merge.py` does for the batch path.

It is O(n) rather than O(n²) in compute, and "finalized" becomes the
decoder's own notion instead of a time heuristic.

**Two unknowns must be spiked before committing to it:**

1. **Boundary loss.** A term straddling the finalized/draft boundary gets
   only half its frames. Mitigation to test: run the spotter over a window
   that includes already-finalized frames and accept only spots landing in
   the new region — the shape `merge.py` already uses. Unmeasured.
2. **Spotter cost.** The beam search currently runs once per 120-second
   chunk. Streaming moves it to every window or two. Its cost at that rate is
   unmeasured.

A third, smaller risk: `StreamingParakeet` switches the encoder to
`rel_pos_local_attn`. Local attention has never been scored on this
material, so accuracy under it is unknown independently of the spotter.

## Out of scope

Speaker diarization. Multiple concurrent sessions. Remote or non-loopback
access. Any capture path other than the browser microphone.
