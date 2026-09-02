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


def _now() -> float:
    """Wall clock, as a module-level seam so a test can replace it without
    patching the global `time` module out from under pytest."""
    return time.monotonic()


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

        # Decoding on schedule is the normal path. Decoding because the buffer
        # outgrew the window is the safety one: trimming only happens after a
        # pass, so without this a session whose interval had been raised would
        # accumulate audio without bound, and a single long delivery would be
        # trimmed before anything had read it.
        overfull = len(self._buffer) > self.window * self._rate
        if not overfull and self.now - self._last_decode < self.interval:
            return []
        return self._decode()

    def finish(self) -> list[AlignedToken]:
        """Commit everything left, including inside the horizon."""
        return self._decode(cutoff=float("inf"))

    def _decode(self, cutoff: float | None = None) -> list[AlignedToken]:
        if len(self._buffer) < self._rate // 10:      # under 100ms, nothing to decode
            return []

        started = _now()
        # get_logmel's FFT-magnitude step reinterprets complex64 STFT output
        # by byte width, which only lines up correctly for float32 (2x) --
        # bf16 (4x) silently corrupts the frame count. parakeet_mlx's own
        # load_audio always hands it float32 for exactly this reason,
        # regardless of what dtype its own (unused) parameter requests.
        mel = get_logmel(mx.array(self._buffer, dtype=mx.float32), self.model.preprocessor_config)
        result = self.model.generate(mel, decoding_config=self._decoding_config)[0]
        elapsed = _now() - started

        for token in result.tokens:
            token.start += self._buffer_start
            token.end = token.start + token.duration

        if cutoff is None:
            cutoff = self.now - self.horizon

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

        # Nothing should reach here uncommitted -- add_audio forces a pass
        # before the buffer can outgrow the window. If it ever does, the audio
        # is genuinely gone, and saying so beats resuming as if it were not.
        if self._buffer_start > self.committed_until + 1e-6:
            self.degraded = True
            self.committed_until = self._buffer_start

    def _retune(self, elapsed: float) -> None:
        """Decode less often when a pass costs more than the interval.

        Decoding less often is what reduces duty cycle; a longer horizon does
        nothing for throughput. The cap keeps at least two commits per horizon
        so the transcript does not arrive in one lump. `degraded` is exactly
        this condition -- the session is not keeping up -- and clears once it
        does.
        """
        ceiling = self.horizon / 2
        if elapsed > self.interval:
            self.interval = min(max(self._configured_interval, elapsed), ceiling)
            self.degraded = True
            self._good_passes = 0
        else:
            self._good_passes += 1
            if self._good_passes >= 3:
                self.interval = self._configured_interval
                self.degraded = False
                self._good_passes = 0
