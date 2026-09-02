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

import mlx.core as mx
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
    audio = np.array(load_audio(wav, rate, mx.float32), copy=False)
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
