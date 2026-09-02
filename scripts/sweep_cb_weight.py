"""Sweep cb_weight across named term-list conditions.

Why this exists: the first real eval reported that biasing hurt (precision
-0.089, recall +0.003, WER +0.002) at NeMo's stock cb_weight of 3.0. That
number cannot distinguish three different failures -- a miscalibrated
spotter, a bad term list, or the method not working -- so it is not yet a
finding about anything.

Two design choices make it separable:

1. **Conditions are just named term-list files.** The literal path
   `per-clip` is the exception: it means "bias each clip with its own
   <clip>.terms.txt", which is the configuration cli.py actually ships,
   and the only one a product number may be quoted from. Every other
   condition applies one list to every clip, which is what makes the
   oracle/distractor contrast measurable but is not what a user gets. Running Earnings-21's
   own oracle list (words known to BE in the transcripts) against its
   distractor list (the same words plus ~769 Fortune 500 names and CEOs
   known NOT to be) isolates false-accept behaviour from term-extraction
   quality. A product term list from pyate or spaCy NER can be added as
   another condition and compared on the same axis.

2. **One scoring vocabulary for every condition.** F-score is always
   computed against --score-terms (default: the distractor list, the
   superset). This matters: compute_fscore only counts keywords that are
   in the list it is given, so scoring the distractor run against the
   oracle list would silently ignore every distractor false-accept --
   hiding exactly the harm the run exists to measure.

The unbiased baseline does not depend on cb_weight, so it is transcribed
once and reused across every cell. A 4-condition x 5-weight sweep costs
21 passes rather than 40.
"""
import argparse
import json
import os
import time

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.eval import load_clips, score, transcribe_baseline, transcribe_biased
from captionlm.terms import load_term_list, load_tokenizer

# --condition NAME=per-clip biases each clip with its own <clip>.terms.txt.
PER_CLIP = "per-clip"


def _split(stats: dict) -> str:
    """B-WER/U-WER: errors on term-list words against errors on the rest.
    The pair is the point of the sweep -- overall WER cannot separate a
    weight that recovers terms from one that rewrites correct ordinary
    words, because listed words are a minority of the words and the two
    effects cancel in the aggregate."""
    b = "n/a" if stats["b_wer"] is None else f"{stats['b_wer']:.4f}"
    u = "n/a" if stats["u_wer"] is None else f"{stats['u_wer']:.4f}"
    return f"B={b} U={u}"


def _rate(stats: dict) -> str:
    """Injection rate: listed-but-unspoken terms that got transcribed anyway.
    Rises when biasing starts hallucinating the term list; "n/a" means the
    clip set has no unspoken terms, so this condition cannot measure it."""
    if stats["injection_rate"] is None:
        return "n/a"
    return f"{stats['injection_rate']:.4f} ({len(stats['injected'])}/{stats['unspoken_terms']})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_dir", help="Directory of <clip>.wav/<clip>.txt pairs")
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named term list to bias with; repeatable (e.g. oracle=.../oracle_list.txt)",
    )
    parser.add_argument(
        "--score-terms",
        required=True,
        help="Term list used for F-score in EVERY condition, so results are comparable",
    )
    parser.add_argument("--weights", default="0.5,1.0,1.5,2.0,3.0")
    parser.add_argument("--out-dir", default="data/eval_results")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    conditions = {}
    for spec in args.condition:
        name, _, path = spec.partition("=")
        if not path:
            parser.error(f"--condition needs NAME=PATH, got {spec!r}")
        conditions[name] = None if path == PER_CLIP else load_term_list(path)

    weights = [float(w) for w in args.weights.split(",")]
    score_terms = load_term_list(args.score_terms)
    os.makedirs(args.out_dir, exist_ok=True)

    clips = load_clips(args.clip_dir)
    assert clips, f"no .wav/.txt pairs found in {args.clip_dir}"
    own_terms = [clip["terms"] for clip in clips]
    print(f"{len(clips)} clips | scoring vocabulary: {len(score_terms)} terms")
    for name, terms in conditions.items():
        if terms is None:
            print(f"  condition {name!r}: per-clip, {[len(t) for t in own_terms]} terms")
        else:
            print(f"  condition {name!r}: {len(terms)} terms")

    model = load_biased_model(args.model)
    tokenizer = load_tokenizer(args.model)
    blank_idx = len(model.vocabulary)

    print("\ntranscribing unbiased baseline once...", flush=True)
    t0 = time.time()
    baseline = transcribe_baseline(model, clips)
    print(f"  done in {time.time() - t0:.0f}s")

    base_stats = score(clips, baseline, baseline, score_terms)["baseline"]
    print(
        f"\nbaseline: WER={base_stats['wer']:.4f} {_split(base_stats)} "
        f"P={base_stats['precision']:.4f} "
        f"R={base_stats['recall']:.4f} F={base_stats['fscore']:.4f} "
        f"inject={_rate(base_stats)}  "
        f"[{base_stats['b_words']} listed / {base_stats['u_words']} other words]\n"
    )

    rows = []
    for name, terms in conditions.items():
        for clip, own in zip(clips, own_terms):
            clip["terms"] = own if terms is None else terms
        n_terms = sum(len(c["terms"]) for c in clips) // len(clips) if terms is None else len(terms)
        for weight in weights:
            t0 = time.time()
            biased = transcribe_biased(model, clips, tokenizer, blank_idx, cb_weight=weight)
            stats = score(clips, baseline, biased, score_terms)
            b = stats["biased"]
            row = {
                "condition": name,
                "cb_weight": weight,
                "n_terms": n_terms,
                **b,
                "wer_delta": b["wer"] - base_stats["wer"],
                "fscore_delta": b["fscore"] - base_stats["fscore"],
                **stats["yield"],
            }
            rows.append(row)
            out = os.path.join(args.out_dir, f"sweep-{name}-{weight}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(
                    {"baseline": base_stats, "biased": b, "per_clip": stats["per_clip"], **row},
                    f,
                    indent=2,
                )
            print(
                f"{name:<12} cb={weight:<4} WER={b['wer']:.4f} ({row['wer_delta']:+.4f})  "
                f"{_split(b)}  "
                f"P={b['precision']:.4f} R={b['recall']:.4f} F={b['fscore']:.4f} "
                f"({row['fscore_delta']:+.4f})  inject={_rate(b)}  "
                f"net={stats['yield']['net_term_gain']:+d}  "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )

    summary = os.path.join(args.out_dir, "sweep-summary.json")
    with open(summary, "w", encoding="utf-8") as f:
        json.dump({"baseline": base_stats, "rows": rows}, f, indent=2)

    best = min(rows, key=lambda r: r["wer"])
    print(f"\nlowest WER: {best['condition']} @ cb_weight={best['cb_weight']} -> {best['wer']:.4f}")
    if best["wer"] >= base_stats["wer"]:
        print("NOTE: no swept point beat the unbiased baseline's WER.")
    print(f"summary written to {summary}")


if __name__ == "__main__":
    main()
