"""Sweep cb_weight across named term-list conditions.

Why this exists: the first real eval reported that biasing hurt (precision
-0.089, recall +0.003, WER +0.002) at NeMo's stock cb_weight of 3.0. That
number cannot distinguish three different failures -- a miscalibrated
spotter, a bad term list, or the method not working -- so it is not yet a
finding about anything.

Two design choices make it separable:

1. **Conditions are just named term-list files.** Running Earnings-21's
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
    parser.add_argument("--out-dir", default="eval_results")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    conditions = {}
    for spec in args.condition:
        name, _, path = spec.partition("=")
        if not path:
            parser.error(f"--condition needs NAME=PATH, got {spec!r}")
        conditions[name] = load_term_list(path)

    weights = [float(w) for w in args.weights.split(",")]
    score_terms = load_term_list(args.score_terms)
    os.makedirs(args.out_dir, exist_ok=True)

    clips = load_clips(args.clip_dir)
    assert clips, f"no .wav/.txt pairs found in {args.clip_dir}"
    print(f"{len(clips)} clips | scoring vocabulary: {len(score_terms)} terms")
    for name, terms in conditions.items():
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
        f"\nbaseline: WER={base_stats['wer']:.4f} P={base_stats['precision']:.4f} "
        f"R={base_stats['recall']:.4f} F={base_stats['fscore']:.4f}\n"
    )

    rows = []
    for name, terms in conditions.items():
        for clip in clips:
            clip["terms"] = terms
        for weight in weights:
            t0 = time.time()
            biased = transcribe_biased(model, clips, tokenizer, blank_idx, cb_weight=weight)
            stats = score(clips, baseline, biased, score_terms)
            b = stats["biased"]
            row = {
                "condition": name,
                "cb_weight": weight,
                "n_terms": len(terms),
                **b,
                "wer_delta": b["wer"] - base_stats["wer"],
                "fscore_delta": b["fscore"] - base_stats["fscore"],
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
                f"P={b['precision']:.4f} R={b['recall']:.4f} F={b['fscore']:.4f} "
                f"({row['fscore_delta']:+.4f})  [{time.time() - t0:.0f}s]",
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
