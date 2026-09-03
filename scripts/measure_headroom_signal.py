"""Does the headroom signal in captionlm/headroom.py actually predict
whether biasing a term will help or hurt? Retrospective: reads a sweep
already run by scripts/sweep_cb_weight.py, captures the unbiased CTC
logits for the same clips (one extra transcription pass per clip -- the
logits were never persisted, so this is unavoidable), computes each
listed term's headroom score, and checks whether that score actually
separates terms the sweep recovered from terms it falsely injected.

See docs/superpowers/specs/2026-09-03-headroom-signal-validation-design.md.
Nothing here is wired into cli.py -- this is a measurement, not a feature.
"""
import argparse
import json
import os

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.eval import load_clips, per_term_stats
from captionlm.headroom import unbiased_headroom
from captionlm.terms import build_context_graph, load_tokenizer


def _load_sweep_predictions(sweep_dir: str, condition: str, weight: float) -> dict[str, dict]:
    """clip basename -> {"reference", "baseline_pred", "biased_pred"}, as
    already recorded by scripts/sweep_cb_weight.py's per-weight JSON."""
    path = os.path.join(sweep_dir, f"sweep-{condition}-{weight}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {row["clip"]: row for row in data["per_clip"]}


def _clip_headroom(model, tokenizer, clip: dict, blank_idx: int) -> dict[str, float]:
    """Max headroom score per term across every chunk of this one clip."""
    model.context_graph = build_context_graph(clip["terms"], tokenizer, blank_idx)
    model.capture_logits = []
    model.transcribe(clip["wav"], chunk_duration=120.0)

    combined: dict[str, float] = {}
    for chunk_logprobs in model.capture_logits:
        chunk_scores = unbiased_headroom(
            model.context_graph, chunk_logprobs, model, blank_idx,
            ctc_ali_token_weight=model.spotter_config.ctc_ali_token_weight,
        )
        for term, score in chunk_scores.items():
            # WSHyp.word carries the term's original casing from the
            # .terms.txt file (build_context_graph stores it that way even
            # though it tokenizes both the original and lowercase spellings
            # to match either spoken form). per_term_stats's row["term"]
            # is always lowercased (captionlm.vendor.fscore.keyword_stats
            # does `kw.lower()`), so key on the same lowercase form here or
            # every term with an uppercase letter fails this lookup by
            # construction, regardless of what raw_spot actually found.
            term = term.lower()
            if term not in combined or score > combined[term]:
                combined[term] = score
    return combined


def _outcome(row: dict) -> str:
    if row["biased_correct"] > row["baseline_correct"]:
        return "recovered"
    if row["biased_false_accepts"] > row["baseline_false_accepts"]:
        return "injected"
    return "neither"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_dir", help="Directory of <clip>.wav/<clip>.txt/<clip>.terms.txt")
    parser.add_argument("--sweep-dir", required=True, help="Directory holding sweep-<condition>-<weight>.json")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    clips = [c for c in load_clips(args.clip_dir) if c["terms"]]
    if not clips:
        print(f"no clips with a .terms.txt found in {args.clip_dir}")
        return

    predictions = _load_sweep_predictions(args.sweep_dir, args.condition, args.weight)

    model = load_biased_model(args.model)
    tokenizer = load_tokenizer(args.model)
    blank_idx = len(model.vocabulary)

    rows: list[dict] = []
    for clip in clips:
        basename = os.path.basename(clip["wav"])
        if basename not in predictions:
            print(f"skip {basename}: no matching entry in {args.sweep_dir}'s sweep JSON")
            continue

        pred = predictions[basename]
        headroom = _clip_headroom(model, tokenizer, clip, blank_idx)

        baseline_sample = [{"text": pred["reference"], "pred_text": pred["baseline_pred"]}]
        biased_sample = [{"text": pred["reference"], "pred_text": pred["biased_pred"]}]
        term_rows = per_term_stats(clip["terms"], baseline_sample, biased_sample)

        for row in term_rows:
            rows.append(
                {
                    "clip": basename,
                    "term": row["term"],
                    "outcome": _outcome(row),
                    "headroom": headroom.get(row["term"]),
                }
            )

    by_outcome: dict[str, list[float]] = {"recovered": [], "injected": [], "neither": []}
    no_candidate = {"recovered": 0, "injected": 0, "neither": 0}
    for row in rows:
        if row["headroom"] is None:
            no_candidate[row["outcome"]] += 1
        else:
            by_outcome[row["outcome"]].append(row["headroom"])

    print(f"\n{'outcome':<12} {'n (scored)':<12} {'n (no candidate)':<18} {'mean headroom':<15} {'min':<10} {'max':<10}")
    for outcome, scores in by_outcome.items():
        if scores:
            mean_s, min_s, max_s = sum(scores) / len(scores), min(scores), max(scores)
        else:
            mean_s = min_s = max_s = float("nan")
        print(f"{outcome:<12} {len(scores):<12} {no_candidate[outcome]:<18} {mean_s:<15.3f} {min_s:<10.3f} {max_s:<10.3f}")

    out_path = os.path.join(args.sweep_dir, f"headroom-{args.condition}-{args.weight}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nper-term rows written to {out_path}")


if __name__ == "__main__":
    main()
