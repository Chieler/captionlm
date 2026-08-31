"""Score domain-biased captioning against a labeled clip set: term
F-score and WER, with biasing on vs off. Run captionlm/build_eval_set.py
first to populate clip_dir automatically from Earnings-21, or hand-collect
<clip>.wav/<clip>.txt pairs yourself.

Each clip is biased with its own <clip>.terms.txt when present, matching
what cli.py does: one document biases its own audio. The combined term
list passed on the command line is used for F-score scoring, so the
metric's vocabulary stays constant across clips.
"""
import argparse
import glob
import json
import os

import jiwer

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer
from captionlm.vendor.fscore import compute_fscore


def load_clips(clip_dir: str, fallback_terms: list[str] | None = None) -> list[dict]:
    clips = []
    for wav_path in sorted(glob.glob(os.path.join(clip_dir, "*.wav"))):
        base = os.path.splitext(wav_path)[0]
        txt_path = base + ".txt"
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            reference = f.read().strip()
        terms_path = base + ".terms.txt"
        terms = load_term_list(terms_path) if os.path.isfile(terms_path) else list(fallback_terms or [])
        clips.append({"wav": wav_path, "text": reference, "terms": terms})
    return clips


def transcribe_clips(model, clips: list[dict]) -> list[dict]:
    samples = []
    for clip in clips:
        result = model.transcribe(clip["wav"], chunk_duration=120.0)
        samples.append({"text": clip["text"], "pred_text": result.text})
    return samples


def transcribe_baseline(model, clips: list[dict]) -> list[dict]:
    model.context_graph = None
    return transcribe_clips(model, clips)


def transcribe_biased(
    model,
    clips: list[dict],
    tokenizer,
    blank_idx: int,
    cb_weight: float | None = None,
) -> list[dict]:
    if cb_weight is not None:
        model.spotter_config.cb_weight = cb_weight
    samples = []
    for clip in clips:
        model.context_graph = build_context_graph(clip["terms"], tokenizer, blank_idx)
        samples.extend(transcribe_clips(model, [clip]))
    return samples


# jiwer 4.0.0's default transform does NOT normalize -- wer_default is
# [RemoveMultipleSpaces, Strip, ReduceToListOfListOfWords], so
# jiwer.wer("Net Sales grew.", "net sales grew") is 1.0. The model emits
# cased, punctuated text and the references are rebuilt from the .nlp
# token+punctuation columns, so without this the score conflates
# recognition errors with casing and punctuation mismatch.
#
# jiwer's own wer_standardize is not the fix: it includes RemoveWhiteSpace,
# which glues every word into a single token. Compose it explicitly.
_NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def _wer(references: list[str], hypotheses: list[str]) -> float:
    # NB: jiwer.wer takes reference_transform, NOT truth_transform.
    return jiwer.wer(
        references,
        hypotheses,
        reference_transform=_NORMALIZE,
        hypothesis_transform=_NORMALIZE,
    )


def score(clips: list[dict], baseline: list[dict], biased: list[dict], all_terms: list[str]) -> dict:
    references = [c["text"] for c in clips]
    baseline_preds = [s["pred_text"] for s in baseline]
    biased_preds = [s["pred_text"] for s in biased]

    baseline_p, baseline_r, baseline_f = compute_fscore(baseline, all_terms)
    biased_p, biased_r, biased_f = compute_fscore(biased, all_terms)

    per_clip = [
        {
            "clip": os.path.basename(clip["wav"]),
            "reference": clip["text"],
            "baseline_pred": base_pred,
            "biased_pred": bias_pred,
            "baseline_wer": _wer([clip["text"]], [base_pred]),
            "biased_wer": _wer([clip["text"]], [bias_pred]),
            "n_terms": len(clip["terms"]),
        }
        for clip, base_pred, bias_pred in zip(clips, baseline_preds, biased_preds)
    ]

    return {
        "baseline": {
            "wer": _wer(references, baseline_preds),
            "precision": baseline_p,
            "recall": baseline_r,
            "fscore": baseline_f,
        },
        "biased": {
            "wer": _wer(references, biased_preds),
            "precision": biased_p,
            "recall": biased_r,
            "fscore": biased_f,
        },
        "per_clip": per_clip,
    }


def run_eval(
    clip_dir: str,
    term_list_path: str,
    model_id: str = MODEL_ID,
    cb_weight: float | None = None,
    baseline: list[dict] | None = None,
) -> dict:
    all_terms = load_term_list(term_list_path)
    clips = load_clips(clip_dir, fallback_terms=all_terms)
    assert clips, f"no .wav/.txt pairs found in {clip_dir}"

    model = load_biased_model(model_id)
    if baseline is None:
        baseline = transcribe_baseline(model, clips)

    tokenizer = load_tokenizer(model_id)
    biased = transcribe_biased(model, clips, tokenizer, len(model.vocabulary), cb_weight)

    stats = score(clips, baseline, biased, all_terms)
    stats["cb_weight"] = cb_weight
    stats["baseline_samples"] = baseline
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate domain-biased captioning")
    parser.add_argument("clip_dir", help="Directory of <clip>.wav/<clip>.txt pairs")
    parser.add_argument("terms", help="Path to the combined domain term list (.txt), used for F-score scoring")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cb-weight", type=float, default=None)
    parser.add_argument("--out-json", help="Write full per-clip results to this path as JSON")
    args = parser.parse_args()

    stats = run_eval(args.clip_dir, args.terms, args.model, args.cb_weight)
    for label in ("baseline", "biased"):
        s = stats[label]
        print(f"{label}: WER={s['wer']:.3f} P={s['precision']:.3f} R={s['recall']:.3f} F={s['fscore']:.3f}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in stats.items() if k != "baseline_samples"}, f, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
