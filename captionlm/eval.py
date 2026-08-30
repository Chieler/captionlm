"""Score domain-biased captioning against a labeled clip set: term
F-score and WER, with biasing on vs off. Run captionlm/eval_dataset.py
first to populate clip_dir automatically from Earnings-21, or hand-collect
<clip>.wav/<clip>.txt pairs yourself."""
import argparse
import glob
import os

import jiwer

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer
from captionlm.vendor.fscore import compute_fscore


def load_clips(clip_dir: str) -> list[dict]:
    clips = []
    for wav_path in sorted(glob.glob(os.path.join(clip_dir, "*.wav"))):
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            reference = f.read().strip()
        clips.append({"wav": wav_path, "text": reference})
    return clips


def transcribe_clips(model, clips: list[dict]) -> list[dict]:
    samples = []
    for clip in clips:
        result = model.transcribe(clip["wav"])
        samples.append({"text": clip["text"], "pred_text": result.text})
    return samples


def run_eval(
    clip_dir: str,
    term_list_path: str,
    model_id: str = MODEL_ID,
    cb_weight: float | None = None,
) -> dict:
    terms = load_term_list(term_list_path)
    clips = load_clips(clip_dir)
    assert clips, f"no .wav/.txt pairs found in {clip_dir}"

    model = load_biased_model(model_id)
    baseline = transcribe_clips(model, clips)

    tokenizer = load_tokenizer(model_id)
    blank_idx = len(model.vocabulary)
    model.context_graph = build_context_graph(terms, tokenizer, blank_idx)
    if cb_weight is not None:
        model.spotter_config.cb_weight = cb_weight
    biased = transcribe_clips(model, clips)

    references = [c["text"] for c in clips]
    baseline_wer = jiwer.wer(references, [s["pred_text"] for s in baseline])
    biased_wer = jiwer.wer(references, [s["pred_text"] for s in biased])

    baseline_p, baseline_r, baseline_f = compute_fscore(baseline, terms)
    biased_p, biased_r, biased_f = compute_fscore(biased, terms)

    return {
        "baseline": {"wer": baseline_wer, "precision": baseline_p, "recall": baseline_r, "fscore": baseline_f},
        "biased": {"wer": biased_wer, "precision": biased_p, "recall": biased_r, "fscore": biased_f},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate domain-biased captioning")
    parser.add_argument("clip_dir", help="Directory of <clip>.wav/<clip>.txt pairs")
    parser.add_argument("terms", help="Path to the domain term list (.txt)")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cb-weight", type=float, default=None)
    args = parser.parse_args()

    stats = run_eval(args.clip_dir, args.terms, args.model, args.cb_weight)
    for label, s in stats.items():
        print(f"{label}: WER={s['wer']:.3f} P={s['precision']:.3f} R={s['recall']:.3f} F={s['fscore']:.3f}")


if __name__ == "__main__":
    main()
