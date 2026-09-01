"""Caption every audio file in a drop-off directory, biased by the
document sitting next to it.

This is the product path end to end, in one command: a user has a
recording and the document it is about, and wants an .srt in which the
domain terms are spelled correctly. Pair them by filename --
`standup.m4a` next to `standup.pdf` -- and this extracts the terms,
compiles them into the context graph, and transcribes.

Nothing is trained. The term list becomes a trie at call time, which is
the whole reason CTC-WS was chosen over fine-tuning: the user waits for
transcription only, never for training.

An audio file with no matching document is still transcribed, unbiased,
with a warning -- that is the honest baseline, not a failure.
"""
import argparse
import glob
import os

from parakeet_mlx.alignment import sentences_to_result, tokens_to_sentences

from captionlm.biasable import extract_bias_terms
from captionlm.build_eval_set import convert_to_wav
from captionlm.cli import result_to_srt
from captionlm.config import MODEL_ID
from captionlm.doc_import import extract_text
from captionlm.biased_model import load_biased_model
from captionlm.fusion import WHISPER_MODEL_ID, fuse_tokens, second_opinion
from captionlm.terms import build_context_graph, load_tokenizer

AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".mov", ".aac", ".flac", ".ogg", ".webm", ".mkv"}
DOC_EXTS = {".txt", ".md", ".pdf", ".docx", ".doc", ".csv", ".rtf", ".html"}


def find_pairs(directory: str) -> list[tuple[str, str | None]]:
    """(audio, document-or-None) for every audio file, matched by basename."""
    by_base: dict[str, dict[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        base, ext = os.path.splitext(os.path.basename(path))
        ext = ext.lower()
        if ext in AUDIO_EXTS:
            by_base.setdefault(base, {})["audio"] = path
        elif ext in DOC_EXTS:
            by_base.setdefault(base, {})["doc"] = path
    return [(v["audio"], v.get("doc")) for v in by_base.values() if "audio" in v]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="dropoff")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--mode", choices=("terminology", "ner", "combined"), default="combined")
    parser.add_argument("--top-n", type=int, default=None, help="Cap the term list (default: no cap)")
    parser.add_argument("--cb-weight", type=float, default=None,
                        help="Biasing strength. Default is SpotterConfig's 3.0, "
                             "which measured best on the read-aloud eval set.")
    parser.add_argument("--second-opinion", nargs="?", const=WHISPER_MODEL_ID, default=None,
                        metavar="MODEL_ID",
                        help="Transcribe again with an independent model and let it "
                             "correct the spans it disagrees on. Measured at 1.1b: "
                             "read-aloud WER 0.0712 -> 0.0560, Earnings-21 0.1929 -> "
                             "0.1858, with term recall up on both. Costs a 1.6 GB "
                             "download and about a quarter again of the runtime.")
    args = parser.parse_args()

    pairs = find_pairs(args.directory)
    if not pairs:
        raise SystemExit(
            f"No audio found in {args.directory}/. Drop in a recording and, next to it, "
            f"the document it is about, with the SAME basename:\n"
            f"    {args.directory}/standup.m4a\n"
            f"    {args.directory}/standup.pdf"
        )

    print(f"{len(pairs)} recording(s) in {args.directory}/\nloading {args.model} ...", flush=True)
    model = load_biased_model(args.model)
    tokenizer = load_tokenizer(args.model)
    blank_idx = len(model.vocabulary)
    if args.cb_weight is not None:
        model.spotter_config.cb_weight = args.cb_weight

    for audio, doc in pairs:
        base = os.path.splitext(audio)[0]
        print(f"\n=== {os.path.basename(audio)} ===")

        terms: list[str] = []
        if doc:
            terms = extract_bias_terms(extract_text(doc), mode=args.mode, top_n=args.top_n)
            with open(f"{base}.terms.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(terms) + "\n")
            print(f"  {len(terms)} terms from {os.path.basename(doc)} -> {os.path.basename(base)}.terms.txt")
            print(f"    {', '.join(terms[:10])}{' ...' if len(terms) > 10 else ''}")
            model.context_graph = build_context_graph(terms, tokenizer, blank_idx)
        else:
            print("  no matching document -- transcribing UNBIASED")
            model.context_graph = None

        wav = audio
        if not audio.lower().endswith(".wav"):
            wav = f"{base}.converted.wav"
            convert_to_wav(audio, wav)

        result = model.transcribe(wav, chunk_duration=120.0)

        if args.second_opinion:
            print("  asking for a second opinion ...", flush=True)
            result = sentences_to_result(
                tokens_to_sentences(
                    fuse_tokens(result.tokens, second_opinion(wav, args.second_opinion), terms)
                )
            )

        out = f"{base}.srt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(result_to_srt(result))
        print(f"  wrote {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
