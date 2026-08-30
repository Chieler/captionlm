"""CLI: caption a video/audio file, optionally biased toward a domain
term list, and write an .srt file."""
import argparse
from pathlib import Path

from parakeet_mlx.alignment import AlignedResult

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def result_to_srt(result: AlignedResult) -> str:
    lines = []
    for i, sentence in enumerate(result.sentences, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(sentence.start)} --> {format_timestamp(sentence.end)}")
        lines.append(sentence.text.strip())
        lines.append("")
    return "\n".join(lines) + "\n"


def caption_file(audio_path: str, term_list_path: str | None, model_id: str = MODEL_ID) -> str:
    model = load_biased_model(model_id)

    if term_list_path:
        tokenizer = load_tokenizer(model_id)
        terms = load_term_list(term_list_path)
        blank_idx = len(model.vocabulary)
        model.context_graph = build_context_graph(terms, tokenizer, blank_idx)

    result = model.transcribe(audio_path, chunk_duration=120.0)
    return result_to_srt(result)


def main():
    parser = argparse.ArgumentParser(description="Domain-biased captioning")
    parser.add_argument("audio", help="Path to an audio/video file")
    parser.add_argument("--terms", help="Path to a term list (.txt, one term per line)")
    parser.add_argument("--out", help="Output .srt path (default: <audio>.srt)")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    srt = caption_file(args.audio, args.terms, args.model)
    out_path = args.out or str(Path(args.audio).with_suffix(".srt"))
    Path(out_path).write_text(srt, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
