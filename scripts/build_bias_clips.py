"""Build eval clips straight from a cloned Earnings-21, with no SEC EDGAR
lookup and no term extraction.

captionlm/build_eval_set.py exists to assemble the *product* scenario: a
user's own document biasing their own audio. That path depends on filing
lookup and pyate, so a bad result there is ambiguous -- it could be the
spotter or it could be the term extractor.

This script strips both away. It emits only <id>.wav and <id>.txt, so the
sweep can point the context graph at Earnings-21's own published bias
lists (bias_lists/oracle_list.txt and distractor_list.txt) and isolate
the spotter's behaviour from term-extraction quality.

Audio is never truncated: the .nlp files ship empty ts/endTs columns, so
there is no way to cut a reference to match a shortened wav without
introducing a boundary error, and at ~47x realtime full clips are cheap
enough that the question never arises. lol
"""
import argparse
import csv
import os

from captionlm.build_eval_set import convert_to_wav, reconstruct_transcript_text


def eval10_ids(earnings21_dir: str) -> list[str]:
    path = os.path.join(earnings21_dir, "eval10-file-metadata.csv")
    with open(path, encoding="utf-8") as f:
        return [row["file_id"] for row in csv.DictReader(f)]


def build_clip(earnings21_dir: str, file_id: str, out_dir: str) -> bool:
    mp3 = os.path.join(earnings21_dir, "media", f"{file_id}.mp3")
    nlp = os.path.join(earnings21_dir, "transcripts", "nlp_references", f"{file_id}.nlp")
    if not (os.path.exists(mp3) and os.path.exists(nlp)):
        print(f"skip {file_id}: missing media or transcript")
        return False

    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, f"{file_id}.wav")
    if not os.path.exists(wav):
        convert_to_wav(mp3, wav)
    with open(os.path.join(out_dir, f"{file_id}.txt"), "w", encoding="utf-8") as f:
        f.write(reconstruct_transcript_text(nlp))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earnings21_dir", help="Path to the cloned earnings21/ directory")
    parser.add_argument("--out", default="data/eval_data/bias_clips")
    parser.add_argument("--limit", type=int, default=None, help="Max clips to build")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use every file in earnings21-file-metadata.csv instead of the eval10 subset",
    )
    args = parser.parse_args()

    if args.all:
        meta = os.path.join(args.earnings21_dir, "earnings21-file-metadata.csv")
        with open(meta, encoding="utf-8") as f:
            ids = [row["file_id"] for row in csv.DictReader(f)]
    else:
        ids = eval10_ids(args.earnings21_dir)

    built = 0
    for file_id in ids:
        if args.limit is not None and built >= args.limit:
            break
        if build_clip(args.earnings21_dir, file_id, args.out):
            built += 1
            print(f"built {file_id} ({built})")
    print(f"Built {built} clips into {args.out}")


if __name__ == "__main__":
    main()
