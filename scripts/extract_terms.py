"""Turn a source document into a bias term list file.

This is the missing link between captionlm/biasable.py and
scripts/sweep_cb_weight.py: the sweep takes conditions as term-list
files, so an extractor has to be able to write one.

The three modes are the three hypotheses worth separating. `terminology`
is pyate's statistical extraction with the biasable filter applied;
`ner` is spaCy named entities; `combined` is their union, which is the
mode the product would actually ship, because the two streams find
different things -- pyate finds multi-word domain phrases, NER finds the
proper nouns that pyate never returns.
"""
import argparse

from captionlm.biasable import extract_bias_terms
from captionlm.config import TERM_EXTRACTION_TOP_N
from captionlm.doc_import import extract_text


def terms_for(text: str, mode: str, top_n: int) -> list[str]:
    return extract_bias_terms(text, mode=mode, top_n=top_n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docs", nargs="+", help="Source document(s); text is pooled across all")
    parser.add_argument("--out", required=True, help="Term list file to write, one term per line")
    parser.add_argument("--mode", choices=("terminology", "ner", "combined"), default="combined")
    parser.add_argument("--top-n", type=int, default=TERM_EXTRACTION_TOP_N)
    args = parser.parse_args()

    text = "\n".join(extract_text(d) for d in args.docs)
    terms = terms_for(text, args.mode, args.top_n)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(terms) + "\n")
    print(f"{args.mode}: {len(terms)} terms -> {args.out}")
    print("  " + ", ".join(terms[:15]) + (" ..." if len(terms) > 15 else ""))


if __name__ == "__main__":
    main()
