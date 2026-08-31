"""Build a ContextGraphCTC from a domain term list, tokenized against the
model's own sentencepiece vocabulary (same vocab and id order the model's
CTC head outputs logits over — verified: sentencepiece piece ids for this
model line up 1:1 with the config's aux_ctc.decoder.vocabulary list)."""
import sentencepiece as spm
from huggingface_hub import hf_hub_download

from captionlm.vendor.context_graph_ctc import ContextGraphCTC


def load_term_list(path: str) -> list[str]:
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(line)
    return terms


def load_tokenizer(model_id: str) -> spm.SentencePieceProcessor:
    model_path = hf_hub_download(model_id, "tokenizer.model")
    return spm.SentencePieceProcessor(model_file=model_path)


_SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh")


def _surface_variants(term: str) -> set[str]:
    """Every spoken surface form of one term to enter into the trie.

    The list holds "subtree" and the speaker says "subtrees", so the
    spotter never fires. Regular -s/-es on the final word covers the
    measured cases; irregular plurals and comparatives ("shorter") are
    left alone, because a real morphological generator is a larger change
    than the remaining residual justifies.

    An already-plural term gains a nonsense variant -- "subtrees" yields
    "subtreeses". That is deliberate: no speaker says it, so the extra
    trie entry can never fire, and detecting real plurals costs more than
    an inert entry does. Do not "fix" it by skipping words ending in s;
    that would drop "witness" -> "witnesses" too.
    """
    variants = {term, term.lower()}
    for base in (term, term.lower()):
        head, _, last = base.rpartition(" ")
        if not last:
            continue
        plural = last + ("es" if last.endswith(_SIBILANT_ENDINGS) else "s")
        variants.add(f"{head} {plural}" if head else plural)
    return variants


def build_context_graph(
    terms: list[str],
    tokenizer: spm.SentencePieceProcessor,
    blank_idx: int,
) -> ContextGraphCTC:
    if blank_idx != tokenizer.get_piece_size():
        raise ValueError(
            f"blank_idx {blank_idx} does not match this tokenizer's vocabulary "
            f"size {tokenizer.get_piece_size()}. The tokenizer and the model "
            f"weights come from different model ids; the token ids would "
            f"address the wrong vocabulary and the graph would never fire."
        )
    graph = ContextGraphCTC(blank_id=blank_idx)
    word_items = []
    for term in terms:
        variants = _surface_variants(term)
        # Shortest first: a plural variant can share a token prefix with the
        # base term (e.g. "kubernetes" -> "kuberneteses" tokenize as the same
        # 6 tokens plus one more). add_to_graph reuses an existing node for a
        # shared prefix without touching its is_end flag, so if the longer
        # variant were inserted first it would leave that shared node
        # is_end=False and the base term would silently stop matching. A set
        # has no defined order, so sort explicitly rather than rely on luck.
        word_items.append(
            (term, [tokenizer.encode(v, out_type=int) for v in sorted(variants, key=len)])
        )
    graph.add_to_graph(word_items)
    return graph
