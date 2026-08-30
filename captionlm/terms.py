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


def build_context_graph(
    terms: list[str],
    tokenizer: spm.SentencePieceProcessor,
    blank_idx: int,
) -> ContextGraphCTC:
    graph = ContextGraphCTC(blank_id=blank_idx)
    word_items = [(term, [tokenizer.encode(term, out_type=int)]) for term in terms]
    graph.add_to_graph(word_items)
    return graph
