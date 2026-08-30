from captionlm.config import MODEL_ID
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer


def test_load_term_list_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "terms.txt"
    path.write_text("kubernetes\n\n# a comment\nHelm chart\n", encoding="utf-8")
    assert load_term_list(str(path)) == ["kubernetes", "Helm chart"]


def test_single_word_term_roundtrips_through_graph():
    tokenizer = load_tokenizer(MODEL_ID)
    blank_idx = tokenizer.get_piece_size()  # same vocab as the model's CTC head
    graph = build_context_graph(["kubernetes"], tokenizer, blank_idx)

    token_ids = tokenizer.encode("kubernetes", out_type=int)
    node = graph.root
    for token_id in token_ids:
        node = node.next[token_id]
    assert node.is_end
    assert node.word == "kubernetes"


def test_multiword_phrase_roundtrips_through_graph():
    tokenizer = load_tokenizer(MODEL_ID)
    blank_idx = tokenizer.get_piece_size()
    graph = build_context_graph(["San Francisco"], tokenizer, blank_idx)

    token_ids = tokenizer.encode("San Francisco", out_type=int)
    node = graph.root
    for token_id in token_ids:
        node = node.next[token_id]
    assert node.is_end
    assert node.word == "San Francisco"
