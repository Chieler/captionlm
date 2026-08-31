import pytest

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


def test_build_context_graph_rejects_a_blank_idx_from_a_different_model():
    # A tokenizer from one model and a blank_idx from another produce token
    # ids that address the wrong vocabulary. The graph would build without
    # complaint and then never fire, which is the worst possible failure:
    # biasing that silently does nothing.
    tokenizer = load_tokenizer(MODEL_ID)
    wrong_blank_idx = tokenizer.get_piece_size() + 500

    with pytest.raises(ValueError, match="blank_idx"):
        build_context_graph(["Kubernetes"], tokenizer, wrong_blank_idx)


def test_build_context_graph_accepts_the_matching_blank_idx():
    tokenizer = load_tokenizer(MODEL_ID)
    graph = build_context_graph(["Kubernetes"], tokenizer, tokenizer.get_piece_size())
    assert graph is not None
