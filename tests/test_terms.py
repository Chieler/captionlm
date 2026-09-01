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


def test_the_trie_carries_the_blank_token_per_ctc_topology():
    """CTC-WS needs blank arcs in the graph, not just in the model.

    A CTC alignment emits blanks between and inside tokens, so a trie with
    only token arcs would lose every token pass at the first blank frame and
    spot nothing. This is an architectural invariant that fails silently --
    the spotter would simply stop firing -- so it is asserted rather than
    assumed.
    """
    tokenizer = load_tokenizer(MODEL_ID)
    blank = tokenizer.get_piece_size()
    graph = build_context_graph(["Groq", "Grok", "linearizability"], tokenizer, blank)

    nodes, stack = {}, [graph.root]
    while stack:
        node = stack.pop()
        if node.index in nodes:
            continue
        nodes[node.index] = node
        stack.extend(n for n in node.next.values() if n.index != node.index)

    is_blank_node = lambda n: n.next.get(blank) is n  # noqa: E731

    # The root must NOT have a blank arc: the spotter seeds a fresh empty
    # token at the root on every frame, so a blank self-loop there would
    # only duplicate that.
    assert blank not in graph.root.next

    blank_nodes = [n for n in nodes.values() if n is not graph.root and is_blank_node(n)]
    assert blank_nodes, "no blank nodes in the graph at all"

    for node in nodes.values():
        if node is graph.root or is_blank_node(node):
            continue
        forward = {t for t, m in node.next.items() if m.index != node.index}
        if forward - {blank}:
            assert blank in node.next, f"node {node.index} has no blank detour"
        # A token held across several frames repeats, so every token node
        # needs a self-loop on its own token.
        assert any(m.index == node.index for m in node.next.values())


def test_the_blank_detour_rejoins_the_same_next_state():
    """Blank arcs must be a detour, not a dead end.

    "G" -> "ro" has to be reachable both directly and through any number of
    blank frames, and both paths must land on the SAME node -- otherwise the
    blank path builds a parallel branch whose end state is never marked as a
    word.
    """
    tokenizer = load_tokenizer(MODEL_ID)
    blank = tokenizer.get_piece_size()
    ids = tokenizer.encode("Groq", out_type=int)
    graph = build_context_graph(["Groq"], tokenizer, blank)

    first = graph.root.next[ids[0]]
    blank_node = first.next[blank]
    assert blank_node.next[blank] is blank_node, "blank node must self-loop"
    assert first.next[ids[1]] is blank_node.next[ids[1]]
