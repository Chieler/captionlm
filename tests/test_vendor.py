from captionlm.vendor.context_graph_ctc import ContextGraphCTC
from captionlm.vendor.ctc_word_spotter import Token, beam_pruning, state_pruning
from captionlm.vendor.fscore import compute_fscore


def test_context_graph_builds_simple_word():
    graph = ContextGraphCTC(blank_id=99)
    graph.add_to_graph([("hi", [[1, 2]])])
    node = graph.root.next[1].next[2]
    assert node.is_end
    assert node.word == "hi"


def test_beam_pruning_drops_low_score_tokens():
    root = ContextGraphCTC(blank_id=99).root
    tokens = [Token(root, score=0.0), Token(root, score=-10.0)]
    pruned = beam_pruning(tokens, beam_threshold=5.0)
    assert len(pruned) == 1
    assert pruned[0].score == 0.0


def test_state_pruning_keeps_best_per_state():
    root = ContextGraphCTC(blank_id=99).root
    tokens = [Token(root, score=1.0), Token(root, score=5.0)]
    pruned = state_pruning(tokens)
    assert len(pruned) == 1
    assert pruned[0].score == 5.0


def test_compute_fscore_perfect_match():
    samples = [{"text": "deploy the kubernetes cluster", "pred_text": "deploy the kubernetes cluster"}]
    precision, recall, fscore = compute_fscore(samples, ["kubernetes"])
    assert precision > 0.99
    assert recall > 0.99
    assert fscore > 0.99


def test_compute_fscore_missed_term():
    samples = [{"text": "deploy the kubernetes cluster", "pred_text": "deploy the koobernetti cluster"}]
    precision, recall, fscore = compute_fscore(samples, ["kubernetes"])
    assert recall < 0.01
