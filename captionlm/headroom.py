"""Does the model's own unbiased decode confidence at a candidate term's
span predict whether biasing it would help or hurt? This module computes
that signal; it does not decide anything or touch any production path --
see docs/RESULTS.md.
"""
import numpy as np

from captionlm.vendor.context_graph_ctc import ContextGraphCTC
from captionlm.vendor.ctc_word_spotter import (
    Token,
    WSHyp,
    beam_pruning,
    get_ctc_word_alignment,
    state_pruning,
)


def raw_spot(
    logprobs: np.ndarray,
    context_graph: ContextGraphCTC,
    blank_idx: int,
    cb_weight: float = 0.0,
    beam_threshold: float = 5.0,
    keyword_threshold: float = -5.0,
    blank_threshold: float = 0.8,
    non_blank_threshold: float = 0.001,
) -> list[WSHyp]:
    """The token-passing beam search from vendor/ctc_word_spotter.py's
    run_word_spotter, unmodified, stopping before that function's own
    trailing find_best_hyps + filter_wb_hyps calls.

    run_word_spotter itself cannot be reused here: it already runs the
    exact argmax comparison this module computes on its own (as
    filter_wb_hyps), but only as a binary keep/drop, and returns only
    what survives. At cb_weight=0 that survival bar is far too strict to
    be a usable signal -- a candidate only wins outright against the
    argmax path with zero boost helping it, which this module needs the
    graded version of, not the pass/fail.
    """
    start_state = context_graph.root
    active_tokens: list[Token] = []
    next_tokens: list[Token] = []
    spotted_words: list[WSHyp] = []
    blank_threshold = np.log(blank_threshold)
    non_blank_threshold = np.log(non_blank_threshold)

    for frame in range(logprobs.shape[0]):
        active_tokens.append(Token(start_state, start_frame=frame))
        best_score = None
        for token in active_tokens:
            if token.state is context_graph.root and logprobs[frame][blank_idx] > blank_threshold:
                continue
            for transition_state in token.state.next:
                if (
                    token.state is context_graph.root
                    and logprobs[frame][int(transition_state)] < non_blank_threshold
                ):
                    continue
                if transition_state != blank_idx:
                    current_score = token.score + logprobs[frame][int(transition_state)].item() + cb_weight
                else:
                    current_score = token.score + logprobs[frame][int(transition_state)].item()
                if not best_score:
                    best_score = current_score
                else:
                    if current_score < best_score - beam_threshold:
                        continue
                    elif current_score > best_score:
                        best_score = current_score
                new_token = Token(token.state.next[transition_state], current_score, token.start_frame)
                if new_token.state.is_end and new_token.score > keyword_threshold:
                    spotted_words.append(
                        WSHyp(
                            word=new_token.state.word,
                            score=new_token.score,
                            start_frame=new_token.start_frame,
                            end_frame=frame,
                        )
                    )
                    if len(new_token.state.next) == 1:
                        if current_score is best_score:
                            best_score = None
                        continue
                next_tokens.append(new_token)
        next_tokens = beam_pruning(next_tokens, beam_threshold)
        next_tokens = state_pruning(next_tokens)
        active_tokens = next_tokens
        next_tokens = []
    return spotted_words


def argmax_span_score(word_alignment: list[tuple], start_frame: int, end_frame: int) -> float:
    """Overlap-weighted sum of get_ctc_word_alignment's word scores across
    [start_frame, end_frame]. Same accumulation vendor/ctc_word_spotter.py's
    filter_wb_hyps does internally (its overall_spot_score) -- the first
    overlapping word counts in full, each subsequent overlapping word is
    weighted by its overlap percentage -- factored out here as a graded
    value because filter_wb_hyps only returns a survive/drop decision.

    Diverges from the vendored filter_wb_hyps in two edge cases: an empty
    word_alignment (vendor treats this as "no competitor exists, don't
    filter anything out"; here it returns 0.0, the most pessimistic
    possible competitor score) and a span that overlaps no word in the
    alignment at all (vendor drops such hyps from consideration; here it
    also returns 0.0). Both are unreachable in practice on real speech --
    there is essentially always some argmax word somewhere near any
    candidate span -- so this is left as an accepted divergence, not a
    deliberate design choice, rather than special-cased.
    """
    hyp_interval = set(range(start_frame, end_frame + 1))
    overall_spot_score = 0.0
    hyp_intersects = False
    for _word, l, r, score in word_alignment:
        word_interval = set(range(l, r + 1))
        intersection_part = 100 / len(word_interval) * len(hyp_interval & word_interval)
        if intersection_part:
            if not hyp_intersects:
                overall_spot_score = score
            else:
                overall_spot_score += intersection_part / 100 * score
            hyp_intersects = True
    return overall_spot_score


def unbiased_headroom(
    context_graph: ContextGraphCTC,
    logprobs: np.ndarray,
    asr_model,
    blank_idx: int,
    ctc_ali_token_weight: float = 0.5,
    beam_threshold: float = 5.0,
    keyword_threshold: float = -5.0,
    blank_threshold: float = 0.8,
    non_blank_threshold: float = 0.001,
) -> dict[str, float]:
    """One chunk's headroom score per term in context_graph that was
    spotted at all. A term absent from the result had no surviving
    candidate span in this chunk -- the caller (which sees every chunk
    in a clip) should treat a term missing from every chunk's result the
    same way: no acoustic trace found anywhere in the clip.
    """
    hyps = raw_spot(
        logprobs, context_graph, blank_idx, cb_weight=0.0,
        beam_threshold=beam_threshold, keyword_threshold=keyword_threshold,
        blank_threshold=blank_threshold, non_blank_threshold=non_blank_threshold,
    )
    alignment = get_ctc_word_alignment(logprobs, asr_model, token_weight=ctc_ali_token_weight, blank_idx=blank_idx)

    scores: dict[str, float] = {}
    for hyp in hyps:
        gap = hyp.score - argmax_span_score(alignment, hyp.start_frame, hyp.end_frame)
        if hyp.word not in scores or gap > scores[hyp.word]:
            scores[hyp.word] = gap
    return scores
