"""Replace transducer word spans with higher-confidence CTC-WS spotted
terms, by frame-overlap, following the same replace-on-overlap idea as
NeMo's merge_alignment_with_ws_hyps (this is a from-scratch
reimplementation against parakeet-mlx's AlignedToken type, not a port —
parakeet-mlx has no equivalent of NeMo's rnnt_utils.Hypothesis)."""
from parakeet_mlx.alignment import AlignedToken

from captionlm.vendor.ctc_word_spotter import WSHyp


def _group_words(tokens: list[AlignedToken]) -> list[list[AlignedToken]]:
    words: list[list[AlignedToken]] = []
    for tok in tokens:
        if words and " " not in tok.text:
            words[-1].append(tok)
        else:
            words.append([tok])
    return words


def _overlap_pct(word_start: float, word_end: float, hyp_start: float, hyp_end: float) -> float:
    intersection = max(0.0, min(word_end, hyp_end) - max(word_start, hyp_start))
    word_len = word_end - word_start
    return 0.0 if word_len <= 0 else 100.0 * intersection / word_len


def merge_spans(
    tokens: list[AlignedToken],
    ws_hyps: list[WSHyp],
    time_ratio: float,
    intersection_threshold: float = 30.0,
) -> list[AlignedToken]:
    if not tokens or not ws_hyps:
        return tokens

    words = _group_words(tokens)
    used = [False] * len(words)
    replacements: dict[int, tuple[int, AlignedToken]] = {}

    for hyp in sorted(ws_hyps, key=lambda h: h.start_frame):
        hyp_start = hyp.start_frame * time_ratio
        hyp_end = (hyp.end_frame + 1) * time_ratio
        overlap_idxs = [
            i
            for i, w in enumerate(words)
            if not used[i]
            and _overlap_pct(w[0].start, w[-1].end, hyp_start, hyp_end) >= intersection_threshold
        ]
        if not overlap_idxs:
            continue

        first_idx, last_idx = overlap_idxs[0], overlap_idxs[-1]
        for i in range(first_idx, last_idx + 1):
            used[i] = True

        lead_space = " " if " " in words[first_idx][0].text else ""
        replacements[first_idx] = (
            last_idx,
            AlignedToken(
                id=-1,
                text=lead_space + hyp.word,
                start=words[first_idx][0].start,
                duration=words[last_idx][-1].end - words[first_idx][0].start,
                confidence=1.0,
            ),
        )

    merged: list[AlignedToken] = []
    i = 0
    while i < len(words):
        if i in replacements:
            last_idx, replacement = replacements[i]
            merged.append(replacement)
            i = last_idx + 1
        else:
            merged.extend(words[i])
            i += 1
    return merged
