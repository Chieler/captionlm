from parakeet_mlx.alignment import AlignedToken

from captionlm.fusion import accepted_replacements, fuse_text, fuse_tokens


def _words(text: str) -> list[str]:
    return text.lower().split()


def test_takes_the_second_opinion_on_a_span_it_gets_right():
    # Real span: the spotter's merge left "guardrails catch" as three wrong
    # words. Whisper heard it correctly, and "guardrails" clears the
    # well-formedness rule because it is on the speaker's own term list --
    # which is where most domain words that general English has never heard
    # of get their licence to appear.
    assert fuse_text(
        "the call failed bar rails catches some of this",
        "the call failed guardrails catch some of this",
        ["guardrails"],
    ) == "the call failed guardrails catch some of this"


def test_refuses_a_span_holding_a_word_nobody_says():
    # Whisper writes "writeahead log" where the reference has two words.
    # Well-formed English is the price of replacing a span.
    assert fuse_text(
        "the storage engine a rank ahead log makes durability cheap",
        "the storage engine a writeahead log makes durability cheap",
        [],
    ) == "the storage engine a rank ahead log makes durability cheap"


def test_refuses_a_span_holding_a_digit():
    # The scripts spell numbers out on purpose; Whisper normalises them back.
    assert fuse_text(
        "iterates ten times faster", "iterates 10 times faster", []
    ) == "iterates ten times faster"


def test_never_spends_a_term_the_spotter_found():
    # Whisper is not biased and cannot know the vocabulary, so where it would
    # overwrite a listed term it is wrong by construction.
    assert fuse_text(
        "and Paxos assume that machines work",
        "and taxes assume that machines work",
        ["Paxos"],
    ) == "and Paxos assume that machines work"


def test_a_term_may_still_be_replaced_by_another_term():
    assert fuse_text(
        "then Paxos assume that machines work",
        "then quorum assume that machines work",
        ["Paxos", "quorum"],
    ) == "then quorum assume that machines work"


def test_a_proper_noun_does_not_lend_its_capital_to_what_replaces_it():
    assert fuse_text(
        "then Paxos assumes that", "then quorum assumes that", ["Paxos", "quorum"]
    ) == "then quorum assumes that"


def test_refuses_words_only_the_second_model_heard():
    # A pure insertion: parakeet is the backbone because it is the better
    # transcript, and Whisper's extra words are usually its own repetition.
    assert fuse_text(
        "replicas that have drifted apart", "replicas that have to drifted apart", []
    ) == "replicas that have drifted apart"


def test_accepts_a_deletion_the_second_model_makes():
    # The mirror case is safe: both models heard this stretch, and only one
    # of them transcribed the reader's false start.
    assert fuse_text(
        "the chunking strategy is always part is almost always the part",
        "the chunking strategy is almost always the part",
        [],
    ) == "the chunking strategy is almost always the part"


def test_keeps_the_capital_and_the_punctuation_of_the_span_it_replaces():
    assert fuse_text(
        "It failed. Bar rails catches.", "it failed. guardrails catch.", ["guardrails"]
    ) == "It failed. Guardrails catch."


def test_a_replacement_is_reported_against_primary_indices():
    replacements = accepted_replacements(
        _words("the call failed bar rails catches some"),
        _words("the call failed guardrails catch some"),
        ["guardrails"],
    )
    assert replacements == {3: (6, ["guardrails", "catch"])}


def _token(text: str, start: float) -> AlignedToken:
    return AlignedToken(id=0, text=text, start=start, duration=1.0)


def test_fusing_tokens_collapses_the_span_into_its_own_time_range():
    tokens = [
        _token(" failed", 0.0),
        _token(" bar", 1.0),
        _token(" rails", 2.0),
        _token(" catches", 3.0),
        _token(" some", 4.0),
    ]
    fused = fuse_tokens(tokens, "failed guardrails catch some", ["guardrails"])
    assert [t.text for t in fused] == [" failed", " guardrails catch", " some"]
    # The replacement stands in the same stretch of audio as the words it
    # replaced, so a caption cue still ends where the speaker finished.
    assert (fused[1].start, fused[1].end) == (1.0, 4.0)


def test_fusing_tokens_keeps_trailing_punctuation_of_the_span():
    tokens = [_token(" bar", 0.0), _token(" rails", 1.0), _token(".", 2.0), _token(" some", 2.0)]
    fused = fuse_tokens(tokens, "guardrails catch. some", ["guardrails"])
    assert [t.text for t in fused] == [" Guardrails catch.", " some"]


def test_identical_transcripts_are_left_alone():
    text = "the storage engine makes durability cheap"
    assert fuse_text(text, text, []) == text
