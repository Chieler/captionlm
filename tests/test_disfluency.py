from captionlm.disfluency import strip_restarts


def test_drops_the_first_copy_of_an_exact_repeat():
    # Real span from the read-aloud set: the reader restarted mid-sentence
    # and the model transcribed it faithfully.
    assert strip_restarts("enough the failure modes the failure modes are worth naming") == (
        "enough the failure modes are worth naming"
    )


def test_keeps_the_second_copy_because_it_carries_the_punctuation():
    assert strip_restarts("a common mistake mistake. Ordering matters") == (
        "a common mistake. Ordering matters"
    )


def test_catches_a_repeat_whose_restarted_word_came_back_garbled():
    # "Groq" is what was said; the model produced two different wrong
    # spellings of it, so the two copies are not textually identical.
    assert strip_restarts("Grok built custom silicon Grock built custom silicon for low latency") == (
        "Grock built custom silicon for low latency"
    )


def test_longest_repeat_wins_so_words_are_not_left_interleaved():
    # Taken as three unigram repeats this would collapse wrongly; the
    # 3-gram has to be found first.
    assert strip_restarts("x the failure modes the failure modes y") == "x the failure modes y"


def test_a_fuzzy_window_offset_from_the_real_repeat_does_not_win():
    # "enough the failure" vs "modes the failure" differ in exactly one
    # position and would qualify as a fuzzy repeat, but the exact 3-gram
    # sitting one word later is the real restart. Exact must be consumed
    # first, and enough/modes share no prefix.
    assert "enough" in strip_restarts("enough the failure modes the failure modes are worth naming")


def test_ordinary_english_with_two_words_in_common_is_left_alone():
    # A single mismatch inside a 2-gram leaves only one word matching, and
    # English supplies those constantly.
    assert strip_restarts("in the end the beginning of it") == "in the end the beginning of it"
    assert strip_restarts("the cat sat on the mat") == "the cat sat on the mat"


def test_leaves_clean_prose_untouched():
    # Measured: zero words removed across the 1517-word read-aloud
    # reference set, which is scripted prose with no restarts in it.
    text = "Serving is where the money actually goes. Nvidia still supplies most of the training hardware."
    assert strip_restarts(text) == text
