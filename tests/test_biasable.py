from captionlm.biasable import (
    extract_entities,
    filter_biasable,
    is_biasable,
    merge_term_lists,
)


def test_rejects_junk_seen_in_real_pyate_output():
    # Every one of these came out of pyate on a real SEC earnings release.
    assert not is_biasable("+")  # no letters
    assert not is_biasable("of")  # too short
    assert not is_biasable("line results in the second quarter")  # fragment
    assert not is_biasable("line of undercar repair services")  # fragment
    assert not is_biasable("share of costs")  # function word


def test_rejects_bare_common_nouns_the_asr_already_gets():
    # Biasing a word the model already transcribes correctly has no recall
    # upside and a real chance of overwriting something right.
    assert not is_biasable("line")
    assert not is_biasable("revenue")
    assert not is_biasable("shares")


def test_keeps_proper_nouns_acronyms_and_real_phrases():
    assert is_biasable("Alcoa")
    assert is_biasable("Airbus")
    assert is_biasable("EBITDA")
    assert is_biasable("comparable store sales")


def test_rejects_terms_longer_than_the_word_cap():
    # cb_weight is applied per token in the spotter, so long entries
    # accumulate the largest bonus and defeat the false-accept guard hardest.
    assert not is_biasable("one two three four five")


def test_filter_biasable_preserves_order_of_survivors():
    assert filter_biasable(["+", "Alcoa", "line", "Airbus"]) == ["Alcoa", "Airbus"]


def test_merge_term_lists_dedupes_case_insensitively_keeping_first():
    assert merge_term_lists(["Alcoa", "EBITDA"], ["alcoa", "Airbus"]) == [
        "Alcoa",
        "EBITDA",
        "Airbus",
    ]


def test_extract_entities_finds_orgs_and_people_not_common_nouns():
    text = (
        "Monro, Inc. reported results. Brett Ponton discussed the quarter. "
        "Analysts from Jefferies asked about revenue and shares outstanding. "
        "Monro, Inc. confirmed guidance."
    )
    terms = extract_entities(text)
    lowered = {t.lower() for t in terms}

    assert any("monro" in t for t in lowered)
    # Common nouns must not survive, whatever the NER model tags.
    assert "revenue" not in lowered
    assert "shares" not in lowered
