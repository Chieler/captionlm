from captionlm.biasable import (
    extract_entities,
    filter_biasable,
    is_biasable,
    is_rare,
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


def test_rarity_separates_words_the_asr_gets_from_words_it_misses():
    # The threshold has to sit between these two: "node" is a common word
    # the model already transcribes, "raft" is a protocol name it does not.
    assert not is_rare("node")
    assert not is_rare("line")
    assert not is_rare("revenue")
    assert is_rare("raft")
    assert is_rare("quorum")
    assert is_rare("linearizability")


def test_rare_lowercase_jargon_is_biasable_though_it_is_no_proper_noun():
    # The previous part-of-speech rule rejected all of these, because a
    # lowercase common noun is exactly what they look like.
    assert is_biasable("quorum")
    assert is_biasable("tombstone")
    assert is_biasable("linearizability")


def test_capitalization_beats_rarity_for_single_words():
    # "Apple" is a company however common "apple" is.
    assert is_biasable("Apple")
    assert not is_biasable("apple")


def test_extract_entities_finds_proper_nouns_spacy_ner_does_not_label():
    # en_core_web_sm assigns no entity label at all to any of these, and
    # each appears exactly once, at the start of its only sentence.
    text = (
        "Local inference matters more than it sounds. Ollama made it trivial. "
        "Anthropic published the protocol. Perplexity built a search product."
    )
    found = {t.lower() for t in extract_entities(text)}
    assert {"ollama", "anthropic", "perplexity"} <= found


def test_extract_entities_finds_multiword_lowercase_jargon():
    text = (
        "A bloom filter sits in front of each run. The defense is a fencing "
        "token. Snapshot isolation permits write skew in practice."
    )
    found = {t.lower() for t in extract_entities(text)}
    assert {"bloom filter", "fencing token"} <= found
    # "write skew" is the documented gap: "write" tags as a VERB here, and
    # admitting verb neighbours costs more fragments than it recovers terms.
    assert "write skew" not in found


def test_bigram_harvest_rejects_sentence_fragments():
    # All three came out of the harvest before the neighbour had to be
    # nominal, and all three are noise on any document that is not itself
    # the transcript being scored.
    text = "Agent tooling actually stands still. Alibaba ships open weights."
    found = {t.lower() for t in extract_entities(text)}
    assert "tooling actually" not in found  # adverb neighbour
    assert "stands still" not in found  # verb neighbour
    assert "alibaba" in found


def test_rare_but_ordinary_non_nouns_are_not_harvested():
    # All four were extracted from the read-aloud scripts and caused
    # substitutions. They clear the rarity bar because pyate's corpus is
    # small, but they are ordinary English the model already gets right.
    text = (
        "Guardrails catch some of this. Better tool design catches more. "
        "Snapshot isolation lets readers see a consistent view of real data."
    )
    found = {t.lower() for t in extract_entities(text)}
    assert "better" not in found
    assert "lets" not in found
    assert "real" not in found


def test_rare_nouns_are_still_harvested():
    text = (
        "The protocol requires a quorum of members. A tombstone survives "
        "until every replica has seen it, and sharding spreads the load."
    )
    found = {t.lower() for t in extract_entities(text)}
    assert {"quorum", "tombstone", "sharding"} <= found
