import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from caption_dropoff import find_pairs, partition_documents  # noqa: E402


def test_several_documents_can_bias_one_recording(tmp_path):
    # One recording, three source documents. Before this, find_pairs held a
    # single document per basename and the other two were silently dropped.
    for name in ("standup.m4a", "standup.txt", "standup-notes.md", "standup_slides.csv"):
        (tmp_path / name).write_text("x")
    audio, docs = find_pairs(str(tmp_path))[0]

    assert os.path.basename(audio) == "standup.m4a"
    assert [os.path.basename(d) for d in docs] == [
        "standup-notes.md", "standup.txt", "standup_slides.csv"
    ]


def test_a_document_goes_to_the_longest_recording_it_could_match(tmp_path):
    for name in ("standup.m4a", "standup-part2.m4a", "standup-part2.txt"):
        (tmp_path / name).write_text("x")
    got = {os.path.basename(a): [os.path.basename(d) for d in ds] for a, ds in find_pairs(str(tmp_path))}

    assert got == {"standup.m4a": [], "standup-part2.m4a": ["standup-part2.txt"]}


def test_a_run_does_not_feed_its_own_output_back_in(tmp_path):
    # standup.terms.txt reads as a document for standup.m4a, and
    # standup.converted.wav as a recording of its own.
    for name in ("standup.m4a", "standup.terms.txt", "standup.converted.wav", "standup.srt"):
        (tmp_path / name).write_text("x")
    pairs = find_pairs(str(tmp_path))

    assert [(os.path.basename(a), ds) for a, ds in pairs] == [("standup.m4a", [])]


def test_a_document_naming_no_recording_biases_every_one(tmp_path):
    # Real documents are called glossary.pdf, not standup.pdf. Requiring the
    # rename is a contract nobody keeps, so an unmatched document is shared.
    for name in ("standup.m4a", "retro.m4a", "standup.txt", "glossary.txt"):
        (tmp_path / name).write_text("x")
    got = {os.path.basename(a): [os.path.basename(d) for d in ds]
           for a, ds in find_pairs(str(tmp_path))}

    assert got == {
        "standup.m4a": ["glossary.txt", "standup.txt"],
        "retro.m4a": ["glossary.txt"],
    }


def test_naming_a_document_after_a_recording_still_scopes_it(tmp_path):
    # The escape hatch from sharing: standup.txt must not reach retro.m4a.
    for name in ("standup.m4a", "retro.m4a", "standup.txt"):
        (tmp_path / name).write_text("x")
    named, shared = partition_documents(str(tmp_path))

    assert shared == []
    assert {os.path.basename(a): [os.path.basename(d) for d in ds]
            for a, ds in named.items()} == {"standup.m4a": ["standup.txt"], "retro.m4a": []}


def test_the_dropoff_readme_is_not_a_source_document(tmp_path):
    # Unmatched, so sharing would bias every recording toward this repo.
    (tmp_path / "standup.m4a").write_text("x")
    (tmp_path / "README.md").write_text("# Drop-off")

    assert find_pairs(str(tmp_path)) == [(str(tmp_path / "standup.m4a"), [])]
