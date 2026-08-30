from pathlib import Path

from captionlm.doc_import import extract_text

FIXTURE = (
    Path(__file__).parent.parent
    / "static_assets"
    / "Philo_Homes_Organization_Structure_Proposal.docx"
)


def test_extract_text_handles_google_docs_tika_198():
    assert FIXTURE.exists(), f"missing fixture file: {FIXTURE}"
    text = extract_text(str(FIXTURE))
    assert len(text) > 100
