from pathlib import Path

import pytest

from captionlm.doc_import import extract_text

FIXTURE = (
    Path(__file__).parent.parent
    / "static_assets"
    / "Philo_Homes_Organization_Structure_Proposal.docx"
)


def test_extract_text_handles_google_docs_tika_198():
    if not FIXTURE.exists():
        pytest.skip(f"fixture not present on this machine: {FIXTURE}")
    text = extract_text(str(FIXTURE))
    assert len(text) > 100
