"""Extract plain text from local files (docx, pdf, csv, and whatever
else extractous/Tika support), including a workaround for
Google-Docs-exported .docx files that trigger Tika's TIKA-198 error."""
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from extractous import Extractor

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)
W = f"{{{W_NS}}}"


def unwrap_google_docs_sdt(xml_bytes):
    """Unwrap <w:sdt> tags Google Docs inserts (tag="goog_rdk_*") around runs.

    Apache POI/Tika's OOXMLParser throws TIKA-198 on these when a .docx was
    exported from Google Docs. They carry no real content control, so
    replacing each with its own sdtContent children is safe.
    """
    root = ET.fromstring(xml_bytes)
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for sdt in list(root.iter(W + "sdt")):
        sdt_pr = sdt.find(W + "sdtPr")
        tag_el = sdt_pr.find(W + "tag") if sdt_pr is not None else None
        tag_val = tag_el.get(W + "val") if tag_el is not None else ""
        if not (tag_val or "").startswith("goog_rdk"):
            continue
        content = sdt.find(W + "sdtContent")
        parent = parent_map[sdt]
        idx = list(parent).index(sdt)
        parent.remove(sdt)
        for i, child in enumerate(content if content is not None else []):
            parent.insert(idx + i, child)
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def fix_google_docs_docx(path):
    """Return a temp copy of a Google-Docs-exported .docx with the
    goog_rdk sdt wrappers stripped, so Tika can parse it."""
    with zipfile.ZipFile(path) as src:
        fd, tmp_path = tempfile.mkstemp(suffix=".docx")
        with os.fdopen(fd, "wb") as tmp_f, zipfile.ZipFile(tmp_f, "w", zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "word/document.xml":
                    data = unwrap_google_docs_sdt(data)
                dst.writestr(item, data)
    return tmp_path


def extract_text(path: str, max_length: int = 50_000) -> str:
    extractor = Extractor().set_extract_string_max_length(max_length)
    try:
        result, _metadata = extractor.extract_file_to_string(path)
    except TypeError as e:
        if path.lower().endswith(".docx") and "TIKA-198" in str(e):
            fixed = fix_google_docs_docx(path)
            try:
                result, _metadata = extractor.extract_file_to_string(fixed)
            finally:
                os.remove(fixed)
        else:
            raise
    return result


if __name__ == "__main__":
    static_assets_dir = os.path.join(os.path.dirname(__file__), "..", "static_assets")
    for asset in os.listdir(static_assets_dir):
        p = os.path.join(static_assets_dir, asset)
        print(p, extract_text(p))
