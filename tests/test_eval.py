from captionlm.eval import _wer, load_clips


def test_load_clips_pairs_wav_and_txt(tmp_path):
    (tmp_path / "clip1.wav").write_bytes(b"\x00")
    (tmp_path / "clip1.txt").write_text("hello there", encoding="utf-8")
    (tmp_path / "clip2.wav").write_bytes(b"\x00")
    (tmp_path / "clip2.txt").write_text("second clip", encoding="utf-8")
    (tmp_path / "orphan.wav").write_bytes(b"\x00")  # no matching .txt

    clips = load_clips(str(tmp_path))

    assert len(clips) == 2
    texts = {c["text"] for c in clips}
    assert texts == {"hello there", "second clip"}
    assert all(c["wav"].endswith(".wav") for c in clips)


def test_wer_ignores_case_and_punctuation_but_counts_real_errors():
    # Regression: jiwer 4.0.0's default transform does not normalize, and
    # jiwer.wer() takes reference_transform (not truth_transform).
    assert _wer(["Net Sales grew."], ["net sales grew"]) == 0.0
    assert _wer(["net sales grew"], ["net sails grew"]) > 0.0
