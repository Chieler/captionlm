from captionlm.eval import load_clips


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
