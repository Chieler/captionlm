import json

from captionlm.eval import _wer, load_clips, score, transcribe_biased


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


def test_load_clips_reads_per_clip_terms(tmp_path):
    (tmp_path / "clip1.wav").write_bytes(b"\x00")
    (tmp_path / "clip1.txt").write_text("hello there", encoding="utf-8")
    (tmp_path / "clip1.terms.txt").write_text("net sales\nfree cash flow\n", encoding="utf-8")

    clips = load_clips(str(tmp_path), fallback_terms=["ignored"])

    assert clips[0]["terms"] == ["net sales", "free cash flow"]


def test_load_clips_falls_back_when_no_per_clip_terms(tmp_path):
    (tmp_path / "clip1.wav").write_bytes(b"\x00")
    (tmp_path / "clip1.txt").write_text("hello there", encoding="utf-8")

    clips = load_clips(str(tmp_path), fallback_terms=["combined term"])

    assert clips[0]["terms"] == ["combined term"]


def test_transcribe_biased_builds_a_graph_per_clip(monkeypatch):
    graphs_seen = []

    class FakeResult:
        text = "hello"

    class FakeModel:
        context_graph = None

        def transcribe(self, wav_path, chunk_duration=120.0):
            graphs_seen.append(self.context_graph)
            return FakeResult()

    monkeypatch.setattr(
        "captionlm.eval.build_context_graph",
        lambda terms, tokenizer, blank_idx: tuple(terms),
    )

    clips = [
        {"wav": "a.wav", "text": "x", "terms": ["alpha"]},
        {"wav": "b.wav", "text": "y", "terms": ["beta"]},
    ]
    samples = transcribe_biased(FakeModel(), clips, tokenizer=None, blank_idx=0)

    assert graphs_seen == [("alpha",), ("beta",)]
    assert [s["text"] for s in samples] == ["x", "y"]


def test_score_emits_one_record_per_clip():
    clips = [
        {"wav": "/tmp/a.wav", "text": "net sales grew", "terms": ["net sales"]},
        {"wav": "/tmp/b.wav", "text": "free cash flow rose", "terms": ["free cash flow"]},
    ]
    baseline = [
        {"text": "net sales grew", "pred_text": "net sails grew"},
        {"text": "free cash flow rose", "pred_text": "free cash flow rose"},
    ]
    biased = [
        {"text": "net sales grew", "pred_text": "net sales grew"},
        {"text": "free cash flow rose", "pred_text": "free cash flow rose"},
    ]

    stats = score(clips, baseline, biased, ["net sales", "free cash flow"])

    assert [r["clip"] for r in stats["per_clip"]] == ["a.wav", "b.wav"]
    assert stats["per_clip"][0]["baseline_wer"] > 0.0
    assert stats["per_clip"][0]["biased_wer"] == 0.0
    assert stats["biased"]["wer"] < stats["baseline"]["wer"]
    assert stats["per_clip"][0]["n_terms"] == 1


def test_wer_ignores_case_and_punctuation_but_counts_real_errors():
    # Regression: jiwer 4.0.0's default transform does not normalize, and
    # jiwer.wer() takes reference_transform (not truth_transform).
    assert _wer(["Net Sales grew."], ["net sales grew"]) == 0.0
    assert _wer(["net sales grew"], ["net sails grew"]) > 0.0


def test_score_ignores_case_and_punctuation():
    clips = [{"wav": "/tmp/a.wav", "text": "Net Sales grew.", "terms": ["net sales"]}]
    samples = [{"text": "Net Sales grew.", "pred_text": "net sales grew"}]

    stats = score(clips, samples, samples, ["net sales"])

    assert stats["baseline"]["wer"] == 0.0
