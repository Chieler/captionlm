import wave

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.terms import build_context_graph, load_tokenizer


def _write_silent_wav(path, seconds=1.0, sample_rate=16000):
    n_samples = int(seconds * sample_rate)
    with wave.open(path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_samples)


def test_unbiased_generate_matches_base_class_shape(tmp_path):
    wav_path = str(tmp_path / "silence.wav")
    _write_silent_wav(wav_path)

    model = load_biased_model(MODEL_ID)
    assert model.context_graph is None

    result = model.transcribe(wav_path)
    assert hasattr(result, "text")
    assert hasattr(result, "sentences")


def test_biased_generate_runs_with_context_graph(tmp_path):
    wav_path = str(tmp_path / "silence.wav")
    _write_silent_wav(wav_path)

    model = load_biased_model(MODEL_ID)
    tokenizer = load_tokenizer(MODEL_ID)
    blank_idx = len(model.vocabulary)
    model.context_graph = build_context_graph(["kubernetes"], tokenizer, blank_idx)

    result = model.transcribe(wav_path)
    assert hasattr(result, "text")


def test_blank_idx_matches_vocabulary_length():
    model = load_biased_model(MODEL_ID)
    # ConvASRDecoder appends blank as the last class: num_classes = len(vocabulary) + 1.
    # Silently using the vendored spotter's default blank_idx=0 spots nothing; this
    # assertion is the self-check the design doc calls for.
    assert len(model.vocabulary) > 0
    # CTC head has exactly len(vocabulary) + 1 output classes with blank last;
    # this is the invariant the blank-index computation above depends on.
    assert model.ctc_decoder.decoder_layers[0].weight.shape[0] == len(model.vocabulary) + 1
