"""Project-wide configuration constants."""
from dataclasses import dataclass

MODEL_ID = "mlx-community/parakeet-tdt_ctc-110m"
TERM_EXTRACTION_TOP_N = 50

# SEC EDGAR requires a descriptive contact string in the User-Agent header
# on every request. Fill this in with a real contact before running
# captionlm/eval_dataset.py against the live API.
SEC_CONTACT = "captionlm/1.0 (chielerli@gmail.com)"


@dataclass
class SpotterConfig:
    """CTC-WS hyperparameters. Defaults match NeMo's own
    run_word_spotter defaults; retune cb_weight against your own term
    list via eval.py's --cb-weight sweep before trusting these."""

    beam_threshold: float = 5.0
    cb_weight: float = 3.0
    ctc_ali_token_weight: float = 0.5
    keyword_threshold: float = -5.0
    blank_threshold: float = 0.8
    non_blank_threshold: float = 0.001
    intersection_threshold: float = 30.0
