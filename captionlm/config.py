"""Project-wide configuration constants."""
from dataclasses import dataclass

# Both are hybrid TDT-CTC, which is what the CTC-WS spotter requires: a
# TDT-only or CTC-only checkpoint has no compatible pair of heads. Measured
# on the read-aloud eval set at cb_weight 3.0 -- 110m: 0.1160 unbiased,
# 0.0989 biased, 438 MB, ~47x realtime. 1.1b: 0.1009 unbiased, 0.0778
# biased, 4.0 GB, ~28x realtime. Neither trains anything at use time.
MODEL_ID_110M = "mlx-community/parakeet-tdt_ctc-110m"
MODEL_ID_1_1B = "mlx-community/parakeet-tdt_ctc-1.1b"

# The small model stays the default: 4.0 GB is the user's disk and the
# user's download, and that is their choice to make with --model, not ours
# to make silently on their behalf.
MODEL_ID = MODEL_ID_110M
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
