# Vendored files

These files are derived from NVIDIA NeMo (Apache-2.0). They were fetched
from the NeMo `main` branch on 2026-08-29. The exact commit SHA was not
recorded at vendoring time, so it isn't stated here — recording a fabricated
SHA would be worse than admitting it wasn't captured.

See `LICENSE` in this directory for the full Apache-2.0 text these files
remain under.

## `context_graph_ctc.py`

- Upstream: `nemo/collections/asr/parts/context_biasing/context_graph_ctc.py`
- Byte-identical to upstream.

## `ctc_word_spotter.py`

- Upstream: `nemo/collections/asr/parts/context_biasing/ctc_based_word_spotter.py`
- Modified: one behavioral line (`tokenizer.ids_to_tokens` -> `vocabulary[idx]`,
  to work against parakeet-mlx's vocabulary list instead of a NeMo
  tokenizer object) plus one import-path fix (`captionlm.vendor.context_graph_ctc`
  instead of the NeMo-internal import path). See the file's own header
  comment for the same note.

## `fscore.py`

- Upstream: `nemo/collections/asr/parts/context_biasing/context_biasing_utils.py`
  (`compute_fscore`)
- Adapted to drop the `nemo_toolkit` dependency: takes an in-memory list of
  `{"text", "pred_text"}` dicts instead of a NeMo manifest file path, uses
  `kaldialign` directly instead of `nemo.utils.logging` for stats, and drops
  the per-sample-keyword-field mode (this project always scores against one
  global term list). The core alignment-based scoring algorithm is
  unchanged. See the file's own docstring for the same note.
