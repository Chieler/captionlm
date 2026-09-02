# Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e .          # so `import captionlm` works from any directory
venv/bin/python -m spacy download en_core_web_sm
# See the two workarounds below if pyate import fails
```

The spaCy model (`en_core_web_sm`) is not a pip-installable pinned dependency
that `requirements.txt` can capture (it's fetched as a model package via
spaCy's own downloader), so it's a separate step.

## Known install landmines

### 1. `pyate` is missing its bundled reference corpus on PyPI

`pyate==0.5.5`'s PyPI package is missing its bundled
`default_general_domain.en.zip` reference corpus (an upstream packaging bug).
Without it, `pyate` raises a `FileNotFoundError` at import/use time.

Fix: download it from pyate's GitHub repo
(`kevinlu1248/pyate`, path `src/pyate/default_general_domain.en.zip`) and
place it at:

```
venv/lib/python3.12/site-packages/pyate/default_general_domain.en.zip
```

### 2. `pyate` imports `pkg_resources`, removed in `setuptools>=81`

`pyate` imports `pkg_resources`, which was removed from `setuptools>=81`.
`requirements.txt` already pins `setuptools<81` to work around this, but it's
easy to silently violate that pin with a later `pip install --upgrade
setuptools` (or a fresh venv bootstrap that pulls the newest setuptools
before `requirements.txt` is applied) — so keep the pin in mind if `pyate`
starts raising `ModuleNotFoundError: No module named 'pkg_resources'`.

## Reproducing the eval numbers on a fresh machine

The read-aloud set is tracked -- `.wav` and the extracted `.terms.txt` are
both in the repo -- so it runs straight after the install above. Model
weights download themselves on first use (`parakeet-tdt_ctc-1.1b` is 4.0 GB).

```bash
# combined scoring vocabulary: one list, so the metric is constant across clips
cat data/read_aloud/*.terms.txt | sort -u > data/eval_data/read_aloud_all_terms.txt

# WER / B-WER / U-WER, biased against unbiased
venv/bin/python -m captionlm.eval data/read_aloud \
    data/eval_data/read_aloud_all_terms.txt \
    --model mlx-community/parakeet-tdt_ctc-1.1b \
    --out-json data/eval_results/bwer_read_aloud_1b.json

# the cb_weight curve behind the 3.0 default
venv/bin/python scripts/sweep_cb_weight.py data/read_aloud \
    --condition per-clip=per-clip \
    --score-terms data/eval_data/read_aloud_all_terms.txt \
    --weights 0.5,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0 \
    --model mlx-community/parakeet-tdt_ctc-1.1b \
    --out-dir data/eval_results/sweep_read_aloud
```

Earnings-21 is 1.7 GB of public dataset and stays out of git. Clone it and
rebuild the clips:

```bash
git clone https://github.com/revdotcom/speech-datasets.git data/speech-datasets
venv/bin/python scripts/build_bias_clips.py data/speech-datasets/earnings21
# -> data/eval_data/bias_clips/<id>.wav + <id>.txt, the eval10 subset (9 clips)

venv/bin/python -m captionlm.eval data/eval_data/bias_clips \
    data/speech-datasets/earnings21/bias_lists/distractor_list.txt \
    --model mlx-community/parakeet-tdt_ctc-1.1b \
    --out-json data/eval_results/bwer_earnings21_1b.json
```

The oracle-against-distractor sweep that `docs/results/` records as open.
Both conditions score against the distractor vocabulary on purpose: scoring
the distractor run against the oracle list would silently drop every
distractor false accept, which is the harm the run exists to measure.
`cb_weight` 0.5 is inert on both eval sets and can be skipped. Budget ~16
min per pass on an unloaded machine, plus one shared baseline pass.

```bash
venv/bin/python scripts/sweep_cb_weight.py data/eval_data/bias_clips \
    --condition oracle=data/speech-datasets/earnings21/bias_lists/oracle_list.txt \
    --condition distractor=data/speech-datasets/earnings21/bias_lists/distractor_list.txt \
    --score-terms data/speech-datasets/earnings21/bias_lists/distractor_list.txt \
    --weights 1.0,2.0,3.0 \
    --model mlx-community/parakeet-tdt_ctc-1.1b \
    --out-dir data/eval_results/sweep_earnings21
```

Long runs die with the terminal. Detach them:

```bash
nohup venv/bin/python scripts/sweep_cb_weight.py ... > sweep.log 2>&1 &
```

## Layout

```
captionlm/   the package
scripts/     entry points, plus dashboard.html for scripts/serve.py
tests/
docs/        results and plans
dropoff/     drop audio (+ an optional source document) here to caption it
data/        eval sets, results, corpora -- gitignored except read_aloud/,
             whose .wav and .terms.txt are tracked so the numbers reproduce
```

The editable install above is what makes `venv/bin/python scripts/serve.py`
work from any working directory. Without it every command needs a
`PYTHONPATH=.` prefix and only runs from the repo root.
