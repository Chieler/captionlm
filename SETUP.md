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

## Layout

```
captionlm/   the package
scripts/     entry points, plus dashboard.html for scripts/serve.py
tests/
docs/        results and plans
dropoff/     drop audio (+ an optional source document) here to caption it
data/        eval sets, results, corpora -- all gitignored except read_aloud/
```

The editable install above is what makes `venv/bin/python scripts/serve.py`
work from any working directory. Without it every command needs a
`PYTHONPATH=.` prefix and only runs from the repo root.
