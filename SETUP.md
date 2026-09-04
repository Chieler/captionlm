# Setup and evaluation

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e .
venv/bin/python -m spacy download en_core_web_sm
venv/bin/python -m pytest -q
```

`pyate==0.5.5` needs its `default_general_domain.en.zip` corpus installed at
`venv/lib/python3.12/site-packages/pyate/` and requires `setuptools<81`;
`requirements.txt` pins the latter. If term extraction reports either a
missing corpus or `pkg_resources`, repair those two prerequisites first.

## Reproduce the maintained regression

The read-aloud audio and per-clip term lists are tracked. This runs unbiased
and document-biased transcription with the 1.1B hybrid model:

```bash
venv/bin/python -m captionlm.eval data/read_aloud \
  data/eval_data/read_aloud_all_terms.txt \
  --model mlx-community/parakeet-tdt_ctc-1.1b \
  --cb-weight 3.0 \
  --out-json data/eval_results/read_aloud_regression.json
```

For the larger private Earnings-21 evaluation, clone
`revdotcom/speech-datasets` into `data/speech-datasets/`, then use the
per-clip builder/evaluation scripts. Do not quote a shared-list result as a
product result; the product applies each document only to its own recording.

Long evaluations should be run detached and their log polled. They can take
many minutes per pass and use cached local model weights after first download.
