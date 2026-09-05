# Setup

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
