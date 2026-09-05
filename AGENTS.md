# captionlm

Domain-biased captioning on Apple silicon, MLX, entirely local. Nothing is
trained: the term list compiles into a trie at call time. `README.md` has the
architecture, `SETUP.md` the install landmines and the commands that reproduce
every measurement.

## Read this first

**`docs/HANDOFF.md`** — what to work on next and why, the traps that have
already cost time, and the claims that must not be made from the existing
numbers. Read it before starting any eval work.

`docs/results/` is the measurement record, newest first; each file is a run
that happened, including the ones that failed. `docs/superpowers/plans/` is
specs and plans.

## House rules

- Measure before claiming. Every number in `docs/results/` names the model,
  the eval set and the configuration it came from.
- Only the `per-clip` condition is the shipped configuration. No product
  number may be quoted from a run that applies one term list to every clip.
- Long eval runs need `nohup`; they die with the terminal. Budget ~16 min per
  pass over the 9 Earnings-21 clips.
- `venv/bin/python -m pytest -q` should report 135 passed. Exactly four
  failures in `tests/test_biased_model.py` means the Metal XPC service died —
  reboot, do not debug.
