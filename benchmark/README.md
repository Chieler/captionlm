# Benchmark drop-off

Open http://localhost:8756/benchmark after starting the server.

Drop a recording and a plain-text source of truth with the same basename:

```text
benchmark/demo.m4a
benchmark/demo.reference.txt
```

The source text scores the completed transcript. It does not bias the model.
The page writes `demo.srt` and `demo.txt`, then reports normalized word error
rate plus substitutions, deletions, and insertions.
