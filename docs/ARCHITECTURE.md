# Architecture

```text
trusted notes / slides / glossary
              |
              v
filtered domain terms -> context-graph trie
              |                 |
audio --------+-----------------v
        Parakeet TDT base transcript + CTC word spotter
                              |
                              v
                 conservative aligned merge -> SRT
```

`captionlm.biasable` extracts a bounded list of useful terms;
`captionlm.terms` compiles them into a trie; `captionlm.biased_model` runs a
hybrid TDT-CTC Parakeet checkpoint; and `captionlm.merge` replaces only
time-aligned spans in the base transcript. The graph is constructed for each
recording, so no user data is used for training or retained by a service.

The CTC head is required for the word spotter. Consequently, a plain TDT
checkpoint can be used as an independent baseline but cannot replace the
hybrid transcription-and-biasing path. NVIDIA's 0.6B-v3 checkpoint is such a
plain TDT model in the available MLX conversion: it transcribes, but lacks the
trained CTC head required here.

The optional second-opinion path transcribes locally with an independent model
and accepts only guarded corrections. It is optional because it increases
runtime and model storage.

## Policy boundary

The router is intentionally not learned. The trusted-document boundary is the
first decision: no document means no term bias. If a document is present,
duration supplies a cheap conservative safeguard for long-form speech. An
acoustic "headroom" diagnostic was tested but is not a safe acceptance gate.
