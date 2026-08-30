"""BiasedParakeetTDTCTC: CTC-WS context-biasing on top of parakeet-mlx's
hybrid TDT-CTC model. Runs the CTC head's log-probs through the vendored
word spotter, then merges spotted terms into the TDT hypothesis by
frame-overlap. When .context_graph is None, behaves exactly like the
stock ParakeetTDTCTC.generate()."""
import json

import mlx.core as mx
import numpy as np
from dacite import from_dict
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten
from parakeet_mlx.alignment import sentences_to_result, tokens_to_sentences
from parakeet_mlx.parakeet import DecodingConfig, ParakeetTDTCTC, ParakeetTDTCTCArgs

from captionlm.config import SpotterConfig
from captionlm.merge import merge_spans
from captionlm.vendor.context_graph_ctc import ContextGraphCTC
from captionlm.vendor.ctc_word_spotter import run_word_spotter


class BiasedParakeetTDTCTC(ParakeetTDTCTC):
    def __init__(self, args: ParakeetTDTCTCArgs):
        super().__init__(args)
        self.context_graph: ContextGraphCTC | None = None
        self.spotter_config = SpotterConfig()

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ):
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)

        result, _ = self.decode(features, lengths, config=decoding_config)

        if self.context_graph is None:
            hypotheses = result
        else:
            ctc_logits = self.ctc_decoder(features)
            mx.eval(ctc_logits)
            blank_idx = len(self.vocabulary)

            hypotheses = []
            for batch_idx, tokens in enumerate(result):
                logprobs = np.array(ctc_logits[batch_idx].astype(mx.float32))
                ws_hyps = run_word_spotter(
                    logprobs,
                    self.context_graph,
                    self,
                    blank_idx=blank_idx,
                    beam_threshold=self.spotter_config.beam_threshold,
                    cb_weight=self.spotter_config.cb_weight,
                    ctc_ali_token_weight=self.spotter_config.ctc_ali_token_weight,
                    keyword_threshold=self.spotter_config.keyword_threshold,
                    blank_threshold=self.spotter_config.blank_threshold,
                    non_blank_threshold=self.spotter_config.non_blank_threshold,
                )
                hypotheses.append(
                    merge_spans(
                        tokens,
                        ws_hyps,
                        self.time_ratio,
                        intersection_threshold=self.spotter_config.intersection_threshold,
                    )
                )

        return [
            sentences_to_result(tokens_to_sentences(h, decoding_config.sentence))
            for h in hypotheses
        ]


def load_biased_model(
    hf_id: str, *, dtype: mx.Dtype = mx.bfloat16
) -> BiasedParakeetTDTCTC:
    config = json.load(open(hf_hub_download(hf_id, "config.json"), "r"))
    weight = hf_hub_download(hf_id, "model.safetensors")

    cfg = from_dict(ParakeetTDTCTCArgs, config)
    model = BiasedParakeetTDTCTC(cfg)
    model.eval()
    model.load_weights(weight)

    curr_weights = dict(tree_flatten(model.parameters()))
    curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))

    return model
