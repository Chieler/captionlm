# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""F-score for context-biasing term recognition.

Adapted from NVIDIA NeMo's context_biasing_utils.compute_fscore
(Apache-2.0), source:
nemo/collections/asr/parts/context_biasing/context_biasing_utils.py

Changes from the original: takes an in-memory list of {"text", "pred_text"}
dicts instead of a NeMo manifest file path (drops the
nemo.collections.asr.parts.utils.manifest_utils dependency), uses
kaldialign directly instead of nemo.utils.logging for stats, and drops the
per-sample-keyword-field mode (this project always scores against one
global term list). The core alignment-based scoring algorithm is
unchanged.
"""
import re

from kaldialign import align


def compute_fscore(
    samples: list[dict],
    key_words_list: list[str],
    eps: str = "<eps>",
) -> tuple[float, float, float]:
    assert samples, "samples is empty"
    assert key_words_list, "key_words_list is empty"

    all_key_words_list = [kw.lower() for kw in key_words_list]
    keywords_set = set(all_key_words_list)
    max_ngram_order = max(len(kw.split()) for kw in all_key_words_list)
    key_words_stat = {kw: [0, 0, 0] for kw in all_key_words_list}  # tp, gt, fp

    for sample in samples:
        ref = re.sub(r"[.,!?]", "", sample["text"].lower()).split()
        hyp = re.sub(r"[.,!?]", "", sample["pred_text"].lower()).split()
        ali = align(ref, hyp, eps)

        for word_ref, word_hyp in ali:
            if word_ref in keywords_set:
                key_words_stat[word_ref][1] += 1
                if word_ref == word_hyp:
                    key_words_stat[word_ref][0] += 1
            elif word_hyp in keywords_set:
                key_words_stat[word_hyp][2] += 1

        for ngram_order in range(2, max_ngram_order + 1):
            idx = 0
            item_ref: list[tuple[str, int]] = []
            while idx < len(ali):
                if item_ref:
                    item_ref = [item_ref[1]]
                    idx = item_ref[0][1] + 1
                while len(item_ref) != ngram_order and idx < len(ali):
                    word = ali[idx][0]
                    idx += 1
                    if word == eps:
                        continue
                    item_ref.append((word, idx - 1))
                if len(item_ref) == ngram_order:
                    phrase_ref = " ".join(wr[0] for wr in item_ref)
                    phrase_hyp = " ".join(ali[wr[1]][1] for wr in item_ref)
                    if phrase_ref in keywords_set:
                        key_words_stat[phrase_ref][1] += 1
                        if phrase_ref == phrase_hyp:
                            key_words_stat[phrase_ref][0] += 1

            idx = 0
            item_hyp: list[tuple[str, int]] = []
            while idx < len(ali):
                if item_hyp:
                    item_hyp = [item_hyp[1]]
                    idx = item_hyp[0][1] + 1
                while len(item_hyp) != ngram_order and idx < len(ali):
                    word = ali[idx][1]
                    idx += 1
                    if word == eps:
                        continue
                    item_hyp.append((word, idx - 1))
                if len(item_hyp) == ngram_order:
                    phrase_hyp = " ".join(wh[0] for wh in item_hyp)
                    phrase_ref = " ".join(ali[wh[1]][0] for wh in item_hyp)
                    if phrase_hyp in keywords_set and phrase_hyp != phrase_ref:
                        key_words_stat[phrase_hyp][2] += 1

    tp = sum(v[0] for v in key_words_stat.values())
    gt = sum(v[1] for v in key_words_stat.values())
    fp = sum(v[2] for v in key_words_stat.values())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (gt + 1e-8)
    fscore = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, fscore
