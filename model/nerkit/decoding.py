"""Turn saved logits into location strings: thresholds, merge rules and clean-up."""

import re

import numpy as np

from eda.core.analysis import published_normalize

TRAILING_NUMBER = re.compile(r"[,\s]+\d[\d/\-\w]*$")


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def viterbi(logits, crf):
    """The best tag sequence under CRF transition scores."""
    if len(logits) == 0:
        return np.zeros(0, dtype=np.int64)
    score = crf["start"] + logits[0]
    back = []
    for step in logits[1:]:
        candidates = score[:, None] + crf["transitions"]
        back.append(candidates.argmax(axis=0))
        score = candidates.max(axis=0) + step
    tags = [int((score + crf["end"]).argmax())]
    for pointers in reversed(back):
        tags.append(int(pointers[tags[-1]]))
    return np.array(tags[::-1], dtype=np.int64)


def predict_tags(logits, threshold=None, crf=None, bias=0.0):
    """Tag each token.

    Plain models: with a threshold, a token is a location when 1 - P(O) reaches it.
    CRF models: Viterbi decoding, where `bias` is added to the B and I scores first. A positive
    bias predicts more locations (more recall); a negative one predicts fewer.
    """
    if crf is not None:
        if bias:
            logits = logits.copy()
            logits[:, 1:] += bias
        return viterbi(logits, crf)
    if threshold is None:
        return logits.argmax(axis=-1)
    probabilities = softmax(logits)
    tags = probabilities[:, 1:].argmax(axis=-1) + 1
    tags[1 - probabilities[:, 0] < threshold] = 0
    return tags


def tags_to_spans(tags, offsets, merge="lenient", text=""):
    """Merge token tags into character spans.

    lenient   I without a preceding B starts a location (Herman's original rule)
    strict    I without a preceding B is dropped
    join      like lenient, and locations separated only by a hyphen or apostrophe are joined
    """
    spans = []
    current = None
    for tag, (start, end) in zip(tags, offsets):
        if tag == 0:
            if current:
                spans.append(current)
            current = None
        elif tag == 1 or current is None:
            if current:
                spans.append(current)
            current = None if (tag == 2 and merge == "strict") else [int(start), int(end)]
        else:
            current[1] = int(end)
    if current:
        spans.append(current)
    if merge == "join":
        joined = []
        for span in spans:
            if joined and text[joined[-1][1]:span[0]].strip() in ("-", "–", "'", "’", ""):
                joined[-1][1] = span[1]
            else:
                joined.append(span)
        spans = joined
    return [tuple(span) for span in spans]


def clean_strings(strings, drop_empty=True, trim_number=False):
    """Drop strings the metric reduces to nothing; optionally remove a trailing house number."""
    result = []
    for value in strings:
        value = value.strip()
        if trim_number:
            value = TRAILING_NUMBER.sub("", value)
        if value and not (drop_empty and not published_normalize(value)):
            result.append(value)
    return list(dict.fromkeys(result))


def decode_row(logits, offsets, text, threshold=None, merge="lenient", crf=None,
               drop_empty=True, trim_number=False, bias=0.0):
    """The location strings for one text."""
    tags = predict_tags(logits, threshold, crf, bias)
    spans = tags_to_spans(tags, offsets, merge, text)
    return clean_strings([text[a:b] for a, b in spans], drop_empty, trim_number)


def reproject(logits, offsets, target_offsets, text_length):
    """Convert one model's logits to another tokenization (for word + BPE ensembles).

    P(O) is averaged over the characters of each target token. The share between B
    and I comes from the first source token overlapping that target token.
    """
    probabilities = softmax(logits)
    owner = np.full(text_length, -1, dtype=np.int64)
    for index, (start, end) in enumerate(offsets):
        owner[start:end] = index
    result = np.zeros((len(target_offsets), 3), dtype=np.float32)
    result[:, 0] = 1.0
    for index, (start, end) in enumerate(target_offsets):
        pieces = owner[start:end]
        pieces = pieces[pieces >= 0]
        if len(pieces) == 0:
            continue
        outside = float(probabilities[pieces, 0].mean())
        first = probabilities[pieces[0]]
        share_b = first[1] / max(first[1] + first[2], 1e-8)
        result[index] = (outside, (1 - outside) * share_b, (1 - outside) * (1 - share_b))
    return np.log(np.clip(result, 1e-8, 1.0))
