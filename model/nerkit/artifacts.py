"""Save and load per-token logits so that anyone can ensemble without rerunning models.

One file per (run, split): artifacts/<run key>/<split>.npz

  row_ids   text identifiers, for example "uk:123" or "test:45"
  lengths   number of tokens in each text
  offsets   int32 (total tokens, 2), character start and end in the original text
  logits    float16 (total tokens, 3), raw scores in the order O, B, I
  meta      JSON: tag order, run key, split, fold, seed, tokenizer, and CRF scores if any

Overlapping windows are already averaged, so each text has exactly one row per token.
"""

import csv
import json
from collections import Counter

import numpy as np

from .config import ARTIFACTS_DIR, TAGS
from .decoding import decode_row, reproject, softmax


def row_key(row):
    return f"{row['source']}:{row['row_id']}"


def save_logits(key, split, rows, encoded, logits_by_item, meta):
    """Write one split; `logits_by_item` lines up with `encoded`."""
    folder = ARTIFACTS_DIR / key
    folder.mkdir(parents=True, exist_ok=True)
    lengths = [len(values) for values in logits_by_item]
    empty_offsets = np.zeros((0, 2), dtype=np.int32)
    empty_logits = np.zeros((0, 3), dtype=np.float16)
    np.savez_compressed(
        folder / f"{split}.npz",
        row_ids=np.array([row_key(rows[enc.row_index]) for enc in encoded]),
        lengths=np.array(lengths, dtype=np.int32),
        offsets=np.concatenate([enc.offsets for enc in encoded] or [empty_offsets]),
        logits=np.concatenate(logits_by_item or [empty_logits]).astype(np.float16),
        meta=json.dumps(dict(tags=TAGS, key=key, split=split, **meta)),
    )


def load_logits(key, split):
    """Return a dict from row id to (offsets, logits), plus the metadata."""
    saved = np.load(ARTIFACTS_DIR / key / f"{split}.npz", allow_pickle=False)
    meta = json.loads(str(saved["meta"]))
    if meta.get("crf"):
        meta["crf"] = {name: np.array(value) for name, value in meta["crf"].items()}
    boundaries = np.cumsum(saved["lengths"])[:-1]
    offsets = np.split(saved["offsets"], boundaries)
    logits = np.split(saved["logits"].astype(np.float32), boundaries)
    return {row: (o, l) for row, o, l in zip(saved["row_ids"].tolist(), offsets, logits)}, meta


def average_models(models, texts):
    """Average probabilities of several loaded models, on the first model's tokens.

    Models with the same tokenization are averaged directly; others are reprojected
    through characters first. Returns a dict in the same shape as load_logits.
    """
    combined = {}
    for row, (offsets, _) in models[0].items():
        total = np.zeros((len(offsets), 3), dtype=np.float32)
        for model in models:
            other_offsets, other_logits = model[row]
            if not np.array_equal(other_offsets, offsets):
                other_logits = reproject(other_logits, other_offsets, offsets, len(texts[row]))
            total += softmax(other_logits)
        combined[row] = (offsets, np.log(np.clip(total / len(models), 1e-8, 1.0)))
    return combined


def combine_models(models, metas, texts):
    """Average several loaded models on the first model's tokens; returns (combined, crf).

    Without CRFs this averages probabilities. CRF emission scores are not probabilities, so
    log-probabilities and transition scores are averaged instead, and Viterbi decodes the result.
    """
    crfs = [meta.get("crf") for meta in metas]
    if not any(crfs):
        return average_models(models, texts), None
    if not all(crfs):
        raise ValueError("Cannot combine CRF and non-CRF models")
    combined = {}
    for row, (offsets, _) in models[0].items():
        total = np.zeros((len(offsets), 3), dtype=np.float32)
        for model in models:
            other_offsets, other_logits = model[row]
            if np.array_equal(other_offsets, offsets):
                total += np.log(np.clip(softmax(other_logits), 1e-8, 1.0))
            else:
                total += reproject(other_logits, other_offsets, offsets, len(texts[row]))
        combined[row] = (offsets, total / len(models))
    averaged = {name: np.mean([crf[name] for crf in crfs], axis=0) for name in crfs[0]}
    return combined, averaged


def decode_split(loaded, rows, crf=None, **options):
    """Decode a loaded split for `rows`, in the order of `rows`. Pass the CRF scores for CRF models."""
    return [decode_row(*_row_arrays(loaded, row), row["text"], crf=crf, **options) for row in rows]


def _row_arrays(loaded, row):
    offsets, logits = loaded[row_key(row)]
    return logits, offsets


def majority_vote(predictions_by_model, min_votes):
    """Keep a string when at least `min_votes` models predict it (per text)."""
    voted = []
    for lists in zip(*predictions_by_model):
        counts = Counter(value for values in lists for value in set(values))
        voted.append([value for value, count in counts.items() if count >= min_votes])
    return voted


def write_submission(path, test_rows, predictions):
    """Write and re-read a Kaggle CSV: JSON lists of exact substrings, in test order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text_id", "locations"])
        for row, locations in zip(test_rows, predictions):
            writer.writerow([row["row_id"], json.dumps(locations, ensure_ascii=False)])
    with path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    if [row["text_id"] for row in saved] != [row["row_id"] for row in test_rows]:
        raise ValueError("Submission ids differ from test.csv")
    for row, test_row in zip(saved, test_rows):
        if any(value not in test_row["text"] for value in json.loads(row["locations"])):
            raise ValueError(f"A prediction is not a substring of text {row['text_id']}")
    return path
