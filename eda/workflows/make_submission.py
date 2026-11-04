"""Create a Kaggle submission and retain per-model location predictions."""

import argparse
import ast
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eda.colab.colab_extractor import DEVICE, STRIDE, WINDOW, collate, decode, normalize, windows
from eda.colab.eda_comparison import BiLSTM
from eda.core.analysis import DATA, OUTPUT


CHECKPOINTS = tuple(f"fold{fold}_mixed_lstm_word" for fold in range(3))
MODEL_DIR = OUTPUT / "eda_comparison" / "checkpoints"
SUBMISSIONS = OUTPUT.parent / "submissions"
SANITY_REPORT = OUTPUT / "submission_sanity_labeling_sample.json"


def load_checkpoint(name: str) -> tuple[BiLSTM, Tokenizer]:
    if name not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint {name!r}; choose from {', '.join(CHECKPOINTS)}")
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / f"{name}.json"))
    state = torch.load(MODEL_DIR / f"{name}.pt", map_location="cpu", weights_only=True)
    model = BiLSTM(tokenizer.get_vocab_size())
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model, tokenizer


def predict_with_confidence(model, tokenizer, rows):
    """Use the comparison's window/decode rules and record each decoded span's tag confidence."""
    model.eval()
    predictions = {}
    with torch.no_grad():
        for row in rows:
            items, _ = windows(row, tokenizer, False)
            spans = {}
            for start in range(0, len(items), 32):
                batch = items[start:start + 32]
                x, _, lengths = collate(batch)
                probabilities = model(x, lengths).softmax(-1)
                tags = probabilities.argmax(-1)
                confidence = probabilities.gather(-1, tags.unsqueeze(-1)).squeeze(-1)
                for item, labels, scores in zip(batch, tags.cpu().tolist(),
                                                confidence.cpu().tolist()):
                    ids, _, offsets, _, left, right = item
                    labels, scores = labels[:len(ids)], scores[:len(ids)]
                    decoded = decode(labels, offsets, left, right)
                    segments = []
                    segment = None
                    for label, (a, b), probability in zip(labels, offsets, scores):
                        if label == 0:
                            if segment is not None:
                                segments.append(segment)
                                segment = None
                        elif label == 1 or segment is None:
                            if segment is not None:
                                segments.append(segment)
                            segment = [a, b, [probability]]
                        else:
                            segment[1] = b
                            segment[2].append(probability)
                    if segment is not None:
                        segments.append(segment)
                    kept = [(a, b, sum(values) / len(values)) for a, b, values in segments
                            if left <= (a + b) / 2 < right]
                    assert [(a, b) for a, b, _ in kept] == decoded
                    for a, b, mean_prob in kept:
                        spans.setdefault((a, b), mean_prob)
            predictions[(row["source"], row["row_id"])] = [
                {"start": a, "end": b, "text": row["text"][a:b], "mean_prob": probability}
                for (a, b), probability in sorted(spans.items())
            ]
    return predictions


def locations_for(text, predictions, min_votes):
    votes = Counter()
    for spans in predictions.values():
        votes.update({span["text"] for span in spans
                      if span["text"].strip() and span["text"] in text})
    return sorted((location for location, count in votes.items() if count >= min_votes),
                  key=lambda location: (text.index(location), location))


def make_rows(table, source):
    return [dict(source=source, row_id=str(index) if source == "sample" else row["text_id"],
                 doc_id=None, text=row["text"], markers=[])
            for index, row in enumerate(table)]


def sample_metrics(sample, predictions, min_votes):
    tp = fp = fn = 0
    for row, by_model in zip(sample, predictions):
        actual = {normalize(value) for value in ast.literal_eval(row["locations"])}
        predicted = {normalize(value) for value in
                     locations_for(row["text"], by_model, min_votes)}
        tp += len(actual & predicted)
        fp += len(predicted - actual)
        fn += len(actual - predicted)
    return {
        "precision": tp / (tp + fp) if tp + fp else int(fn == 0),
        "recall": tp / (tp + fn) if tp + fn else int(fp == 0),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate_submission(path, test, example_header):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != example_header:
            raise ValueError(f"Unexpected submission header: {reader.fieldnames}")
        saved = list(reader)
    if len(saved) != 477 or len(saved) != len(test):
        raise ValueError(f"Expected 477 rows; found {len(saved)}")
    if [row["text_id"] for row in saved] != [row["text_id"] for row in test]:
        raise ValueError("Submission text IDs differ from test.csv")
    for saved_row, test_row in zip(saved, test):
        if None in saved_row:
            raise ValueError(f"Extra CSV columns at text_id {test_row['text_id']}")
        locations = json.loads(saved_row["locations"])
        if (not isinstance(locations, list)
                or any(not isinstance(value, str) or value not in test_row["text"]
                       for value in locations)):
            raise ValueError(f"Invalid locations at text_id {test_row['text_id']}")
    return saved


def make_submission(checkpoints=CHECKPOINTS, output=None):
    checkpoints = tuple(checkpoints)
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Specify at least one checkpoint, without duplicates")
    if len(checkpoints) > 1 and set(checkpoints) != set(CHECKPOINTS):
        raise ValueError("Use one checkpoint or all three fold checkpoints")
    min_votes = 2 if len(checkpoints) == 3 else 1
    destination = Path(output) if output else SUBMISSIONS / (
        "submission_bilstm_word_vote3.csv" if len(checkpoints) == 3
        else f"submission_{checkpoints[0]}.csv")

    with (DATA / "test.csv").open(newline="", encoding="utf-8") as file:
        test = list(csv.DictReader(file))
    with (DATA / "submission_example.csv").open(newline="", encoding="utf-8") as file:
        example = csv.DictReader(file)
        example_header = example.fieldnames
        example_ids = [row["text_id"] for row in example]
    if (len(test) != 477 or len({row["text_id"] for row in test}) != len(test)
            or [row["text_id"] for row in test[:len(example_ids)]] != example_ids):
        raise ValueError("Unexpected test.csv IDs or row count")

    with (DATA / "labeling_sample.csv").open(newline="", encoding="utf-8") as file:
        sample = list(csv.DictReader(file))
    if len(sample) != 26:
        raise ValueError(f"Expected 26 qualitative sample rows; found {len(sample)}")
    test_rows, sample_rows = make_rows(test, "test"), make_rows(sample, "sample")
    test_predictions = [dict() for _ in test]
    sample_predictions = [dict() for _ in sample]
    for name in CHECKPOINTS:
        if name not in checkpoints and len(checkpoints) != 1:
            continue
        model, tokenizer = load_checkpoint(name)
        rows = (test_rows if name in checkpoints else []) + sample_rows
        predicted = predict_with_confidence(model, tokenizer, rows)
        for row, by_model in zip(test_rows, test_predictions):
            if name in checkpoints:
                by_model[name] = predicted[(row["source"], row["row_id"])]
        for row, by_model in zip(sample_rows, sample_predictions):
            by_model[name] = predicted[(row["source"], row["row_id"])]
        del model
        print(f"Predicted with {name}", flush=True)

    report = {
        "description": "26-row qualitative sample; not a validation or leaderboard score",
        "rows": len(sample),
        "ensemble": sample_metrics(sample, sample_predictions, 2),
        "per_model": {
            name: sample_metrics(sample, [{name: row[name]} for row in sample_predictions], 1)
            for name in CHECKPOINTS
        },
    }
    SANITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SANITY_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"26-row qualitative sample — ensemble F1: {report['ensemble']['f1']:.4f}; "
          + ", ".join(f"{name}: {metric['f1']:.4f}"
                      for name, metric in report["per_model"].items()), flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = destination.with_name(destination.stem + "_raw.jsonl.gz")
    metadata = destination.with_name(destination.stem + "_raw_meta.json")
    metadata.write_text(json.dumps({
        "checkpoints": [
            {"name": name, "sha256": sha256(MODEL_DIR / f"{name}.pt"),
             "tokenizer_sha256": sha256(MODEL_DIR / f"{name}.json")}
            for name in checkpoints
        ],
        "vote_rule": {"min_votes": min_votes, "unit": "exact surface string"},
        "inference": {"window": WINDOW, "stride": STRIDE, "batch_size": 32,
                      "decoding": "BIO argmax; span midpoint in window core"},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with destination.open("w", newline="", encoding="utf-8") as file, \
            gzip.open(raw, "wt", encoding="utf-8") as raw_file:
        writer = csv.writer(file)
        writer.writerow(example_header)
        for row, by_model in zip(test, test_predictions):
            locations = locations_for(row["text"], by_model, min_votes)
            writer.writerow([row["text_id"], json.dumps(locations, ensure_ascii=False)])
            raw_file.write(json.dumps({"text_id": row["text_id"], "text": row["text"],
                                       "models": by_model,
                                       "locations": locations}, ensure_ascii=False) + "\n")
    saved = validate_submission(destination, test, example_header)
    print(f"Validated {len(saved)} CSV rows; {sum(json.loads(row['locations']) != [] for row in saved)} "
          f"texts with locations; {sum(len(json.loads(row['locations'])) for row in saved)} "
          "locations; every JSON cell parses and every value is a text substring", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", choices=CHECKPOINTS,
                        help="Use one fold model instead of the default three-model vote")
    parser.add_argument("--output", type=Path, help="Override the submission CSV path")
    args = parser.parse_args()
    print(make_submission(args.checkpoint or CHECKPOINTS, args.output))
