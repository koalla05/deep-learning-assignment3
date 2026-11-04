"""Reproducible, streaming checks for the location extraction notebooks."""

from __future__ import annotations

import ast
import csv
import hashlib
import heapq
import json
import re
import string
from collections import Counter
from pathlib import Path

import duckdb


DATA = Path(__file__).resolve().parents[2] / "data" / "kse-ua-location-extraction-2026"
OUTPUT = Path(__file__).resolve().parents[1] / "output"
SEED = 3012026
CYRILLIC_UK = re.compile(r"[іїєґІЇЄҐ]")
CYRILLIC_RU = re.compile(r"[ыэъёЫЭЪЁ]")
URL = re.compile(r"https?://|www\.", re.I)
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)
LENGTH_BINS = (0, 20, 50, 100, 200, 400, 800, 1600, 3200)


def parse_markers(raw: str) -> list[tuple[int, int]]:
    """Parse only literal lists of integer offset pairs."""
    if raw == "[]":
        return []
    value = ast.literal_eval(raw)
    if not isinstance(value, list):
        raise ValueError("marker field is not a list")
    pairs = []
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("marker is not a pair")
        start, end = pair
        if type(start) is not int or type(end) is not int:
            raise ValueError("marker endpoints are not integers")
        pairs.append((start, end))
    return pairs


def text_features(text: str) -> dict[str, int]:
    return {
        "characters": len(text),
        "words": len(WORD.findall(text)),
        "newlines": text.count("\n"),
        "urls": int(bool(URL.search(text))),
        "emoji": int(bool(EMOJI.search(text))),
        "uk_chars": len(CYRILLIC_UK.findall(text)),
        "ru_chars": len(CYRILLIC_RU.findall(text)),
        "digits": sum(c.isdigit() for c in text),
    }


def _bin(value: int) -> str:
    for bound in LENGTH_BINS:
        if value <= bound:
            return str(bound)
    return ">3200"


def _sample_key(source: str, row_id: int) -> int:
    material = f"{SEED}:{source}:{row_id}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")


def audit_csv(source: str, sample_size: int) -> dict:
    """Full marker audit and a deterministic bounded row sample."""
    path = DATA / f"{source}_geo_dataset.csv"
    counts: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    mention_lengths: Counter[str] = Counter()
    errors: list[dict] = []
    sample: list[tuple[int, dict]] = []
    csv.field_size_limit(10_000_000)
    with path.open(newline="", encoding="utf-8") as handle:
        for row_id, row in enumerate(csv.DictReader(handle)):
            counts["rows"] += 1
            text = row.get("text") or ""
            if not text:
                counts["empty_text"] += 1
            lengths[_bin(len(text))] += 1
            counts["has_newlines"] += "\n" in text
            counts["has_urls"] += bool(URL.search(text))
            counts["has_emoji"] += bool(EMOJI.search(text))
            counts["has_uk_chars"] += bool(CYRILLIC_UK.search(text))
            counts["has_ru_chars"] += bool(CYRILLIC_RU.search(text))
            if source == "uk":
                counts[f"valid_{row['is_valid']}"] += 1
            else:
                counts["missing_doc_id"] += not row["doc_id"].strip()
            markers = {}
            for kind in ("loc", "org", "per"):
                try:
                    pairs = parse_markers(row[f"{kind}_markers"])
                except (ValueError, SyntaxError, TypeError, MemoryError):
                    counts[f"{kind}_parse_error"] += 1
                    pairs = []
                    if len(errors) < 12:
                        errors.append({"source": source, "row_id": row_id, "kind": kind, "reason": "parse"})
                markers[kind] = pairs
                counts[f"{kind}_mentions"] += len(pairs)
                counts[f"{kind}_positive_rows"] += bool(pairs)
                counts[f"{kind}_duplicate_markers"] += len(pairs) - len(set(pairs))
                for start, end in pairs:
                    if start < 0 or end > len(text) or end <= start:
                        counts[f"{kind}_invalid_bounds"] += 1
                        if len(errors) < 12:
                            errors.append({"source": source, "row_id": row_id, "kind": kind, "reason": "bounds", "span": [start, end], "length": len(text)})
                    elif kind == "loc":
                        mention_lengths[_bin(end - start)] += 1
                        counts["loc_with_digit"] += any(c.isdigit() for c in text[start:end])
                        counts["loc_with_newline"] += "\n" in text[start:end]
            loc = sorted(markers["loc"])
            counts["loc_overlap_pairs"] += sum(a[1] > b[0] for a, b in zip(loc, loc[1:]))
            counts["loc_overlaps_org"] += sum(a[0] < b[1] and b[0] < a[1] for a in loc for b in markers["org"])
            counts["loc_overlaps_per"] += sum(a[0] < b[1] and b[0] < a[1] for a in loc for b in markers["per"])
            if not loc:
                counts["no_locations"] += 1
            key = _sample_key(source, row_id)
            if len(sample) < sample_size or key < -sample[0][0]:
                record = {"source": source, "row_id": row_id, "text": text, "loc": loc,
                          "doc_id": row.get("doc_id"), "is_valid": row.get("is_valid"),
                          **text_features(text)}
                if len(sample) < sample_size:
                    heapq.heappush(sample, (-key, record))
                else:
                    heapq.heapreplace(sample, (-key, record))
    return {"counts": dict(counts), "text_length_bins": dict(lengths),
            "mention_length_bins": dict(mention_lengths), "errors": errors,
            "sample": [item[1] for item in sorted(sample, key=lambda item: -item[0])],
            "file_bytes": path.stat().st_size}


def audit_test() -> dict:
    rows = []
    with (DATA / "test.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({"text_id": row["text_id"], "text": row["text"],
                         "locations_placeholder": row["locations"], **text_features(row["text"])})
    return {"rows": rows, "count": len(rows)}


def duplicate_audit() -> dict:
    """Count exact duplicate texts; hash groups are verified by distinct text."""
    connection = duckdb.connect()
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='2GB'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET temp_directory='{(OUTPUT / 'duckdb_tmp').as_posix()}'")
    uk = (DATA / "uk_geo_dataset.csv").as_posix()
    ru = (DATA / "ru_geo_dataset.csv").as_posix()
    connection.execute("""
        CREATE TEMP TABLE records AS
        SELECT md5(text) AS text_key, 'uk' AS source, is_valid AS split,
               md5(loc_markers) AS marker_key
          FROM read_csv(?, all_varchar=true)
        UNION ALL
        SELECT md5(text) AS text_key, 'ru' AS source, 'ru' AS split,
               md5(loc_markers) AS marker_key
          FROM read_csv(?, all_varchar=true)
    """, [uk, ru])
    connection.execute("""
        CREATE TEMP TABLE duplicate_keys AS
        SELECT text_key FROM records GROUP BY text_key HAVING count(*) > 1
    """)
    sql = """
        WITH duplicate_groups AS (
            SELECT r.text_key, count(*) AS n, count(DISTINCT source) AS sources,
                   count(DISTINCT CASE WHEN source='uk' THEN split END) AS uk_splits,
                   count(DISTINCT marker_key) AS annotation_versions,
                   sum(CASE WHEN split='1' THEN 1 ELSE 0 END) AS uk_valid_rows
            FROM records r JOIN duplicate_keys d USING (text_key)
            GROUP BY r.text_key
        )
        SELECT count(*) AS duplicate_texts, sum(n) AS duplicate_rows,
               sum(CASE WHEN sources>1 THEN 1 ELSE 0 END) AS cross_source_texts,
               sum(CASE WHEN uk_splits>1 THEN 1 ELSE 0 END) AS uk_cross_split_texts,
               sum(CASE WHEN annotation_versions>1 THEN 1 ELSE 0 END) AS conflicting_marker_texts,
               sum(uk_valid_rows) AS uk_valid_rows_in_duplicate_groups
        FROM duplicate_groups
    """
    row = connection.execute(sql).fetchone()
    names = [col[0] for col in connection.description]
    # Verify that no two distinct strings share a digest in candidate groups.
    collision_sql = """
        WITH candidate_texts AS (
            SELECT md5(text) AS text_key, text FROM read_csv(?, all_varchar=true)
            UNION ALL
            SELECT md5(text) AS text_key, text FROM read_csv(?, all_varchar=true)
        )
        SELECT count(*) FROM (
            SELECT c.text_key FROM candidate_texts c
            JOIN duplicate_keys d USING (text_key)
            GROUP BY c.text_key HAVING count(DISTINCT text) > 1
        )
    """
    collisions = connection.execute(collision_sql, [uk, ru]).fetchone()[0]
    connection.close()
    return {**dict(zip(names, row)), "hash_collisions_in_duplicate_groups": collisions}


def parquet_audit(n: int | None = None) -> dict:
    """Check row-order correspondence, label structure, and token alignment."""
    import pyarrow.parquet as pq

    path = DATA / "uk_geo_dataset_processed_v1.parquet"
    file = pq.ParquetFile(path)
    if n is None:
        n = file.metadata.num_rows
    counts: Counter[str] = Counter()
    examples = []
    alignment_examples = []
    raw = (DATA / "uk_geo_dataset.csv").open(newline="", encoding="utf-8")
    reader = csv.DictReader(raw)
    try:
        seen = 0
        for batch in file.iter_batches(batch_size=1024):
            for record in batch.to_pylist():
                if seen >= n:
                    break
                source = next(reader)
                tokens = json.loads(record["tokens"])
                labels = json.loads(record["labels"])
                counts["rows"] += 1
                counts["length_mismatch"] += len(tokens) != len(labels)
                counts["flag_mismatch"] += int(source["is_valid"]) != record["is_valid"]
                offsets = token_offsets(source["text"], tokens)
                counts["token_substring_order_mismatch"] += offsets is None
                counts["rows_with_loc_marker"] += source["loc_markers"] != "[]"
                counts["rows_with_location_label"] += any(label != "O" for label in labels)
                counts.update(f"label:{label}" for label in labels)
                if offsets is not None:
                    markers = parse_markers(source["loc_markers"])
                    for (start, end), label in zip(offsets, labels):
                        expected = any(start < b and a < end for a, b in markers)
                        counts["token_location_disagreements"] += expected != (label != "O")
                        if expected != (label != "O") and len(alignment_examples) < 8:
                            alignment_examples.append({"row": seen, "text": source["text"][:220],
                                                       "token": source["text"][start:end], "token_span": [start, end],
                                                       "label": label, "markers": markers})
                    boundaries = {point for a, b in offsets for point in (a, b)}
                    counts["unaligned_location_boundaries"] += sum(a not in boundaries or b not in boundaries for a,b in markers)
                if any(label != "O" for label in labels) and len(examples) < 5:
                    examples.append({"text": source["text"][:200], "markers": source["loc_markers"],
                                     "tokens": tokens[:40], "labels": labels[:40]})
                seen += 1
            if seen >= n:
                break
    finally:
        raw.close()
    return {"total_rows": file.metadata.num_rows, "sample_prefix_rows": n,
            "counts": dict(counts), "examples": examples, "alignment_examples": alignment_examples}


def tokens_in_order(text: str, tokens: list[str]) -> bool:
    return token_offsets(text, tokens) is not None


def token_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]] | None:
    position = 0
    offsets = []
    for token in tokens:
        found = text.find(token, position)
        if found < 0:
            return None
        position = found + len(token)
        offsets.append((found, position))
    return offsets


def published_normalize(text: str) -> str:
    """Literal code from the published Kaggle evaluation page."""
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\b\w\b\s?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def competition_f1(truth: list[list[str]], predictions: list[list[str]]) -> dict:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same number of texts")
    tp = fp = fn = 0
    for actual, predicted in zip(truth, predictions):
        a, p = {published_normalize(x) for x in actual}, {published_normalize(x) for x in predicted}
        tp += len(a & p)
        fp += len(p - a)
        fn += len(a - p)
    precision = tp / (tp + fp) if tp + fp else int(fn == 0)
    recall = tp / (tp + fn) if tp + fn else int(fp == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": round(f1, 8)}


def span_f1(truth: list[list[tuple[int, int]]], predictions: list[list[tuple[int, int]]]) -> dict:
    """Exact character-span micro-F1, complementary to normalized string F1."""
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same number of texts")
    tp = fp = fn = 0
    for actual, predicted in zip(truth, predictions):
        a, p = set(actual), set(predicted)
        tp += len(a & p)
        fp += len(p - a)
        fn += len(a - p)
    precision = tp / (tp + fp) if tp + fp else int(fn == 0)
    recall = tp / (tp + fn) if tp + fn else int(fp == 0)
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": round(score, 8)}


def write_audit() -> Path:
    OUTPUT.mkdir(exist_ok=True)
    sources = {}
    for source, sample_size in (("uk", 4000), ("ru", 6000)):
        checkpoint = OUTPUT / f"{source}_audit.json"
        file_bytes = (DATA / f"{source}_geo_dataset.csv").stat().st_size
        if checkpoint.exists():
            candidate = json.loads(checkpoint.read_text(encoding="utf-8"))
            if candidate["file_bytes"] == file_bytes:
                sources[source] = candidate
                continue
        sources[source] = audit_csv(source, sample_size)
        checkpoint.write_text(json.dumps(sources[source], ensure_ascii=False), encoding="utf-8")
        print(f"Finished {source.upper()} full scan", flush=True)
    report = {"seed": SEED, **sources, "test": audit_test(),
              "parquet": parquet_audit(), "duplicates": duplicate_audit()}
    target = OUTPUT / "audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(write_audit())
