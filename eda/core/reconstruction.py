"""Audit recoverability of raw spans without changing the source annotations."""

from collections import Counter
import csv
import hashlib
import json

import duckdb
import pyarrow.parquet as pq

from eda.core.analysis import DATA, OUTPUT, parse_markers, token_offsets, published_normalize


def reconstruct(offsets, markers, bio):
    """Project by token overlap; BIO restarts when the owning gold span changes.

    Overlapping spans cannot be represented by one tag per token. The first
    sorted span owns a shared token; those cases are counted separately.
    """
    owners = [next((i for i, (a, b) in enumerate(markers) if s < b and a < e), None)
              for s, e in offsets]
    result = []
    previous = None
    for (start, end), owner in zip(offsets, owners):
        if owner is not None:
            if previous is None or (bio and owner != previous):
                result.append((start, end))
            else:
                result[-1] = (result[-1][0], end)
        previous = owner
    return result


def add_score(counter, actual, predicted):
    a, p = set(actual), set(predicted)
    counter.update(tp=len(a & p), fp=len(p - a), fn=len(a - p))


def finish_score(c):
    tp, fp, fn = (c[k] for k in ('tp', 'fp', 'fn'))
    return dict(c, precision=tp / (tp + fp) if tp + fp else 1,
                recall=tp / (tp + fn) if tp + fn else 1,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1)


def run():
    db = duckdb.connect()
    roles = dict(db.execute("SELECT row_id, split FROM read_parquet(?) WHERE source='uk'",
                            [str(OUTPUT / 'split_manifest.parquet')]).fetchall())
    counts, scores, failures = Counter(), {}, Counter()
    examples = []
    # Only training rows: reserved final evaluation is not inspected here.
    with (DATA / 'uk_geo_dataset.csv').open(newline='') as raw:
        reader = csv.DictReader(raw)
        row_id = -1
        for batch in pq.ParquetFile(DATA / 'uk_geo_dataset_processed_v1.parquet').iter_batches(2048):
            for record in batch.to_pylist():
                row_id += 1
                row = next(reader)
                if roles[row_id] != 'train':
                    continue
                counts['rows'] += 1
                text = row['text']
                markers = sorted(set(parse_markers(row['loc_markers'])))
                offsets = token_offsets(text, json.loads(record['tokens']))
                if offsets is None:
                    counts['unalignable_rows'] += 1
                    continue
                labels = json.loads(record['labels'])
                starts, ends = {s for s, e in offsets}, {e for s, e in offsets}
                unaligned = [(a, b) for a, b in markers if a not in starts or b not in ends]
                disagreements = [(s, e) for (s, e), label in zip(offsets, labels)
                                 if any(s < b and a < e for a, b in markers) != (label != 'O')]
                linked = sum(any(s < b and a < e for a, b in unaligned) for s, e in disagreements)
                counts.update(spans=len(markers), unaligned_spans=len(unaligned),
                              disagreeing_tokens=len(disagreements), linked_disagreeing_tokens=linked,
                              unaligned_rows=bool(unaligned), disagreement_rows=bool(disagreements),
                              both_rows=bool(unaligned and disagreements))
                overlap = any(b > c for (_, b), (c, _) in zip(markers, markers[1:]))
                counts['overlapping_span_rows'] += overlap
                for mode in ('flat', 'bio'):
                    prediction = reconstruct(offsets, markers, mode == 'bio')
                    add_score(scores.setdefault(mode + '_span', Counter()), markers, prediction)
                    add_score(scores.setdefault(mode + '_string', Counter()),
                              [published_normalize(text[a:b]) for a,b in markers],
                              [published_normalize(text[a:b]) for a,b in prediction])
                    if set(prediction) != set(markers):
                        reason = ('overlapping_spans' if overlap else 'token_boundary' if unaligned
                                  else 'adjacent_entities' if mode == 'flat' else 'other')
                        failures[mode + ':' + reason] += 1
                if disagreements or unaligned:
                    examples.append(dict(source='uk', row_id=row_id, text=text, markers=markers,
                                         unaligned=unaligned, disagreements=disagreements))
    result = dict(scope='Ukrainian retained training rows; supplied token offsets',
                  counts=dict(counts), failures=dict(failures),
                  scores={k: finish_score(v) for k,v in scores.items()})
    result['input_sha256'] = {}
    for filename in ('uk_geo_dataset.csv', 'uk_geo_dataset_processed_v1.parquet'):
        with (DATA / filename).open('rb') as f:
            result['input_sha256'][filename] = hashlib.file_digest(f, 'sha256').hexdigest()
    (OUTPUT / 'reconstruction.json').write_text(json.dumps(result, indent=2))
    (OUTPUT / 'boundary_cases.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    exclusions = [dict(source='uk', row_id=r['row_id'], reason='unrepresentable_supplied_token_boundary')
                  for r in examples if r['unaligned']]
    (OUTPUT / 'pilot_exclusions_supplied_tokens_v1.json').write_text(json.dumps(dict(
        scope='Pilot training with supplied Parquet tokens only; recompute for other tokenizers',
        policy='Original labels and role membership unchanged; development is not filtered',
        rows=exclusions), indent=2))
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
