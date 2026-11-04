"""Summarize frozen common-row EDA runs and paired grouped uncertainty."""

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eda.core.analysis import OUTPUT

BASE = OUTPUT / 'eda_comparison'
SEED = 3012026


def tally(record, kind='string'):
    if kind == 'string':
        gold = set(record['actual_strings'])
        predicted = set(record['predicted_strings'])
    else:
        gold = {tuple(pair) for pair in record['actual_spans']}
        predicted = {tuple(pair) for pair in record['predicted_spans']}
    return np.array([len(gold & predicted), len(predicted - gold), len(gold - predicted)],
                    dtype=np.int64)


def rates(counts):
    tp, fp, fn = map(int, counts)
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else int(fn == 0),
            'recall': tp / (tp + fn) if tp + fn else int(fp == 0),
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0}


def predictions(name):
    path = BASE / f'{name}_predictions.jsonl.gz'
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        return {(row['source'], row['row_id']): row for line in source
                if (row := json.loads(line))}


def source_summary(records):
    answer = {}
    for source in ('uk', 'ru', 'combined'):
        filtered = [row for row in records.values() if source == 'combined' or row['source'] == source]
        answer[source] = {
            kind: rates(sum((tally(row, kind) for row in filtered), np.zeros(3, dtype=np.int64)))
            for kind in ('string', 'span')}
        answer[source]['rows'] = len(filtered)
    return answer


def paired(left, right, rng, n_bootstrap=500):
    assert left.keys() == right.keys()
    by_source = {}
    for source in ('uk', 'ru', 'combined'):
        groups = defaultdict(lambda: np.zeros((2, 3), dtype=np.int64))
        for key in left:
            a, b = left[key], right[key]
            assert a['group_id'] == b['group_id']
            if source != 'combined' and a['source'] != source:
                continue
            group = (a['source'], a['group_id'])
            groups[group][0] += tally(a)
            groups[group][1] += tally(b)
        keys = list(groups)
        values = np.stack([groups[key] for key in keys])
        observed = rates(values[:, 0].sum(axis=0))['f1'] - rates(values[:, 1].sum(axis=0))['f1']
        partitions = defaultdict(list)
        for i, key in enumerate(keys):
            partitions[key[0]].append(i)
        draws = []
        for _ in range(n_bootstrap):
            sampled = np.concatenate([rng.choice(indices, len(indices), replace=True)
                                      for indices in partitions.values()])
            counts = values[sampled].sum(axis=0)
            draws.append(rates(counts[0])['f1'] - rates(counts[1])['f1'])
        by_source[source] = {'difference': observed,
                             'group_bootstrap_95': np.quantile(draws, [.025, .975]).tolist(),
                             'groups': len(keys)}
    return by_source


def main():
    report = json.loads((BASE / 'results.json').read_text())
    runs = report['runs']
    if len(runs) != 15:
        raise RuntimeError(f'Expected 15 completed runs; found {len(runs)}')
    loaded = {run['name']: predictions(run['name']) for run in runs}
    by_candidate = defaultdict(list)
    for run in runs:
        label = f"{run['mixture']}_{run['architecture']}_{run['tokenizer']}"
        by_candidate[label].append((run, loaded[run['name']]))
    summary = {'protocol': report['protocol'], 'bundle_sha256': report['bundle_sha256'],
               'candidates': {}, 'pairs': {},
               'uncertainty': 'Group bootstrap conditions on fitted models; it does not capture training-seed variation.'}
    for label, entries in by_candidate.items():
        assert len(entries) == 3
        assert {run['fold'] for run, _ in entries} == {0, 1, 2}
        joined = {}
        fold_scores = []
        for run, records in entries:
            fold_scores.append({'fold': run['fold'], 'scores': source_summary(records)})
            joined.update(records)
        assert len(joined) == 63388
        summary['candidates'][label] = {'scores': source_summary(joined),
                                        'folds': sorted(fold_scores, key=lambda x: x['fold']),
                                        'training': [run['training'] for run, _ in entries]}
    rng = np.random.default_rng(SEED)
    comparisons = [('mixed_gru_word', 'mixed_gru_bpe'),
                   ('mixed_lstm_word', 'mixed_lstm_bpe'),
                   ('mixed_lstm_word', 'mixed_gru_word'),
                   ('mixed_lstm_bpe', 'mixed_gru_bpe'),
                   ('mixed_gru_word', 'uk_gru_word')]
    for a, b in comparisons:
        joined = []
        for label in (a, b):
            combined = {}
            for _, records in by_candidate[label]:
                combined.update(records)
            joined.append(combined)
        summary['pairs'][f'{a}_minus_{b}'] = paired(*joined, rng)
    path = BASE / 'summary.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    main()
