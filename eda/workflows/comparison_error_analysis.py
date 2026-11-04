"""Slice common-row predictions and select paired examples for manual review."""

import gzip
import hashlib
import json
from collections import defaultdict

from eda.colab.colab_extractor import normalize
from eda.core.analysis import OUTPUT
from eda.workflows.summarize_comparison import BASE, predictions, rates, tally


def length_bin(n):
    if n < 50:
        return '<50'
    if n < 100:
        return '50-99'
    if n < 200:
        return '100-199'
    return '200+'


def count_bin(n):
    if n == 0:
        return '0'
    if n == 1:
        return '1'
    return '2+'


def row_f1(row):
    return rates(tally(row))['f1']


def read_rows():
    with gzip.open(OUTPUT / 'extractor_bundle_v1.jsonl.gz', 'rt', encoding='utf-8') as source:
        return {(row['source'], row['row_id']): row for line in source
                if (row := json.loads(line))['partition'] == 'cv'}


def names(rows):
    return {normalize(row['text'][a:b]) for row in rows for a, b in row['markers']}


def run():
    report = json.loads((BASE / 'results.json').read_text())
    selection = json.loads((BASE / 'selection.json').read_text())
    if len(report['runs']) != 15:
        raise RuntimeError('Comparison predictions are incomplete')
    rows = read_rows()
    by_id = {f"{row['source']}:{row['row_id']}": row for row in rows.values()}
    counts = defaultdict(lambda: {'rows': 0, 'tp': 0, 'fp': 0, 'fn': 0})
    exposure = defaultdict(lambda: {'gold': 0, 'recovered': 0})
    loaded = {}
    for fold_info in selection['folds']:
        fold = fold_info['fold']
        available = {source: names(row for row in rows.values()
                                   if row['fold'] != fold and row['source'] == source)
                     for source in ('uk', 'ru')}
        for model in report['runs']:
            if model['fold'] != fold:
                continue
            name_key = model['name']
            label = f"{model['mixture']}_{model['architecture']}_{model['tokenizer']}"
            ids = fold_info['mixed_ids' if model['mixture'] == 'mixed' else 'uk_ids']
            supervised = {source: names(by_id[key] for key in ids if key.startswith(source + ':'))
                          for source in ('uk', 'ru')}
            records = predictions(name_key)
            loaded[name_key] = records
            for key, item in records.items():
                row = rows[key]
                gold_count = len(item['actual_strings'])
                result = tally(item)
                for kind, value in [('length', length_bin(len(row['text']))),
                                    ('locations', count_bin(gold_count)), ('all', 'all')]:
                    counter = counts[(label, item['source'], kind, value)]
                    counter['rows'] += 1
                    counter['tp'] += int(result[0])
                    counter['fp'] += int(result[1])
                    counter['fn'] += int(result[2])
                predicted = set(item['predicted_strings'])
                for gold in item['actual_strings']:
                    bucket = ('supervised' if gold in supervised[item['source']]
                              else 'available_unsampled' if gold in available[item['source']]
                              else 'absent_from_fold_training')
                    counter = exposure[(label, item['source'], bucket)]
                    counter['gold'] += 1
                    counter['recovered'] += gold in predicted
    slices = []
    for (label, source, kind, value), counter in sorted(counts.items()):
        score = rates((counter['tp'], counter['fp'], counter['fn']))
        slices.append({'candidate': label, 'source': source, 'slice_kind': kind,
                       'slice': value, 'rows': counter['rows'], **score,
                       'false_positives_per_1000_rows': 1000 * counter['fp'] / counter['rows']})
    recalls = []
    for (label, source, bucket), counter in sorted(exposure.items()):
        recalls.append({'candidate': label, 'source': source, 'exposure': bucket,
                        **counter, 'recall': counter['recovered'] / counter['gold'] if counter['gold'] else None})
    cases = []
    pairs = [('mixed_gru_word', 'mixed_gru_bpe'),
             ('mixed_lstm_word', 'mixed_gru_word'),
             ('mixed_gru_word', 'uk_gru_word')]
    lookup = defaultdict(dict)
    for model in report['runs']:
        label = f"{model['mixture']}_{model['architecture']}_{model['tokenizer']}"
        lookup[label].update(loaded[model['name']])
    for left, right in pairs:
        assert lookup[left].keys() == lookup[right].keys()
        for source in ('uk', 'ru'):
            buckets = {'left_better': [], 'right_better': [], 'shared_failure': []}
            for key, a in lookup[left].items():
                if key[0] != source:
                    continue
                b = lookup[right][key]
                difference = row_f1(a) - row_f1(b)
                tie = hashlib.md5(f'{key[0]}:{key[1]}'.encode()).hexdigest()
                if difference > 0:
                    buckets['left_better'].append((-difference, tie, key, difference))
                elif difference < 0:
                    buckets['right_better'].append((difference, tie, key, difference))
                elif set(a['actual_strings']) - set(a['predicted_strings']) and set(b['actual_strings']) - set(b['predicted_strings']):
                    buckets['shared_failure'].append((0, tie, key, difference))
            for category, ranked in buckets.items():
                for _, _, key, difference in sorted(ranked)[:6]:
                    row = rows[key]
                    cases.append({'comparison': f'{left}_minus_{right}', 'category': category,
                                  'source': source, 'row_id': key[1], 'fold': row['fold'],
                                  'difference_in_row_f1': difference, 'text': row['text'],
                                  'gold': lookup[left][key]['actual_strings'],
                                  'left': lookup[left][key]['predicted_strings'],
                                  'right': lookup[right][key]['predicted_strings'],
                                  'gold_raw': [row['text'][a:b] for a, b in lookup[left][key]['actual_spans']],
                                  'left_raw': [row['text'][a:b] for a, b in lookup[left][key]['predicted_spans']],
                                  'right_raw': [row['text'][a:b] for a, b in lookup[right][key]['predicted_spans']]})
    result = {'slices': slices, 'exposure': recalls,
              'exposure_definition': 'Gold normalized string seen in selected labeled rows of the same source, only in the remaining fold training pool, or neither.',
              'examples': cases,
              'caveat': 'Rows in examples are selected for disagreement, not sampled to estimate error rates.'}
    (BASE / 'error_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    run()
