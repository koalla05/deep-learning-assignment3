"""Measure fold-specific word/BPE coverage on the frozen CV evaluation rows."""

import gzip
import json
from collections import defaultdict

from eda.colab.colab_extractor import fit_tokenizer, representable
from eda.core.analysis import OUTPUT


def run():
    with gzip.open(OUTPUT / 'extractor_bundle_v1.jsonl.gz', 'rt', encoding='utf-8') as source:
        rows = [row for line in source if (row := json.loads(line))['partition'] == 'cv']
    result = []
    for fold in range(3):
        training = [row for row in rows if row['fold'] != fold]
        evaluation = [row for row in rows if row['fold'] == fold]
        for kind in ('word', 'bpe'):
            tokenizer = fit_tokenizer(training, kind)
            for source_name in ('uk', 'ru'):
                counts = defaultdict(int)
                for row in evaluation:
                    if row['source'] != source_name:
                        continue
                    encoding = tokenizer.encode(row['text'])
                    counts['rows'] += 1
                    counts['tokens'] += len(encoding.ids)
                    counts['rows_over_256_tokens'] += len(encoding.ids) > 256
                    markers = [tuple(pair) for pair in row['markers']]
                    counts['gold_mentions'] += len(markers)
                    counts['unrepresentable_rows'] += not representable(encoding.offsets, markers)
                    for a, b in markers:
                        indices = [i for i, (s, e) in enumerate(encoding.offsets)
                                   if s < b and a < e]
                        counts['gold_mention_pieces'] += len(indices)
                        counts['mentions_with_unknown_token'] += any(
                            encoding.ids[i] == tokenizer.token_to_id('[UNK]') for i in indices)
                        counts['unrepresentable_mentions'] += not (
                            any(encoding.offsets[i][0] == a for i in indices) and
                            any(encoding.offsets[i][1] == b for i in indices))
                result.append({'fold': fold, 'tokenizer': kind, 'source': source_name,
                               **counts,
                               'mean_pieces_per_mention': counts['gold_mention_pieces'] / counts['gold_mentions'],
                               'unknown_mention_share': counts['mentions_with_unknown_token'] / counts['gold_mentions'],
                               'unrepresentable_mention_share': counts['unrepresentable_mentions'] / counts['gold_mentions']})
        print('completed fold', fold, flush=True)
    report = {'definition': 'Vocabularies fitted on the other two grouped folds; all held-out CV rows measured once per tokenizer. A mention has unknown-token exposure if any overlapping token is [UNK].',
              'results': result}
    path = OUTPUT / 'eda_comparison' / 'representation.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    run()
