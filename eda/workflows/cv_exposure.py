"""Audit name recall against the examples actually consumed by each CV model."""

import gzip
import json
import tarfile
from collections import defaultdict

from tokenizers import Tokenizer

from eda.core.analysis import OUTPUT
from eda.colab.colab_extractor import choose_training, normalize


def main():
    with gzip.open(OUTPUT / 'extractor_bundle_v1.jsonl.gz', 'rt') as file:
        rows = [row for line in file if (row := json.loads(line))['partition'] == 'cv']
    # The bundle and prediction keys are unique within each source.
    row_map = {(row['source'], row['row_id']): row for row in rows}
    assert len(row_map) == len(rows) == 63388

    predictions = defaultdict(dict)
    with gzip.open(OUTPUT / 'eda_cv_predictions.jsonl.gz', 'rt') as file:
        for line in file:
            item = json.loads(line)
            predictions[item['model']][(item['source'], item['row_id'])] = item

    results = []
    with tarfile.open(OUTPUT / 'eda_cv_checkpoints.tar.gz', 'r:gz') as archive:
        for fold in range(3):
            training = [row for row in rows if row['fold'] != fold]
            evaluation = [row for row in rows if row['fold'] == fold]
            for kind, mix in [('bpe', 'mixed'), ('bpe', 'uk'), ('word', 'mixed')]:
                name = f'cv_{fold}_{kind}_{mix}'
                member = archive.extractfile(f'eda_checkpoints/{name}.json')
                assert member is not None
                tokenizer = Tokenizer.from_str(member.read().decode())
                windows, spent, excluded = choose_training(training, tokenizer, mix)
                selected = {(item[3]['source'], item[3]['row_id']) for item in windows}
                available_names = defaultdict(set)
                supervised_names = defaultdict(set)
                for row in training:
                    names = {normalize(row['text'][a:b]) for a, b in row['markers']}
                    available_names[row['source']].update(names)
                    if (row['source'], row['row_id']) in selected:
                        supervised_names[row['source']].update(names)
                assert len(predictions[name]) == len(evaluation)
                counters = defaultdict(lambda: {'gold': 0, 'correct': 0, 'rows': 0})
                for row in evaluation:
                    source = row['source']
                    item = predictions[name][(source, row['row_id'])]
                    gold = set(item['actual_strings'])
                    predicted = set(item['predicted_strings'])
                    for value in gold:
                        exposure = ('supervised_seen' if value in supervised_names[source]
                                    else 'available_unsampled' if value in available_names[source]
                                    else 'absent_from_fold_training')
                        counter = counters[(source, exposure)]
                        counter['gold'] += 1
                        counter['correct'] += value in predicted
                    counters[(source, 'all')]['rows'] += 1
                results.append({
                    'model': name,
                    'training_rows': {source: sum(key[0] == source for key in selected)
                                      for source in ('uk', 'ru')},
                    'spent_tokens': spent,
                    'excluded_unrepresentable': excluded,
                    'source_results': [
                        {'source': source, 'exposure': exposure, **count,
                         'recall': count['correct'] / count['gold'] if count['gold'] else None}
                        for (source, exposure), count in sorted(counters.items())
                        if exposure != 'all'
                    ],
                })
    report = {
        'definition': 'Each gold normalized string is classified by whether it appeared in a supervised row actually selected by the fixed token budget, only in the remaining training-fold pool, or nowhere in that pool. Tokenizers were fitted on the whole training-fold pool.',
        'caveat': 'String exposure does not prove the model learned a name; a string may occur without a location label in other rows. Recall is conditional on gold labels and does not measure false positives.',
        'results': results,
    }
    (OUTPUT / 'eda_cv_exposure.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    main()
