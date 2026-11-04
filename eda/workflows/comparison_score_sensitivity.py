"""Check whether correcting the scorer's punctuation expression changes ranking."""

import gzip
import json
from collections import defaultdict

from eda.core.analysis import OUTPUT
from eda.workflows.score_sensitivity import corrected
from eda.workflows.summarize_comparison import BASE, rates


def run():
    with gzip.open(OUTPUT / 'extractor_bundle_v1.jsonl.gz', 'rt', encoding='utf-8') as source:
        texts = {(row['source'], row['row_id']): row['text'] for line in source
                 if (row := json.loads(line))['partition'] == 'cv'}
    report = json.loads((BASE / 'results.json').read_text())
    if len(report['runs']) != 15:
        raise RuntimeError('Comparison predictions are incomplete')
    counts = defaultdict(lambda: [0, 0, 0])
    for run in report['runs']:
        label = f"{run['mixture']}_{run['architecture']}_{run['tokenizer']}"
        with gzip.open(BASE / f"{run['name']}_predictions.jsonl.gz", 'rt', encoding='utf-8') as source:
            for line in source:
                item = json.loads(line)
                text = texts[(item['source'], item['row_id'])]
                actual = {corrected(text[a:b]) for a, b in item['actual_spans']}
                predicted = {corrected(text[a:b]) for a, b in item['predicted_spans']}
                for source_name in (item['source'], 'combined'):
                    count = counts[(label, source_name)]
                    count[0] += len(actual & predicted)
                    count[1] += len(predicted - actual)
                    count[2] += len(actual - predicted)
    result = {f'{label}:{source}': rates(value)
              for (label, source), value in sorted(counts.items())}
    output = {'definition': 'Diagnostic correction to the punctuation expression; not verified as the live Kaggle scorer.',
              'scores': result}
    (BASE / 'score_sensitivity.json').write_text(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    run()
