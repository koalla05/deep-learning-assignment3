"""Inspect saved length-controlled domain predictions without retraining."""

import json
from pathlib import Path

from eda.core.analysis import OUTPUT, text_features


def run():
    report = json.loads((OUTPUT / 'domain_controls.json').read_text())
    audit = json.loads((OUTPUT / 'audit.json').read_text())
    test = {i: row for i, row in enumerate(audit['test']['rows'])}
    source = {name: {row['row_id']: row for row in audit[name]['sample']}
              for name in ('uk', 'ru')}
    review = []
    for result in report['results']:
        if result['seed'] != 3012026 or result['condition'] != 'length_matched_cues_masked':
            continue
        n = result['heldout_source']
        values = result['heldout_probabilities']
        false_test = sorted(range(n), key=lambda i: values[i], reverse=True)[:5]
        false_source = sorted(range(n, len(values)), key=lambda i: values[i])[:5]
        for kind, indices in [('source_looks_like_test', false_test),
                              ('test_looks_like_source', false_source)]:
            for i in indices:
                row = (source[result['source']][result['heldout_source_ids'][i]]
                       if i < n else test[result['heldout_test_ids'][i - n]])
                review.append({'source': result['source'], 'kind': kind,
                               'probability_test': values[i], 'text': row['text'],
                               'features': text_features(row['text'])})
    path = OUTPUT / 'domain_error_review.json'
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2))
    return review


if __name__ == '__main__':
    for item in run():
        print(item['source'], item['kind'], round(item['probability_test'], 3),
              repr(item['text'][:160]))
