"""Measure how much held-out domain AUC is explained by script and length alone."""

import json

from sklearn.metrics import roc_auc_score

from eda.core.analysis import OUTPUT
from eda.workflows.domain_controls import mask_cues


def run():
    domain = json.loads((OUTPUT / 'eda_comparison/domain_word_bigru.json').read_text())
    audit = json.loads((OUTPUT / 'audit.json').read_text())
    source = {name: {row['row_id']: row['text'] for row in audit[name]['sample']}
              for name in ('uk', 'ru')}
    test = {i: row['text'] for i, row in enumerate(audit['test']['rows'])}
    results = []
    for item in domain['results']:
        src = [source[item['source']][i] for i in item['heldout_source_ids']]
        tst = [test[i] for i in item['heldout_test_ids']]
        labels = [0] * len(src) + [1] * len(tst)
        texts = src + tst
        ukrainian_letters = [int(any(c in 'іїєґІЇЄҐ' for c in text)) for text in texts]
        lengths = [len(mask_cues(text)) for text in texts]
        results.append({'source': item['source'], 'seed': item['seed'],
                        'bigru_auc': item['auc'],
                        'ukrainian_letter_auc': roc_auc_score(labels, ukrainian_letters),
                        'length_auc': roc_auc_score(labels, lengths),
                        'heldout_per_class': len(src)})
    report = {'definition': 'AUC of a single indicator for Ukrainian-specific letters, and of masked text length, on exactly the same held-out rows as the word BiGRU. An AUC below 0.5 still indicates separability if the direction is reversed.',
              'results': results}
    (OUTPUT / 'eda_comparison/domain_cue_baselines.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    run()
