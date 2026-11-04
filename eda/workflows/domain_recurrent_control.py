"""Confirm source/test separation with the extractor's word BiGRU encoder."""

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from eda.colab.colab_extractor import fit_tokenizer
from eda.core.analysis import OUTPUT
from eda.workflows.domain_controls import SEEDS, mask_cues, select, split_groups


# Historical public name retained for callers; the implementation is shared.
from eda.core.domain_recurrent import DomainWordRecurrent, encode, fit


class DomainWordBiGRU(DomainWordRecurrent):
    def __init__(self, vocabulary_size):
        super().__init__(vocabulary_size, 'gru')


def one(source_rows, test_rows, source, seed):
    torch.manual_seed(seed)
    src_train, src_hold = split_groups(source_rows, source)
    tst_train, tst_hold = split_groups(test_rows, 'test')
    train = [mask_cues(row['text']) for row in src_train + tst_train]
    hold = [mask_cues(row['text']) for row in src_hold + tst_hold]
    tokenizer = fit_tokenizer([{'text': text} for text in train], 'word')
    labels = [0] * len(src_train) + [1] * len(tst_train)
    heldout_labels = np.array([0] * len(src_hold) + [1] * len(tst_hold))
    model, probabilities, diagnostics = fit(train, hold, labels, tokenizer, 'gru', seed)
    rng = np.random.default_rng(seed)
    intervals = []
    n = len(src_hold)
    for _ in range(500):
        sampled = np.concatenate([rng.choice(n, n, replace=True),
                                  n + rng.choice(len(tst_hold), len(tst_hold), replace=True)])
        intervals.append(roc_auc_score(heldout_labels[sampled], probabilities[sampled]))
    return {'source': source, 'seed': seed, 'auc': roc_auc_score(heldout_labels, probabilities),
            'group_bootstrap_95': np.quantile(intervals, [.025, .975]).tolist(),
            'train_per_class': len(src_train), 'heldout_per_class': n,
            'vocabulary_size': tokenizer.get_vocab_size(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'seconds': diagnostics['seconds'],
            'heldout_source_ids': [row['row_id'] for row in src_hold],
            'heldout_test_ids': [row['row_id'] for row in tst_hold],
            'probabilities': probabilities.tolist()}


def run():
    torch.set_num_threads(2)
    audit = json.loads((OUTPUT / 'audit.json').read_text())
    results = []
    for source in ('uk', 'ru'):
        (_, _), (src, test) = select(audit, source)
        for seed in SEEDS:
            result = one(src, test, source, seed)
            results.append(result)
            print(source, seed, round(result['auc'], 3), flush=True)
    report = {'protocol': 'Same 96-dimensional word embedding and 96-per-direction BiGRU encoder as extractor; binary document head, six epochs, matched 160 per class, URLs/emoji/newlines masked, grouped 75/25 split.',
              'caveat': 'This is a domain-classification experiment, not an extraction score. The 40 held-out rows per class give wide intervals; the source/test collections remain different in more ways than length.',
              'results': results}
    (OUTPUT / 'eda_comparison' / 'domain_word_bigru.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    run()
