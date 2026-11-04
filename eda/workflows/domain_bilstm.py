"""Freeze, run, and review the bounded word-BiLSTM domain experiment.

Run ``python -m eda.workflows.domain_bilstm --max-new-runs 1`` to time the
first fit, then rerun without the limit. Existing results resume only when
input, code, and training configuration fingerprints still match.
"""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tokenizers import Tokenizer

from eda.colab.colab_extractor import fit_tokenizer
from eda.core.analysis import OUTPUT
from eda.core.domain_recurrent import fit
from eda.workflows.domain_controls import SEEDS, mask_cues, select, split_groups, template

BASE = OUTPUT / 'eda_comparison' / 'domain_word_bilstm'
PROTOCOL = 'word_bilstm_domain_v1'
CONDITIONS = ('matched_masked', 'broader_raw')
SETTINGS = {'epochs': 6, 'batch_size': 32, 'learning_rate': .003,
            'weight_decay': .01, 'dropout': .15, 'gradient_clip': 1.,
            'embedding': 96, 'hidden_per_direction': 96, 'max_tokens': 256,
            'vocabulary_limit': 16000, 'seeds': list(SEEDS),
            'shuffle': 'torch global RNG, historical GRU convention; orders recorded'}


def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def transformed(row, condition):
    return mask_cues(row['text']) if condition == 'matched_masked' else row['text']


def audit_membership(rows, condition):
    """Fail closed on accepted groups, documents, or duplicate input leakage."""
    report = {}
    for name, key in (
        ('accepted_groups', lambda r: r['group_id']),
        ('documents', lambda r: f"{r['domain']}:{r['doc_id']}" if r['doc_id'] else r['group_id']),
        ('exact_text', lambda r: r['text']),
        ('normalized_templates', lambda r: template(r['text'])),
        ('model_input', lambda r: transformed(r, condition)),
    ):
        splits = defaultdict(set)
        for row in rows:
            splits[key(row)].add(row['split'])
        crossing = sum(len(value) > 1 for value in splits.values())
        report[name + '_crossing'] = crossing
        if crossing:
            raise ValueError(f'{condition}: {crossing} {name} cross train/holdout; '
                             'correct the frozen split and rerun both recurrent models')
    source_text = {template(r['text']) for r in rows if r['domain'] == 0}
    test_text = {template(r['text']) for r in rows if r['domain'] == 1}
    report['cross_domain_template_overlap'] = len(source_text & test_text)
    if report['cross_domain_template_overlap']:
        raise ValueError('Source/test share text templates: resolve conflicting origin targets first')
    assert all(r['role'] in ('train', 'development', 'competition') for r in rows)
    assert len({(r['domain'], r['row_id']) for r in rows}) == len(rows)
    return report


def audit_encoded(rows, condition, tokenizer):
    """Check exactly the IDs consumed, including truncation and empty fallback."""
    splits, domains = defaultdict(set), defaultdict(set)
    for row in rows:
        key = tuple(tokenizer.encode(transformed(row, condition)).ids[:256]) or (0,)
        splits[key].add(row['split'])
        domains[key].add(row['domain'])
    result = {'encoded_input_crossing': sum(len(v) > 1 for v in splits.values()),
              'cross_domain_encoded_overlap': sum(len(v) > 1 for v in domains.values())}
    if any(result.values()):
        raise ValueError(f'Encoded first-256-token inputs collide: {result}')
    return result


def prepare():
    """Freeze text, group membership, training vocabulary, and audit evidence."""
    audit = read(OUTPUT / 'audit.json')
    historical = read(OUTPUT / 'eda_comparison/domain_word_bigru.json')
    input_paths = [OUTPUT / 'audit.json', OUTPUT / 'role_manifest_v2.parquet',
                   OUTPUT / 'eda_comparison/domain_word_bigru.json']
    manifest = {'protocol': PROTOCOL, 'settings': SETTINGS,
                'inputs': {str(p): digest_file(p) for p in input_paths}, 'datasets': []}
    BASE.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as db:
        for source in ('uk', 'ru'):
            broader, matched = select(audit, source)
            for condition, (src, tst) in zip(CONDITIONS, (matched, broader)):
                ids = [r['row_id'] for r in src]
                roles = {row[0]: row[1:] for row in db.execute(
                    'SELECT row_id,group_id,role_v2 FROM read_parquet(?) '
                    'WHERE source=? AND row_id IN (SELECT unnest(?))',
                    [str(OUTPUT / 'role_manifest_v2.parquet'), source, ids]).fetchall()}
                rows = []
                for domain, records, name in ((0, src, source), (1, tst, 'test')):
                    for split, partition in zip(('train', 'holdout'), split_groups(records, name)):
                        for row in partition:
                            group, role = roles[row['row_id']] if domain == 0 else (
                                'test:' + digest(template(row['text'])), 'competition')
                            rows.append({'row_id': row['row_id'], 'domain': domain,
                                         'split': split, 'group_id': group,
                                         'doc_id': row.get('doc_id'), 'role': role, 'text': row['text']})
                checks = audit_membership(rows, condition)
                if condition == 'matched_masked':
                    for old in (r for r in historical['results'] if r['source'] == source):
                        for domain, field in ((0, 'heldout_source_ids'), (1, 'heldout_test_ids')):
                            assert [r['row_id'] for r in rows if r['split'] == 'holdout' and
                                    r['domain'] == domain] == old[field]
                train = [transformed(r, condition) for r in rows if r['split'] == 'train']
                tokenizer = fit_tokenizer([{'text': t} for t in train], 'word')
                audit_encoded(rows, condition, tokenizer)
                name = f'{source}_{condition}'
                tokenizer_path = BASE / f'{name}_tokenizer.json'
                tokenizer_json = tokenizer.to_str()
                if tokenizer_path.exists():
                    assert json.loads(tokenizer_path.read_text()) == json.loads(tokenizer_json)
                else:
                    tokenizer_path.write_text(tokenizer_json)
                item = {'name': name, 'source': source, 'condition': condition,
                        'rows': rows, 'audit': checks,
                        'tokenizer_sha256': digest_file(tokenizer_path),
                        'vocabulary_size': tokenizer.get_vocab_size()}
                item['diagnostics'] = input_diagnostics(item, tokenizer)
                manifest['datasets'].append(item)
    path = BASE / 'selection.json'
    if path.exists():
        assert read(path) == manifest, 'Frozen selection changed; use a new protocol/output folder'
    else:
        save(path, manifest)
    return manifest


def input_diagnostics(dataset, tokenizer):
    answer = []
    for split in ('train', 'holdout'):
        for domain in (0, 1):
            rows = [r for r in dataset['rows'] if r['split'] == split and r['domain'] == domain]
            texts = [transformed(r, dataset['condition']) for r in rows]
            encoded = [tokenizer.encode(t).ids for t in texts]
            total = sum(map(len, encoded))
            answer.append({'split': split, 'domain': domain, 'rows': len(rows),
                           'raw_characters_mean': float(np.mean([len(r['text']) for r in rows])),
                           'input_characters_mean': float(np.mean(list(map(len, texts)))),
                           'tokens_p50_p90_max': np.percentile(list(map(len, encoded)), [50, 90, 100]).tolist(),
                           'truncated_rows': sum(len(ids) > 256 for ids in encoded),
                           'omitted_tokens': sum(max(0, len(ids) - 256) for ids in encoded),
                           'unknown_tokens': sum(ids.count(1) for ids in encoded),
                           'unknown_fraction': sum(ids.count(1) for ids in encoded) / max(total, 1),
                           'empty_inputs': sum(not ids for ids in encoded)})
    return answer


def group_draws(rows, seed, repetitions=1000):
    """Stratified cluster bootstrap; one common index draw for paired models."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['domain'], row['group_id'])].append(i)
    partitions = [[indices for (label, _), indices in groups.items() if label == domain]
                  for domain in (0, 1)]
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        yield np.concatenate([np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
                              for members in partitions])


def metrics(rows, probabilities, seed, reference=None):
    labels = np.array([r['domain'] for r in rows])
    probabilities = np.asarray(probabilities)
    score = roc_auc_score(labels, probabilities)
    values, differences = [], []
    for positions in group_draws(rows, seed):
        current = roc_auc_score(labels[positions], probabilities[positions])
        values.append(current)
        if reference is not None:
            differences.append(current - roc_auc_score(labels[positions], np.asarray(reference)[positions]))
    result = {'auc': score, 'auc_group_bootstrap_95': np.quantile(values, [.025, .975]).tolist(),
              'balanced_accuracy_at_0_5': balanced_accuracy_score(labels, probabilities >= .5)}
    if reference is not None:
        result.update(gru_auc=roc_auc_score(labels, reference),
                      lstm_minus_gru_auc=score - roc_auc_score(labels, reference),
                      paired_difference_95=np.quantile(differences, [.025, .975]).tolist())
    return result


def provenance(manifest):
    paths = [Path(__file__), Path('eda/core/domain_recurrent.py'),
             Path('eda/workflows/domain_controls.py'), Path('eda/colab/colab_extractor.py')]
    return {'protocol': PROTOCOL, 'selection_sha256': digest(manifest), 'settings': SETTINGS,
            'code': {str(p): digest_file(p) for p in paths},
            'python': platform.python_version(), 'torch': torch.__version__,
            'packages': {name: version(name) for name in ('tokenizers', 'numpy', 'scikit-learn')},
            'device': 'cpu', 'threads': 2}


def run(max_new_runs=0):
    torch.set_num_threads(2)
    manifest = prepare()
    prov = provenance(manifest)
    path = BASE / 'results.json'
    report = read(path) if path.exists() else {'provenance': prov, 'runs': []}
    assert report['provenance'] == prov, 'Inputs/code/environment changed; do not silently resume'
    old_results = read(OUTPUT / 'eda_comparison/domain_word_bigru.json')['results']
    done = {r['name'] for r in report['runs']}
    new = 0
    for dataset in manifest['datasets']:
        condition = dataset['condition']
        train = [r for r in dataset['rows'] if r['split'] == 'train']
        hold = [r for r in dataset['rows'] if r['split'] == 'holdout']
        tokenizer = Tokenizer.from_file(str(BASE / f"{dataset['name']}_tokenizer.json"))
        for seed in SEEDS:
            name = f"{dataset['name']}_{seed}"
            if name in done:
                continue
            texts = [[transformed(r, condition) for r in partition] for partition in (train, hold)]
            reference = None
            replay = None
            if condition == 'matched_masked':
                old = next(r for r in old_results if r['source'] == dataset['source'] and r['seed'] == seed)
                # A short replay validates vocabulary IDs, implementation, and row order
                # against saved probabilities before accepting a historical comparator.
                _, replay_predictions, replay_diagnostics = fit(*texts, [r['domain'] for r in train],
                                                               tokenizer, 'gru', seed)
                maximum = float(np.max(np.abs(replay_predictions - np.asarray(old['probabilities']))))
                if maximum > 1e-6:
                    raise ValueError(f'{name}: historical GRU replay mismatch {maximum}; rerun reference explicitly')
                reference = old['probabilities']
                replay = {'maximum_probability_difference': maximum,
                          'seconds': replay_diagnostics['seconds'], 'history': replay_diagnostics['history'],
                          'batch_orders': replay_diagnostics['batch_orders']}
            model, probabilities, diagnostics = fit(*texts, [r['domain'] for r in train],
                                                    tokenizer, 'lstm', seed)
            result = {'name': name, 'source': dataset['source'], 'condition': condition,
                      'architecture': 'lstm', 'seed': seed,
                      'train_per_class': [sum(r['domain'] == d for r in train) for d in (0, 1)],
                      'heldout_per_class': [sum(r['domain'] == d for r in hold) for d in (0, 1)],
                      'prediction_ids': [[r['domain'], r['row_id']] for r in hold],
                      'probabilities': probabilities.tolist(), 'training': diagnostics,
                      'historical_gru_replay': replay, **metrics(hold, probabilities, seed, reference)}
            torch.save(model.state_dict(), BASE / f'{name}.pt')
            report['runs'].append(result)
            save(path, report)
            print(name, 'AUC', round(result['auc'], 4), 'fit seconds', round(diagnostics['seconds'], 2), flush=True)
            new += 1
            if max_new_runs and new >= max_new_runs:
                return report
    summarize(manifest, report)
    return report


def summarize(manifest, report):
    """Summaries use final saved probabilities; no additional fitting or tuning."""
    assert len(report['runs']) == 12
    summaries, cues, review = [], [], []
    old = read(OUTPUT / 'eda_comparison/domain_word_bigru.json')['results']
    for dataset in manifest['datasets']:
        runs = [r for r in report['runs'] if r['source'] == dataset['source'] and r['condition'] == dataset['condition']]
        hold = [r for r in dataset['rows'] if r['split'] == 'holdout']
        labels = [r['domain'] for r in hold]
        means = np.mean([r['probabilities'] for r in runs], axis=0)
        summaries.append({'source': dataset['source'], 'condition': dataset['condition'],
                          'auc_mean': float(np.mean([r['auc'] for r in runs])),
                          'auc_min': min(r['auc'] for r in runs), 'auc_max': max(r['auc'] for r in runs)})
        cues.append({'source': dataset['source'], 'condition': dataset['condition'],
                     'ukrainian_letter_auc': roc_auc_score(labels, [any(c in 'іїєґІЇЄҐ' for c in r['text']) for r in hold]),
                     'input_length_auc': roc_auc_score(labels, [len(transformed(r, dataset['condition'])) for r in hold])})
        categories = {'source_looks_like_test': sorted((i for i,r in enumerate(hold) if r['domain'] == 0), key=lambda i: -means[i])[:3],
                      'test_looks_like_source': sorted((i for i,r in enumerate(hold) if r['domain'] == 1), key=lambda i: means[i])[:3]}
        reference = None
        if dataset['condition'] == 'matched_masked':
            reference = np.mean([r['probabilities'] for r in old if r['source'] == dataset['source']], axis=0)
            categories['largest_gru_lstm_disagreement'] = np.argsort(-abs(means - reference), kind='stable')[:3].tolist()
        for category, indices in categories.items():
            for i in indices:
                row = hold[i]
                review.append({'source': dataset['source'], 'condition': dataset['condition'],
                               'category': category, 'domain': row['domain'], 'row_id': row['row_id'],
                               'p_competition_mean': float(means[i]),
                               'gru_p_competition_mean': float(reference[i]) if reference is not None else None,
                               'text': row['text']})
    save(BASE / 'summary.json', {'protocol': PROTOCOL, 'summaries': summaries, 'cue_baselines': cues,
                               'review_examples': review,
                               'caveat': 'Seeds share a holdout. Group bootstrap conditions on fitted models. '
                               'Broader and matched selections differ in rows and sample size; no causal masking contrast.'})


def review_saved():
    """Audit completed fits without relabeling their original training provenance."""
    manifest = prepare()
    report = read(BASE / 'results.json')
    assert report['provenance']['selection_sha256'] == digest(manifest)
    checks = {}
    for dataset in manifest['datasets']:
        tokenizer = Tokenizer.from_file(str(BASE / f"{dataset['name']}_tokenizer.json"))
        checks[dataset['name']] = audit_encoded(dataset['rows'], dataset['condition'], tokenizer)
    summarize(manifest, report)
    save(BASE / 'post_run_audit.json', {
        'training_provenance': report['provenance'],
        'audit_provenance': provenance(manifest), 'encoded_checks': checks,
        'revision': 'Added encoded-prefix collision audit and package versions after training. '
                    'Training function and model were unchanged. Original fit provenance is retained; '
                    'this audit does not claim revised workflow code produced the saved fits.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-new-runs', type=int, default=0)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--review-only', action='store_true')
    parser.add_argument('--output', type=Path, default=BASE)
    args = parser.parse_args()
    BASE = args.output
    if args.review_only:
        review_saved()
    elif args.prepare_only:
        prepare()
    else:
        run(args.max_new_runs)
