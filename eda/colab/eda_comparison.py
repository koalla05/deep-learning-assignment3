"""Frozen, bounded recurrent comparisons for the remaining EDA.

Upload this file beside colab_extractor.py and extractor_bundle_v1.jsonl.gz.
Run EDA_COMPARE_MODE=audit first, then EDA_COMPARE_MODE=run. Results are
written after every fit so a disconnected Colab session retains progress.
"""

import gzip
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

try:
    from .colab_extractor import (
    DEVICE, EPOCHS, SEED, TOKEN_BUDGET, BiGRU, collate, fit_tokenizer,
    predict, rank, score, windows,
    )
except ImportError:
    from colab_extractor import (
        DEVICE, EPOCHS, SEED, TOKEN_BUDGET, BiGRU, collate, fit_tokenizer,
        predict, rank, score, windows,
    )

DATA = Path(os.getenv('EDA_COMPARE_DATA', '/content/extractor_bundle_v1.jsonl.gz'))
OUT = Path(os.getenv('EDA_COMPARE_OUT', '/content/eda_comparison'))
ARTIFACT = OUT / 'selection.json'


class BiLSTM(BiGRU):
    def __init__(self, vocab):
        super().__init__(vocab)
        del self.gru
        self.gru = nn.LSTM(96, 96, batch_first=True, bidirectional=True)


def load_cv():
    with gzip.open(DATA, 'rt', encoding='utf-8') as source:
        rows = [row for line in source if (row := json.loads(line))['partition'] == 'cv']
    assert len(rows) == 63388
    assert len({(row['source'], row['row_id']) for row in rows}) == len(rows)
    assert {row['fold'] for row in rows} == {0, 1, 2}
    return rows


def identity(row):
    return f"{row['source']}:{row['row_id']}"


def source_prefix(rows, tokenizers, budget):
    selected = []
    spent = defaultdict(int)
    rejected = defaultdict(int)
    for row in sorted(rows, key=rank):
        costs = {}
        for kind, tokenizer in tokenizers.items():
            pieces, ok = windows(row, tokenizer, True)
            if not ok:
                rejected[f'unrepresentable_{kind}'] += 1
                break
            costs[kind] = sum(len(item[0]) for item in pieces)
        if len(costs) != len(tokenizers):
            continue
        if any(spent[kind] + cost > budget for kind, cost in costs.items()):
            break
        selected.append(row)
        for kind, cost in costs.items():
            spent[kind] += cost
    return selected, dict(spent), dict(rejected)


def make_selection(rows):
    folds = []
    for fold in range(3):
        training = [row for row in rows if row['fold'] != fold]
        evaluation = [row for row in rows if row['fold'] == fold]
        tokenizers = {kind: fit_tokenizer(training, kind) for kind in ('word', 'bpe')}
        candidates = {}
        diagnostics = {}
        for source in ('uk', 'ru'):
            candidates[source], costs, rejected = source_prefix(
                [row for row in training if row['source'] == source],
                tokenizers, TOKEN_BUDGET // 2)
            diagnostics[source] = {'eligible_prefix_rows': len(candidates[source]),
                                   'prefix_tokens': costs, 'rejected': rejected}
        count = min(len(candidates['uk']), len(candidates['ru']))
        assert count > 0
        mixed = candidates['uk'][:count] + candidates['ru'][:count]
        ukrainian, _, uk_rejected = source_prefix(
            [row for row in training if row['source'] == 'uk'],
            tokenizers, TOKEN_BUDGET)
        assert len(ukrainian) >= 2 * count, (fold, count, len(ukrainian))
        ukrainian = ukrainian[:2 * count]
        for name, selection in (('mixed', mixed), ('uk', ukrainian)):
            assert len({identity(row) for row in selection}) == len(selection)
            totals = {}
            for kind, tokenizer in tokenizers.items():
                pieces = [window for row in selection for window in windows(row, tokenizer, True)[0]]
                totals[kind] = sum(len(item[0]) for item in pieces)
                assert totals[kind] <= TOKEN_BUDGET
            diagnostics[f'{name}_tokens'] = totals
        diagnostics['uk_only_rejected'] = uk_rejected
        folds.append({'fold': fold, 'training_pool_rows': len(training),
                      'evaluation_rows': len(evaluation), 'rows_per_source': count,
                      'mixed_ids': [identity(row) for row in mixed],
                      'uk_ids': [identity(row) for row in ukrainian],
                      'diagnostics': diagnostics})
        print('audit fold', fold, 'rows per source', count,
              'mixed token totals', diagnostics['mixed_tokens'], flush=True)
    with DATA.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    return {'protocol': 'common_rows_v1', 'bundle_sha256': digest,
            'seed': SEED, 'epochs': EPOCHS, 'token_ceiling': TOKEN_BUDGET,
            'tokenizer_vocab': 16000, 'window': 256, 'stride': 192,
            'folds': folds}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def train_selected(selected, tokenizer, architecture, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    examples = []
    for row in selected:
        items, ok = windows(row, tokenizer, True)
        assert ok, identity(row)
        examples.extend(items)
    model = (BiGRU if architecture == 'gru' else BiLSTM)(tokenizer.get_vocab_size()).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    weights = torch.tensor([1., 3., 2.], device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    losses = []
    started = time.monotonic()
    for epoch in range(EPOCHS):
        random.shuffle(examples)
        model.train()
        total = 0.0
        steps = 0
        for start in range(0, len(examples), 32):
            x, y, lengths = collate(examples[start:start + 32])
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = loss_fn(logits.reshape(-1, 3), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            steps += 1
        losses.append(total / steps)
        print('epoch', epoch + 1, 'loss', round(losses[-1], 4), flush=True)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    return model, {'training_rows': len(selected), 'training_windows': len(examples),
                   'training_tokens': sum(len(item[0]) for item in examples),
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'losses': losses, 'training_seconds': time.monotonic() - started}


def run(rows, selection):
    with DATA.open('rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == selection['bundle_sha256']
    row_lookup = {identity(row): row for row in rows}
    scenarios = [('word', 'gru', 'mixed'), ('bpe', 'gru', 'mixed'),
                 ('word', 'lstm', 'mixed'), ('bpe', 'lstm', 'mixed'),
                 ('word', 'gru', 'uk')]
    results_path = OUT / 'results.json'
    report = json.loads(results_path.read_text()) if results_path.exists() else {
        'protocol': selection['protocol'], 'bundle_sha256': selection['bundle_sha256'],
        'seed': SEED, 'device': str(DEVICE),
        'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None,
        'torch_version': torch.__version__, 'runs': []}
    assert report['bundle_sha256'] == selection['bundle_sha256']
    completed = {run['name'] for run in report['runs']}
    max_new_runs = int(os.getenv('EDA_COMPARE_MAX_RUNS', '0'))
    new_runs = 0
    for manifest in selection['folds']:
        fold = manifest['fold']
        training = [row for row in rows if row['fold'] != fold]
        evaluation = [row for row in rows if row['fold'] == fold]
        tokenizers = {kind: fit_tokenizer(training, kind) for kind in ('word', 'bpe')}
        for kind, architecture, mixture in scenarios:
            name = f'fold{fold}_{mixture}_{architecture}_{kind}'
            if name in completed:
                continue
            selected = [row_lookup[key] for key in manifest['mixed_ids' if mixture == 'mixed' else 'uk_ids']]
            print('run', name, 'rows', len(selected), 'evaluation', len(evaluation), flush=True)
            started = time.monotonic()
            model, diagnostics = train_selected(selected, tokenizers[kind], architecture, SEED)
            guessed = predict(model, tokenizers[kind], evaluation)
            metrics, details = score(evaluation, guessed)
            model_dir = OUT / 'checkpoints'
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.cpu().state_dict(), model_dir / f'{name}.pt')
            tokenizers[kind].save(str(model_dir / f'{name}.json'))
            with gzip.open(OUT / f'{name}_predictions.jsonl.gz', 'wt', encoding='utf-8') as output:
                for detail in details:
                    output.write(json.dumps(detail, ensure_ascii=False) + '\n')
            report['runs'].append({'name': name, 'fold': fold, 'mixture': mixture,
                                   'architecture': architecture, 'tokenizer': kind,
                                   'selection_sha256': hashlib.sha256(
                                       '\n'.join(manifest['mixed_ids' if mixture == 'mixed' else 'uk_ids']).encode()
                                   ).hexdigest(), 'training': diagnostics,
                                   'metrics': metrics, 'evaluation_rows': len(evaluation),
                                   'total_seconds': time.monotonic() - started})
            save_json(results_path, report)
            print('completed', name, 'F1', metrics['combined']['string_f1'], flush=True)
            new_runs += 1
            if max_new_runs and new_runs >= max_new_runs:
                return report
    return report


def main():
    rows = load_cv()
    OUT.mkdir(parents=True, exist_ok=True)
    mode = os.getenv('EDA_COMPARE_MODE', 'audit')
    if mode == 'audit':
        selection = make_selection(rows)
        save_json(ARTIFACT, selection)
        print('saved', ARTIFACT, flush=True)
    elif mode == 'run':
        selection = json.loads(ARTIFACT.read_text())
        run(rows, selection)
    else:
        raise ValueError(mode)


if __name__ == '__main__':
    main()
