"""Small GRU domain classifier for the domain-analysis notebook."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn

from eda.core.analysis import OUTPUT, SEED


MAX_CHARS = 256
SOURCE_ROWS = 477
EPOCHS = 8


class DomainGRU(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 32, padding_idx=0)
        self.gru = nn.GRU(32, 48, batch_first=True)
        self.output = nn.Linear(48, 1)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.output(hidden[-1]).squeeze(-1)


def _select_source(records: list[dict], source: str, limit: int) -> list[dict]:
    chosen = []
    seen_groups = set()
    seen_texts = set()
    for row in records:
        group = row["doc_id"] if source == "ru" else row["text"]
        if group in seen_groups or row["text"] in seen_texts or not row["text"]:
            continue
        seen_groups.add(group)
        seen_texts.add(row["text"])
        chosen.append(row)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        raise ValueError(f"Only {len(chosen)} distinct {source} groups in audit sample")
    return chosen


def _vocabulary(texts: list[str]) -> dict[str, int]:
    frequencies = Counter(char for text in texts for char in text[:MAX_CHARS])
    return {char: index + 2 for index, (char, _) in enumerate(frequencies.most_common(6000))}


def _encode(texts: list[str], vocabulary: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.zeros((len(texts), MAX_CHARS), dtype=torch.long)
    lengths = torch.empty(len(texts), dtype=torch.long)
    for index, text in enumerate(texts):
        chars = text[:MAX_CHARS]
        lengths[index] = max(1, len(chars))
        if chars:
            matrix[index, :len(chars)] = torch.tensor([vocabulary.get(char, 1) for char in chars])
    return matrix, lengths


def run_one(audit: dict, source: str, seed: int = SEED) -> dict:
    """Balanced train/holdout classification; source groups and test texts stay disjoint."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    source_records = _select_source(audit[source]["sample"], source, SOURCE_ROWS)
    test_records = audit["test"]["rows"]
    if len(test_records) != SOURCE_ROWS:
        raise ValueError("Unexpected test row count")
    if {r["text"] for r in source_records} & {r["text"] for r in test_records}:
        raise ValueError("A source text appears in competition test")
    rng.shuffle(source_records)
    test_records = test_records.copy()
    rng.shuffle(test_records)
    cutoff = int(SOURCE_ROWS * 0.75)
    train_texts = [r["text"] for r in source_records[:cutoff] + test_records[:cutoff]]
    holdout_texts = [r["text"] for r in source_records[cutoff:] + test_records[cutoff:]]
    y_train = torch.tensor([0] * cutoff + [1] * cutoff, dtype=torch.float32)
    y_holdout = np.array([0] * (SOURCE_ROWS - cutoff) + [1] * (SOURCE_ROWS - cutoff))
    vocabulary = _vocabulary(train_texts)
    x_train, len_train = _encode(train_texts, vocabulary)
    x_holdout, len_holdout = _encode(holdout_texts, vocabulary)
    model = DomainGRU(len(vocabulary) + 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(y_train))
        epoch_loss = 0.0
        for positions in order.split(64):
            optimizer.zero_grad()
            logits = model(x_train[positions], len_train[positions])
            loss = loss_fn(logits, y_train[positions])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(positions)
        losses.append(epoch_loss / len(y_train))
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_holdout, len_holdout)).numpy()
    auc = roc_auc_score(y_holdout, probabilities)
    balanced_accuracy = balanced_accuracy_score(y_holdout, probabilities >= 0.5)
    bootstrap = np.random.default_rng(seed)
    intervals = []
    for _ in range(500):
        indices = bootstrap.integers(0, len(y_holdout), len(y_holdout))
        if len(set(y_holdout[indices])) == 2:
            intervals.append(roc_auc_score(y_holdout[indices], probabilities[indices]))
    return {"source": source, "seed": seed, "source_rows": SOURCE_ROWS,
            "train_per_class": cutoff, "holdout_per_class": SOURCE_ROWS - cutoff,
            "max_chars": MAX_CHARS, "vocabulary_size": len(vocabulary),
            "epochs": EPOCHS, "train_loss_by_epoch": losses,
            "auc": auc, "auc_interval_95": np.quantile(intervals, [0.025, 0.975]).tolist(),
            "balanced_accuracy": balanced_accuracy,
            "holdout_probabilities": probabilities.tolist(), "holdout_labels": y_holdout.tolist(),
            "holdout_texts": holdout_texts}


def run_both() -> Path:
    audit = json.loads((OUTPUT / "audit.json").read_text(encoding="utf-8"))
    result = {source: run_one(audit, source) for source in ("uk", "ru")}
    target = OUTPUT / "adversarial.json"
    target.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run_both())
