"""Shared recurrent encoder and bounded binary domain-classification training."""

import time

import numpy as np
import torch
from torch import nn


class DomainWordRecurrent(nn.Module):
    """The extraction encoder dimensions, with one document-level binary head."""

    def __init__(self, vocabulary_size, architecture='gru'):
        super().__init__()
        if architecture not in ('gru', 'lstm'):
            raise ValueError(architecture)
        self.emb = nn.Embedding(vocabulary_size, 96, padding_idx=0)
        self.drop = nn.Dropout(.15)
        recurrent = nn.GRU if architecture == 'gru' else nn.LSTM
        self.recurrent = recurrent(96, 96, batch_first=True, bidirectional=True)
        self.out = nn.Linear(192, 1)

    def forward(self, x, lengths):
        embedded = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.recurrent(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        return self.out(self.drop(torch.cat([hidden[-2], hidden[-1]], dim=-1))).squeeze(-1)


def encode(texts, tokenizer):
    values = [tokenizer.encode(text).ids[:256] for text in texts]
    lengths = torch.tensor([max(1, len(value)) for value in values])
    matrix = torch.zeros((len(values), int(max(lengths))), dtype=torch.long)
    for i, value in enumerate(values):
        if value:
            matrix[i, :len(value)] = torch.tensor(value)
    return matrix, lengths


def fit(train, hold, labels, tokenizer, architecture, seed):
    """Fit six fixed epochs; preserve historical torch-RNG minibatch ordering."""
    torch.manual_seed(seed)
    x, lengths = encode(train, tokenizer)
    x_hold, lengths_hold = encode(hold, tokenizer)
    labels = torch.tensor(labels, dtype=torch.float32)
    model = DomainWordRecurrent(tokenizer.get_vocab_size(), architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    loss_fn = nn.BCEWithLogitsLoss()
    history, batch_orders = [], []
    started = time.monotonic()
    for epoch in range(6):
        model.train()
        order = torch.randperm(len(train))
        batch_orders.append(order.tolist())
        loss_total = correct = 0
        for positions in order.split(32):
            optimizer.zero_grad()
            logits = model(x[positions], lengths[positions])
            loss = loss_fn(logits, labels[positions])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item() * len(positions)
            correct += int(((logits.detach() >= 0) == labels[positions].bool()).sum())
        history.append({'epoch': epoch + 1, 'loss': loss_total / len(train),
                        'online_training_accuracy': correct / len(train)})
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_hold, lengths_hold)).numpy()
    return model, probabilities, {'history': history, 'batch_orders': batch_orders,
        'parameters': sum(p.numel() for p in model.parameters()),
        'seconds': time.monotonic() - started}
