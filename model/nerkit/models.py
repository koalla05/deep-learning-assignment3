"""The recurrent tagger: embeddings, optional character BiGRU, GRU/LSTM, optional CRF."""

import torch
from torch import nn
from torchcrf import CRF


class CharEncoder(nn.Module):
    """A small BiGRU over the characters of each distinct token in the batch."""

    def __init__(self, vocabulary_size, hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 32, padding_idx=0)
        self.rnn = nn.GRU(32, hidden, batch_first=True, bidirectional=True)

    def forward(self, char_ids, char_lengths, token_index):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(char_ids), char_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.rnn(packed)
        distinct = torch.cat([state[-2], state[-1]], dim=-1)
        return distinct[token_index]


class SequenceTagger(nn.Module):
    def __init__(self, config, vocabulary_size, char_vocabulary_size=0, pretrained=None):
        super().__init__()
        self.config = config
        self.embedding = self.corrections = None
        if pretrained is None:
            dimension = config.embedding_dim
            self.embedding = nn.Embedding(vocabulary_size, dimension, padding_idx=0)
        else:
            # A trainable table of this size would make every optimizer step slow. The
            # table stays fixed; "finetune" learns corrections for the frequent words,
            # which have the lowest ids because the tokenizer lists them first.
            dimension = pretrained.shape[1]
            self.register_buffer("pretrained_table", torch.as_tensor(pretrained), persistent=False)
            if config.embedding_mode == "finetune":
                self.corrections = nn.Embedding(min(config.vocab_size, vocabulary_size), dimension,
                                                padding_idx=0)
                nn.init.zeros_(self.corrections.weight)
        size = dimension
        self.char_encoder = None
        if config.char_hidden:
            self.char_encoder = CharEncoder(char_vocabulary_size, config.char_hidden)
            size += 2 * config.char_hidden
        self.case_embedding = None
        if config.case_feature:
            self.case_embedding = nn.Embedding(6, 8, padding_idx=0)
            size += 8
        self.dropout = nn.Dropout(config.dropout)
        recurrent = {"gru": nn.GRU, "lstm": nn.LSTM}[config.rnn]
        self.rnn = recurrent(size, config.hidden, num_layers=config.layers, batch_first=True,
                             bidirectional=config.bidirectional,
                             dropout=config.dropout if config.layers > 1 else 0.0)
        self.output = nn.Linear(config.hidden * (2 if config.bidirectional else 1), 3)
        self.crf = CRF(3, batch_first=True) if config.crf else None

    def _embed(self, ids):
        if self.embedding is not None:
            return self.embedding(ids)
        vectors = self.pretrained_table[ids]
        if self.corrections is not None:
            frequent = ids.where(ids < self.corrections.num_embeddings, torch.zeros_like(ids))
            vectors = vectors + self.corrections(frequent)
        return vectors

    def forward(self, batch):
        """Per-token scores for O, B and I (the logits saved for ensembling)."""
        parts = [self._embed(batch["ids"])]
        if self.char_encoder is not None:
            parts.append(self.char_encoder(batch["char_ids"], batch["char_lengths"],
                                           batch["token_index"]))
        if self.case_embedding is not None:
            parts.append(self.case_embedding(batch["case_ids"]))
        inputs = self.dropout(torch.cat(parts, dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, batch["lengths"].cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=batch["ids"].shape[1])
        return self.output(self.dropout(outputs))

    def loss(self, logits, batch, class_weights):
        tags = batch["tags"]
        if self.crf is None:
            return nn.functional.cross_entropy(logits.reshape(-1, 3), tags.reshape(-1),
                                               weight=class_weights, ignore_index=-100)
        positions = torch.arange(tags.shape[1], device=tags.device).unsqueeze(0)
        mask = positions < batch["lengths"].to(tags.device).unsqueeze(1)
        known = tags.clamp(min=0)  # tags ignored at window edges count as O on the CRF path
        return -self.crf(logits, known, mask=mask, reduction="mean")

    def transitions(self):
        """CRF scores as plain arrays for decoding, or None without a CRF."""
        if self.crf is None:
            return None
        return dict(start=self.crf.start_transitions.detach().cpu().numpy(),
                    end=self.crf.end_transitions.detach().cpu().numpy(),
                    transitions=self.crf.transitions.detach().cpu().numpy())
