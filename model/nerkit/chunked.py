"""Predict long posts piece by piece with already trained fold models.

The models were trained on single sentences (median 14 words, no line breaks). Competition posts are
longer (median 36 words), have line breaks and start with emoji bullets. Splitting a post into lines or
sentences, predicting each piece alone and joining the strings brings the input closer to the training data.
"""

import re

import numpy as np
import torch
from tokenizers import Tokenizer

from .config import ARTIFACTS_DIR, Config
from .decoding import decode_row
from .embeddings import embedding_matrix, load_vectors
from .engine import Collator, predict_logits
from .experiment import get_device
from .models import SequenceTagger
from .tokenization import encode_rows, make_windows

# A full stop after one of these does not end a sentence ("вул. Хрещатик", "м. Київ", "млн грн").
ABBREVIATIONS = {"вул", "ул", "м", "г", "обл", "р", "рр", "пр", "просп", "бул", "пров", "пл", "ім", "тис", "млн",
                 "млрд", "грн", "т", "напр", "див", "с", "смт", "ст", "буд", "кв", "км", "ж", "р-н", "рн", "тел",
                 "руб", "коп", "св", "проф", "акад", "ген", "полк", "лют", "ін"}
SENTENCE_END = re.compile(r"[.!?…]+[\"»”’)\]]*[ \t]+")
SENTENCE_START = re.compile(r"[A-ZА-ЯІЇЄҐЁ0-9\"«“„(\[]")
MODES = ("none", "lines", "sentences")


def _sentence_cuts(line):
    cuts = []
    for match in SENTENCE_END.finditer(line):
        following = line[match.end():match.end() + 1]
        if not following or not SENTENCE_START.match(following):
            continue
        word = re.search(r"[\w'’-]+$", line[:match.start()])
        word = word.group(0).lower() if word else ""
        if word in ABBREVIATIONS or len(word) == 1:
            continue
        cuts.append(match.end())
    return cuts


def split_text(text, mode):
    """Character spans (start, end) of the pieces to predict separately.

    none       the whole text
    lines      every non-empty line
    sentences  every line, cut further after . ! ? followed by a capital letter (abbreviations are skipped)
    """
    if mode == "none":
        return [(0, len(text))]
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        start, end = line.span()
        edges = [start]
        if mode == "sentences":
            edges += [start + cut for cut in _sentence_cuts(line.group(0))]
        edges.append(end)
        for first, last in zip(edges, edges[1:]):
            piece = text[first:last]
            first += len(piece) - len(piece.lstrip())
            last -= len(piece) - len(piece.rstrip())
            if re.search(r"\w", text[first:last]):
                spans.append((first, last))
    return spans


def load_trained(record):
    """Rebuild a fold model from its saved tokenizer and weights (record: one line of the results file)."""
    config = Config(**record["config"])
    folder = ARTIFACTS_DIR / record["key"]
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    pretrained = None
    if config.pretrained != "none":
        words, vectors = load_vectors(config.pretrained, config.pretrained_top_k)
        pretrained, _ = embedding_matrix(tokenizer, words, vectors, config.seed)
    tagger = SequenceTagger(config, tokenizer.get_vocab_size(), 0, pretrained)
    tagger.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
    tagger.eval()
    return dict(config=config, tokenizer=tokenizer, tagger=tagger,
                collator=Collator(None, config.case_feature), crf=tagger.transitions())


def make_chunks(rows, mode):
    """Pieces of all rows as small rows of their own, and the index of the row each piece came from."""
    chunks, owners = [], []
    for index, row in enumerate(rows):
        for start, end in split_text(row["text"], mode):
            chunks.append(dict(source=row["source"], row_id=f"{index}:{start}", text=row["text"][start:end], markers=[]))
            owners.append(index)
    return chunks, owners


def infer(model, chunks, device=None):
    """Logits for each chunk as {chunk position: (offsets, logits)}."""
    device = device or get_device()
    model["tagger"].to(device)
    encoded, _ = encode_rows(model["tokenizer"], chunks)
    windows = make_windows(encoded, chunks)
    logits = predict_logits(model["tagger"], windows, encoded, model["collator"], device)
    return {enc.row_index: (enc.offsets, values) for enc, values in zip(encoded, logits)}


def decode_chunks(loaded, chunks, owners, row_count, crf, **options):
    """Location strings per original row: the strings of all its pieces, in order, without repeats."""
    strings = [[] for _ in range(row_count)]
    for position, chunk in enumerate(chunks):
        if position not in loaded:
            continue
        offsets, logits = loaded[position]
        strings[owners[position]] += decode_row(logits, offsets, chunk["text"], crf=crf, **options)
    return [list(dict.fromkeys(values)) for values in strings]
