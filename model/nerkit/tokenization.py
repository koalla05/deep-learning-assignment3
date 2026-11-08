"""Word and BPE tokenizers, BIO targets and overlapping windows."""

from dataclasses import dataclass

import numpy as np
from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers, trainers

WINDOW = 256
STRIDE = 192
SPECIAL_TOKENS = ["[PAD]", "[UNK]"]
URL_PATTERN = r"https?://\S+|www\.\S+"
EMOJI_PATTERN = r"[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]"


def _prepare(tokenizer, lowercase, structure_tokens):
    """Normalizers keep character offsets aligned with the original text."""
    steps = []
    if structure_tokens:
        # The default pre-tokenizer drops newlines, so they become visible symbols.
        steps += [normalizers.Replace(Regex(URL_PATTERN), " ⌂ "),
                  normalizers.Replace(Regex(EMOJI_PATTERN), " ☺ "),
                  normalizers.Replace(Regex(r"\n"), " ⏎ ")]
    if lowercase:
        steps.append(normalizers.Lowercase())
    if steps:
        tokenizer.normalizer = normalizers.Sequence(steps)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def fit_tokenizer(texts, kind="word", vocab_size=16_000, lowercase=False,
                  structure_tokens=False, extra_words=()):
    """Fit on training text only. `extra_words` are pretrained words added to the vocabulary."""
    if kind == "word":
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        trainer = trainers.WordLevelTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                            show_progress=False)
    elif kind == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                      show_progress=False)
    else:
        raise ValueError(kind)
    _prepare(tokenizer, lowercase, structure_tokens)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    if extra_words:
        if kind != "word":
            raise ValueError("Pretrained word vectors need the word tokenizer")
        vocabulary = dict(tokenizer.get_vocab())
        for word in extra_words:
            vocabulary.setdefault(word, len(vocabulary))
        tokenizer = _prepare(Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]")),
                             lowercase, structure_tokens)
    return tokenizer


@dataclass
class Encoded:
    """A whole text after tokenization."""

    row_index: int
    ids: np.ndarray
    offsets: np.ndarray  # shape (tokens, 2), character positions in the original text
    tags: np.ndarray  # 0=O, 1=B, 2=I, -100=ignored by the loss


def bio_tags(offsets, markers):
    """A token inside a location gets I, or B when it starts that location."""
    tags, previous = [], None
    for start, end in offsets:
        owner = next((i for i, (a, b) in enumerate(markers) if start < b and a < end), None)
        tags.append(0 if owner is None else 1 if owner != previous else 2)
        previous = owner
    return tags


def is_representable(offsets, markers):
    """False when a location boundary falls inside a token."""
    starts = {a for a, _ in offsets}
    ends = {b for _, b in offsets}
    return all(a in starts and b in ends for a, b in markers)


def encode_rows(tokenizer, rows, training=False):
    """Tokenize rows. In training, rows with unrepresentable locations are skipped."""
    encoded = []
    skipped = 0
    outputs = tokenizer.encode_batch([row["text"] for row in rows])
    for index, (row, output) in enumerate(zip(rows, outputs)):
        offsets = output.offsets
        if training and not is_representable(offsets, row["markers"]):
            skipped += 1
            continue
        encoded.append(Encoded(index, np.array(output.ids, dtype=np.int64),
                               np.array(offsets, dtype=np.int32).reshape(-1, 2),
                               np.array(bio_tags(offsets, row["markers"]), dtype=np.int64)))
    return encoded, skipped


@dataclass
class Window:
    """Up to 256 tokens of one text; this is what the network sees."""

    item: int  # position in the encoded list
    start: int  # first token of the window in the whole text
    ids: np.ndarray
    tags: np.ndarray
    surfaces: list


def make_windows(encoded, rows, training=False):
    """Split long texts. In training, locations cut by a window edge are ignored by the loss."""
    windows = []
    for item, enc in enumerate(encoded):
        length = len(enc.ids)
        text = rows[enc.row_index]["text"]
        for start in range(0, max(length, 1), STRIDE):
            end = min(start + WINDOW, length)
            if start >= end:
                break
            tags = enc.tags[start:end].copy()
            if training:
                if start > 0 and enc.tags[start] == 2:
                    position = 0
                    while position < len(tags) and tags[position] == 2:
                        tags[position] = -100
                        position += 1
                if end < length and enc.tags[end] == 2:
                    position = len(tags) - 1
                    while position >= 0 and tags[position] in (1, 2):
                        stop = tags[position] == 1
                        tags[position] = -100
                        position -= 1
                        if stop:
                            break
            surfaces = [text[a:b] for a, b in enc.offsets[start:end]]
            windows.append(Window(item, start, enc.ids[start:end], tags, surfaces))
            if end == length:
                break
    return windows


def build_char_vocabulary(rows):
    """Character ids from training text; 0 is padding and 1 is unknown."""
    characters = sorted({character for row in rows for character in row["text"]})
    return {character: index + 2 for index, character in enumerate(characters)}


def case_shape(surface):
    """0 padding, 1 lower, 2 Title, 3 UPPER, 4 digit, 5 other."""
    if surface.isdigit():
        return 4
    if surface.isupper():
        return 3
    if surface[:1].isupper():
        return 2
    if surface.islower():
        return 1
    return 5
