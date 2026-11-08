"""Read pretrained fastText vectors without downloading the whole files.

The .vec files list words from most to least frequent, so reading the first
`top_k` lines over the network is enough. The result is cached on disk.
"""

import gzip
import os
import urllib.request

import numpy as np

from .config import CACHE_DIR

BASE_URL = os.getenv("NER_VECTOR_BASE", "https://dl.fbaipublicfiles.com/fasttext")
SOURCES = {
    # One shared 300-dimensional space for both languages; lowercase Wikipedia vectors.
    "aligned": "vectors-aligned/wiki.{language}.align.vec",
    # Common Crawl vectors; larger coverage, but each language has its own unaligned space.
    "crawl": "vectors-crawl/cc.{language}.300.vec.gz",
}


def _read_top_words(url, top_k):
    with urllib.request.urlopen(url) as response:
        stream = gzip.open(response, "rt", encoding="utf-8") if url.endswith(".gz") else None
        lines = stream if stream else (raw.decode("utf-8", "ignore") for raw in response)
        words, vectors = [], []
        for index, line in enumerate(lines):
            if index == 0:
                continue  # header: number of words and dimension
            if index > top_k:
                break
            parts = line.rstrip().split(" ")
            if len(parts) == 301:
                words.append(parts[0])
                vectors.append(np.array(parts[1:], dtype=np.float32))
    return words, np.stack(vectors)


def load_vectors(name, top_k):
    """Return lowercase (words, matrix), since pretrained runs lowercase their tokens.

    When a word has several forms, the most frequent wins; between languages, Ukrainian wins.
    """
    path = CACHE_DIR / f"vectors_{name}_{top_k}.npz"
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        return saved["words"].tolist(), saved["vectors"]
    merged = {}
    for language in ("uk", "ru"):
        url = f"{BASE_URL}/{SOURCES[name].format(language=language)}"
        print(f"Reading the top {top_k:,} words from {url}", flush=True)
        words, vectors = _read_top_words(url, top_k)
        for word, vector in zip(words, vectors):
            merged.setdefault(word.lower(), vector)
    words = list(merged)
    matrix = np.stack([merged[word] for word in words])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, words=np.array(words), vectors=matrix)
    return words, matrix


def embedding_matrix(tokenizer, words, vectors, seed):
    """Vectors for the tokenizer vocabulary; `known` marks rows filled from pretrained data."""
    lookup = {word: index for index, word in enumerate(words)}
    vocabulary = tokenizer.get_vocab()
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, float(vectors.std()), (len(vocabulary), vectors.shape[1]))
    known = np.zeros(len(vocabulary), dtype=bool)
    for word, position in vocabulary.items():
        match = lookup.get(word, lookup.get(word.lower()))
        if match is not None:
            matrix[position] = vectors[match]
            known[position] = True
    matrix[0] = 0  # padding
    return matrix.astype(np.float32), known
