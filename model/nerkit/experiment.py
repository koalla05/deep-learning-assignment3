"""Run one configuration from data selection to saved logits and scores.

Finished runs are stored in results/<notebook>.jsonl, so rerunning a notebook skips
them and an interrupted notebook continues where it stopped.
"""

import json
import logging
import os
import random
import time
import warnings
from dataclasses import asdict, replace
from functools import lru_cache

import lightning as L
import numpy as np
import pandas as pd
import torch

from .artifacts import save_logits
from .config import ARTIFACTS_DIR, RESULTS_DIR, SEED, SMOKE, Config
from .data import (check_no_leak, has_address, load_labeling_sample, load_rows, load_test,
                   make_pseudo_posts, split_fold, split_inner, take_rows)
from .decisions import base_config
from .decoding import decode_row
from .embeddings import embedding_matrix, load_vectors
from .engine import Collator, TaggerModule, fit, predict_logits
from .metrics import evaluate, seen_names, unseen_recall
from .models import SequenceTagger
from .tokenization import build_char_vocabulary, encode_rows, fit_tokenizer, make_windows

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module=r"lightning\.pytorch\..*")  # advice about GPUs and workers
DEVICE_FILE = RESULTS_DIR / "device.txt"
POST_EVALUATION_ROWS = 4_000


def get_device(name=None):
    """Explicit name, else NER_DEVICE, else the winner saved by notebook 00, else CPU."""
    name = name or os.getenv("NER_DEVICE")
    if name is None and DEVICE_FILE.exists():
        name = DEVICE_FILE.read_text().strip()
    return torch.device(name or "cpu")


@lru_cache(maxsize=1)
def cached_rows():
    return load_rows()


def select_training(config):
    """Supervised rows and the pool for fitting the tokenizer, with leakage checks."""
    rows = cached_rows()
    pool, heldout = split_fold(rows, config.fold)
    extra = [row for row in rows if row["partition"] == "extra"]
    uk_rows, ru_rows = config.uk_rows, config.ru_rows
    if SMOKE:
        uk_rows, ru_rows = min(uk_rows, 200), min(ru_rows, 200)
    pool_fit, inner = split_inner(pool)
    selected = (take_rows(pool_fit, extra, "uk", uk_rows)
                + take_rows(pool_fit, extra, "ru", ru_rows))
    check_no_leak(selected, heldout)
    check_no_leak(selected, inner)
    fitting = {(row["source"], row["row_id"]): row for row in pool_fit + selected}
    return selected, list(fitting.values()), inner, heldout


def evaluation_sets(config, heldout):
    """The held-out fold, synthetic long posts made from it, and the competition files."""
    posts_from = sorted(heldout, key=lambda row: (row["source"], row["row_id"]))
    rng = random.Random(config.fold)
    rng.shuffle(posts_from)
    posts = make_pseudo_posts(posts_from[:POST_EVALUATION_ROWS], seed=config.fold)
    return dict(heldout=heldout, posts=posts, labeling=load_labeling_sample(), test=load_test())


def run_experiment(name, config, notebook, device=None, force=False):
    """Train, predict every split, save logits and return the result record."""
    path = RESULTS_DIR / f"{notebook}.jsonl"
    previous = read_results(notebook, raw=True)
    done = {record["name"]: record for record in previous if record["key"] == config.key()}
    if name in done and not force:
        print(f"{name}: already finished, F1 mix {done[name]['scores']['heldout']['mix']['f1']:.3f}")
        return done[name]
    if SMOKE:
        config = replace(config, max_epochs=min(config.max_epochs, 2), pretrained_top_k=50)
    device = get_device(device)
    L.seed_everything(config.seed, workers=True, verbose=False)
    started = time.monotonic()

    selected, fitting, inner, heldout = select_training(config)
    training = selected
    if config.address_boost > 1:
        training = training + [row for row in selected if has_address(row)] * (config.address_boost - 1)
    if config.pseudo_post_fraction:
        training = make_pseudo_posts(training, config.seed, config.pseudo_post_fraction)

    words, vectors, extra_words = None, None, ()
    if config.pretrained != "none":
        words, vectors = load_vectors(config.pretrained, config.pretrained_top_k)
        extra_words = words
    tokenizer = fit_tokenizer((row["text"] for row in fitting), config.tokenizer, config.vocab_size,
                              config.lowercase, config.structure_tokens, extra_words)
    pretrained = None
    if config.pretrained != "none":
        pretrained, _ = embedding_matrix(tokenizer, words, vectors, config.seed)
    char_vocabulary = build_char_vocabulary(fitting) if config.char_hidden else None
    collator = Collator(char_vocabulary, config.case_feature)

    train_encoded, skipped = encode_rows(tokenizer, training, training=True)
    train_windows = make_windows(train_encoded, training, training=True)
    inner_encoded, _ = encode_rows(tokenizer, inner)
    inner_windows = make_windows(inner_encoded, inner)

    tagger = SequenceTagger(config, tokenizer.get_vocab_size(),
                            len(char_vocabulary) + 2 if char_vocabulary else 0, pretrained)
    module = TaggerModule(tagger, config, inner, inner_encoded, inner_windows, collator)
    train_started = time.monotonic()
    gradient_records = fit(module, config, train_windows, inner_windows, collator, device)
    train_seconds = time.monotonic() - train_started

    tagger.to(device)
    crf = tagger.transitions()
    meta = dict(fold=config.fold, seed=config.seed, config=asdict(config),
                crf={k: v.tolist() for k, v in crf.items()} if crf else None)
    scores = {}
    sets = dict(inner=inner, **evaluation_sets(config, heldout))
    seen = seen_names(selected)
    for split, rows in sets.items():
        encoded, _ = encode_rows(tokenizer, rows)
        windows = make_windows(encoded, rows)
        logits = predict_logits(tagger, windows, encoded, collator, device)
        save_logits(config.key(), split, rows, encoded, logits, meta)
        if split == "test":
            continue
        predictions = [decode_row(values, enc.offsets, rows[enc.row_index]["text"], crf=crf)
                       for values, enc in zip(logits, encoded)]
        ordered = [rows[enc.row_index] for enc in encoded]
        scores[split] = evaluate(ordered, predictions)
        if split == "heldout":
            scores["unseen_recall"] = unseen_recall(ordered, predictions, seen)

    record = dict(name=name, key=config.key(), config=asdict(config), scores=scores,
                  history=module.history, gradients=gradient_records,
                  parameters=sum(p.numel() for p in tagger.parameters()),
                  training_rows=len(train_encoded), skipped_rows=skipped,
                  training_tokens=int(sum(len(window.ids) for window in train_windows)),
                  vocabulary=tokenizer.get_vocab_size(), device=device.type,
                  train_seconds=train_seconds, total_seconds=time.monotonic() - started)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tokenizer.save(str(ARTIFACTS_DIR / config.key() / "tokenizer.json"))
    torch.save(tagger.cpu().state_dict(), ARTIFACTS_DIR / config.key() / "model.pt")
    print(f"{name}: F1 mix {scores['heldout']['mix']['f1']:.3f}, "
          f"uk {scores['heldout']['uk']['f1']:.3f}, ru {scores['heldout']['ru']['f1']:.3f}, "
          f"{record['total_seconds']:.0f} s on {device.type}", flush=True)
    return record


def run_grid(notebook, experiments, seeds=(SEED,), device=None):
    """Run named overrides of the current decisions, each with every seed."""
    return [run_experiment(name, base_config(**overrides, seed=seed), notebook, device)
            for name, overrides in experiments.items() for seed in seeds]


def confirm(notebook, names, folds=(1, 2), seeds=(SEED,), device=None):
    """Repeat finished configurations on other folds, unchanged apart from fold and seed."""
    configs = {}
    for record in read_results(notebook, raw=True):
        configs.setdefault(record["name"], Config(**record["config"]))
    return [run_experiment(name, replace(configs[name], fold=fold, seed=seed), notebook, device)
            for name in names for fold in folds for seed in seeds]


def top_names(notebook, count, fold=0):
    """The best configurations by mean F1 mix on one fold."""
    return summarize(notebook, fold=fold).head(count).index.tolist()


def records_named(notebook, names, fold=0, seed=SEED):
    """Full records in the order of `names`, for curves and gradient charts."""
    found = {record["name"]: record for record in read_results(notebook, raw=True)
             if record["config"]["fold"] == fold and record["config"]["seed"] == seed}
    return [found[name] for name in names if name in found]


def read_results(notebook, raw=False):
    """Finished runs of one notebook, as records or as a comparison table."""
    path = RESULTS_DIR / f"{notebook}.jsonl"
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if raw:
        return records
    table = []
    for record in records:
        scores = record["scores"]
        heldout = scores["heldout"]
        table.append(dict(
            name=record["name"], fold=record["config"]["fold"], seed=record["config"]["seed"],
            mix_f1=heldout["mix"]["f1"], uk_f1=heldout["uk"]["f1"], ru_f1=heldout["ru"]["f1"],
            precision=heldout["mix"]["precision"], recall=heldout["mix"]["recall"],
            uk_unseen_recall=scores["unseen_recall"].get("uk", np.nan),
            posts_f1=scores["posts"]["mix"]["f1"], labeling_f1=scores["labeling"]["pooled"]["f1"],
            epochs=len(record["history"]), parameters=record["parameters"],
            minutes=record["total_seconds"] / 60))
    return pd.DataFrame(table)


def summarize(notebook, baseline=None, fold=0):
    """Mean and spread per configuration, best F1 mix first. fold=None pools all folds."""
    table = read_results(notebook)
    if table.empty:
        return table
    if fold is not None:
        table = table[table["fold"] == fold]
    summary = table.groupby("name").agg(
        runs=("mix_f1", "size"), mix_f1=("mix_f1", "mean"), spread=("mix_f1", lambda values: values.max() - values.min()),
        uk_f1=("uk_f1", "mean"), ru_f1=("ru_f1", "mean"), precision=("precision", "mean"),
        recall=("recall", "mean"), uk_unseen_recall=("uk_unseen_recall", "mean"),
        posts_f1=("posts_f1", "mean"), labeling_f1=("labeling_f1", "mean"),
        minutes=("minutes", "mean")).sort_values("mix_f1", ascending=False)
    if baseline in summary.index:
        summary.insert(1, "vs_baseline", summary["mix_f1"] - summary.loc[baseline, "mix_f1"])
    return summary.round(3)
