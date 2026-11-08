"""The running set of decisions. Edit DECISIONS after each notebook.

It starts from Herman's word BiLSTM recipe, with more rows and early stopping so
that screening runs are not limited by underfitting. Every notebook builds on
base_config(), and the final Colab notebook will start from the same values.
"""

from dataclasses import replace

from .config import Config

DECISIONS = {
    # After 01_tokenization_embeddings: word tokens, frozen fastText aligned vectors.
    # Lowercase is needed because the vectors are lowercase; the case feature restores
    # capitalization. A character encoder added nothing on top of frozen vectors (0.813 vs 0.832).
    "tokenizer": "word", "vocab_size": 16_000, "lowercase": True, "case_feature": True,
    "pretrained": "aligned", "embedding_mode": "frozen", "pretrained_top_k": 200_000,
    # Several 01 runs stopped at the 6-epoch limit while still improving; early stopping ends the rest sooner.
    "max_epochs": 10, "patience": 3,  # one 01 run stopped at epoch 3 because of a noisy dip in validation F1
    # After 02_architecture: GRU and LSTM tie, so the cheaper GRU; a CRF gained +0.023 on all three folds.
    # One layer, 96 units and bidirectional stay at their defaults; larger networks gave no gain.
    "rnn": "gru", "crf": True,
    # After 03_training_setup: nothing beat the defaults by more than the seed spread (0.013), so they stay.
    "optimizer": "adamw", "lr": 0.003, "weight_decay": 0.01, "scheduler": "none",
    "batch_size": 32, "dropout": 0.15, "grad_clip": 1.0,
    # After 04_data_strategy: more data is the biggest lever (0.856 -> 0.885 from 10k to 100k rows per
    # language, still rising); the language mix is a tie, so 50/50. Newline/URL/emoji tokens gained +0.017
    # on synthetic long posts (one seed, the least certain choice). Synthetic posts and address repeats gave nothing.
    "uk_rows": 100_000, "ru_rows": 100_000, "structure_tokens": True,
}


def base_config(**overrides) -> Config:
    """Current decisions plus the changes one experiment wants to test."""
    return replace(Config(), **{**DECISIONS, **overrides})


# After 05_decoding_logits_ensembling (inner validation, fold 0): the CRF bias is best at 0 (F1 0.895; +1.0 gave
# 0.879), strict/lenient merge are identical, joining hyphenated parts and trimming house numbers lose 0.01-0.02.
# Averaging the log-probabilities and transitions of 3 seeds gave 0.893 vs 0.883 for a single seed; votes tie it.
# Adding an LSTM member gave 0.898 (inside the noise, but free diversity); a BPE member hurt (0.889).
DECODING = {
    "merge": "lenient", "drop_empty": True, "trim_number": False, "bias": 0.0,
    "ensemble": "average",  # average the saved logits of all final models (3+ seeds, optionally one LSTM)
}
