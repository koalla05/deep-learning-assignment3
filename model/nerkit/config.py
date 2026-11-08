"""Paths and the experiment configuration used by every notebook.

Paths can be overridden with environment variables, so the same code runs on the
MacBook and later on Colab.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("NER_DATA_DIR", ROOT / "data" / "kse-ua-location-extraction-2026"))
if not (DATA_DIR / "test.csv").exists() and (DATA_DIR / DATA_DIR.name / "test.csv").exists():
    DATA_DIR = DATA_DIR / DATA_DIR.name  # the Kaggle archive unpacks into a folder of the same name
MANIFEST_DIR = Path(os.getenv("NER_MANIFEST_DIR", ROOT / "eda" / "output"))
WORK_DIR = Path(os.getenv("NER_WORK_DIR", ROOT / "model"))
CACHE_DIR = WORK_DIR / "cache"
RESULTS_DIR = WORK_DIR / "results"
ARTIFACTS_DIR = WORK_DIR / "artifacts"

SEED = 3012026
SEEDS = (3012026, 3012027, 3012028)
TAGS = ("O", "B", "I")

# NER_SMOKE=1 shrinks every run; it only exists to test the notebooks end to end.
SMOKE = os.getenv("NER_SMOKE") == "1"


@dataclass(frozen=True)
class Config:
    """One experiment. Two runs with the same values are treated as the same run."""

    # Which data the model trains on.
    fold: int = 0
    uk_rows: int = 10_000
    ru_rows: int = 10_000
    pseudo_post_fraction: float = 0.0  # share of training rows joined into multi-line posts
    address_boost: int = 1  # how many times rows with street/landmark mentions are repeated

    # Tokenization and embeddings.
    tokenizer: str = "word"  # word | bpe
    vocab_size: int = 16_000
    lowercase: bool = False
    structure_tokens: bool = False  # keep newlines, URLs and emoji as special tokens
    pretrained: str = "none"  # none | aligned | crawl
    pretrained_top_k: int = 200_000  # most frequent words read per language
    embedding_mode: str = "random"  # random | frozen | finetune

    # Network.
    rnn: str = "lstm"  # gru | lstm
    embedding_dim: int = 96  # replaced by 300 when pretrained vectors are used
    hidden: int = 96
    layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.15
    char_hidden: int = 0  # 0 disables the character BiGRU
    case_feature: bool = False
    crf: bool = False

    # Optimization.
    optimizer: str = "adamw"  # adamw | adam | sgd | rmsprop
    lr: float = 0.003
    weight_decay: float = 0.01
    scheduler: str = "none"  # none | onecycle | cosine | plateau
    batch_size: int = 32
    max_epochs: int = 6
    patience: int = 2  # 0 disables early stopping
    use_best_epoch: bool = True  # restore the best epoch on the inner validation split
    grad_clip: float = 1.0
    weight_o: float = 1.0
    weight_b: float = 3.0
    weight_i: float = 2.0
    seed: int = SEED

    def key(self) -> str:
        """A short stable identifier of this configuration."""
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:12]
