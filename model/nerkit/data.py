"""Join the raw CSVs with the frozen manifests and select rows without leakage.

Rows carry a `partition`:
  cv        the 63,388 frozen cross-validation rows (with a `fold`)
  extra     role_v2='train' rows outside CV; their groups are disjoint from all folds
  dev_clean role_v2='development' rows outside CV; a fixed holdout for CV-trained models

`final` and `exclude_*` rows are never read into the cache.
"""

import ast
import csv
import hashlib
import random
import re

import duckdb
import pandas as pd

from eda.core.analysis import parse_markers

from .config import CACHE_DIR, DATA_DIR, MANIFEST_DIR, SEED

ROWS_FILE = CACHE_DIR / "rows.parquet"
ADDRESS_CUE = re.compile(
    r"\b(вул|вулиц|улиц|ул|просп|проспект|пров|провулок|переул|бульвар|бул|площ|пл|шосе|шоссе"
    r"|набережн|узвіз|спуск|міст|мост|тц|трц|метро|парк|сквер|район|мікрорайон|микрорайон)",
    re.IGNORECASE,
)
FOOTERS = (
    "👉 Підписуйтесь на наш канал: https://t.me/news_channel",
    "Надіслати новину: @news_bot",
    "❗️Підписатися | Прислати новину",
    "Подписывайтесь на наш канал: https://t.me/news_channel",
)


def build_cache(extra_per_source=200_000, dev_per_source=20_000, force=False):
    """Create cache/rows.parquet once; the first run scans about nine million CSV rows."""
    if ROWS_FILE.exists() and not force:
        return ROWS_FILE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect()
    db.execute("SET threads=1")  # preserves CSV order, which defines row_id
    db.execute("SET memory_limit='8GB'")
    db.execute(f"SET temp_directory='{(CACHE_DIR / 'duckdb_tmp').as_posix()}'")
    roles = (MANIFEST_DIR / "role_manifest_v2.parquet").as_posix()
    folds = (MANIFEST_DIR / "cv_manifest_v1.parquet").as_posix()
    db.execute(f"""CREATE TEMP TABLE chosen AS
        WITH active AS (
            SELECT r.source, r.row_id, r.doc_id, r.group_id, r.role_v2, r.text_key,
                   r.has_location, f.fold
            FROM read_parquet('{roles}') r
            LEFT JOIN read_parquet('{folds}') f USING (source, row_id)
            WHERE r.role_v2 IN ('train', 'development')
        ),
        outside AS (
            SELECT *, row_number() OVER (PARTITION BY source, role_v2
                ORDER BY md5('{SEED}:cache:' || group_id), row_id) AS position
            FROM active WHERE fold IS NULL
        )
        SELECT * EXCLUDE (position), 'cv' AS partition FROM (SELECT *, 0 AS position FROM active
            WHERE fold IS NOT NULL)
        UNION ALL
        SELECT * EXCLUDE (position),
               CASE WHEN role_v2 = 'train' THEN 'extra' ELSE 'dev_clean' END
        FROM outside
        WHERE position <= CASE WHEN role_v2 = 'train' THEN {extra_per_source}
                               ELSE {dev_per_source} END""")
    tables = []
    for source in ("uk", "ru"):
        path = (DATA_DIR / f"{source}_geo_dataset.csv").as_posix()
        db.execute(f"""CREATE TEMP TABLE joined_{source} AS
            SELECT c.*, d.text, d.loc_markers, md5(d.text) AS actual_key
            FROM (SELECT row_number() OVER () - 1 AS row_id, text, loc_markers
                  FROM read_csv('{path}', all_varchar=true)) d
            JOIN (SELECT * FROM chosen WHERE source = '{source}') c USING (row_id)""")
        tables.append(f"joined_{source}")
    union = " UNION ALL ".join(f"SELECT * FROM {name}" for name in tables)
    expected = db.execute("SELECT count(*) FROM chosen").fetchone()[0]
    found, wrong_text, wrong_label = db.execute(f"""SELECT count(*),
        count(*) FILTER (WHERE actual_key != text_key),
        count(*) FILTER (WHERE (loc_markers != '[]') != has_location) FROM ({union})""").fetchone()
    if (found, wrong_text, wrong_label) != (expected, 0, 0):
        raise ValueError(
            f"CSV does not match the manifests: expected {expected} rows, joined {found}, "
            f"{wrong_text} wrong texts, {wrong_label} wrong location flags. "
            "Use the original supplied CSVs with their original row order."
        )
    db.execute(f"""COPY (SELECT * EXCLUDE (actual_key) FROM ({union}) ORDER BY source, row_id)
        TO '{ROWS_FILE.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    db.close()
    return ROWS_FILE


def load_rows(partitions=("cv", "extra", "dev_clean")):
    """Read cached rows as dictionaries with parsed location markers."""
    frame = pd.read_parquet(ROWS_FILE)
    frame = frame[frame["partition"].isin(partitions)]
    rows = frame.to_dict("records")
    for row in rows:
        row["markers"] = sorted(set(parse_markers(row.pop("loc_markers"))))
        row["fold"] = None if pd.isna(row["fold"]) else int(row["fold"])
    return rows


def load_test():
    """The 477 competition posts. Never used for fitting anything."""
    with (DATA_DIR / "test.csv").open(newline="", encoding="utf-8") as handle:
        return [dict(source="test", row_id=row["text_id"], text=row["text"], markers=[])
                for row in csv.DictReader(handle)]


def load_labeling_sample():
    """The 26 labeled posts. Reported for reference; never used for tuning."""
    with (DATA_DIR / "labeling_sample.csv").open(newline="", encoding="utf-8") as handle:
        return [dict(source="sample", row_id=str(index), text=row["text"],
                     markers=[tuple(pair) for pair in ast.literal_eval(row["loc_markers"])])
                for index, row in enumerate(csv.DictReader(handle))]


def rank(key, salt):
    return hashlib.md5(f"{SEED}:{salt}:{key}".encode()).hexdigest()


def split_fold(rows, fold):
    """Training rows from the other two folds, and the held-out fold."""
    training = [r for r in rows if r["partition"] == "cv" and r["fold"] != fold]
    heldout = [r for r in rows if r["partition"] == "cv" and r["fold"] == fold]
    return training, heldout


def split_inner(rows, fraction=0.05):
    """Group-safe inner validation for early stopping and threshold tuning."""
    limit = int(fraction * 10_000)
    inner = [r for r in rows if int(rank(r["group_id"], "inner")[:8], 16) % 10_000 < limit]
    fit = [r for r in rows if int(rank(r["group_id"], "inner")[:8], 16) % 10_000 >= limit]
    return fit, inner


def take_rows(preferred, reserve, source, count):
    """Take `count` rows of one language, using `reserve` only after `preferred` runs out."""
    chosen = sorted((r for r in preferred if r["source"] == source),
                    key=lambda r: rank(r["row_id"], "take"))[:count]
    if len(chosen) < count:
        extra = sorted((r for r in reserve if r["source"] == source),
                       key=lambda r: rank(r["group_id"], "take"))
        chosen += extra[:count - len(chosen)]
    return chosen


def has_address(row):
    return any(ADDRESS_CUE.search(row["text"][a:b]) for a, b in row["markers"])


def check_no_leak(training, evaluation):
    """Stop the run if a group, document or text appears on both sides."""
    for field in ("group_id", "doc_id", "text_key"):
        left = {r[field] for r in training if isinstance(r.get(field), str)}
        right = {r[field] for r in evaluation if isinstance(r.get(field), str)}
        if left & right:
            raise ValueError(f"Leakage: {len(left & right)} shared values of {field}")
    allowed = {"train", "development"}
    if {r["role_v2"] for r in training + evaluation} - allowed:
        raise ValueError("A final or excluded row was selected")


def make_pseudo_posts(rows, seed, fraction=1.0, add_footers=True):
    """Join rows of one language into multi-line posts, shifting their location offsets.

    This imitates the structure of competition posts: several lines, lists and an
    unlabeled footer. It is a stress test for long inputs, not competition data.
    """
    rng = random.Random(seed)
    selected = list(rows)
    rng.shuffle(selected)
    cut = int(len(selected) * fraction)
    to_join, untouched = selected[:cut], selected[cut:]
    posts = []
    for source in ("uk", "ru"):
        pool = [r for r in to_join if r["source"] == source]
        start = 0
        while start < len(pool):
            size = rng.randint(2, 6)
            parts = pool[start:start + size]
            start += size
            text, markers = "", []
            for index, part in enumerate(parts):
                if index:
                    text += rng.choice(("\n", "\n\n"))
                shift = len(text)
                text += part["text"]
                markers += [(a + shift, b + shift) for a, b in part["markers"]]
            if add_footers and rng.random() < 0.3:
                text += "\n\n" + rng.choice(FOOTERS)
            posts.append(dict(source=source, row_id=f"post:{parts[0]['row_id']}", text=text,
                              markers=markers, group_id=None, doc_id=None, text_key=None,
                              role_v2=parts[0]["role_v2"]))
    return untouched + posts
