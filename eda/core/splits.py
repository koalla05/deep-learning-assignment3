"""Materialize deterministic, leakage-checked validation membership."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import duckdb

from eda.core.analysis import DATA, OUTPUT


RULES = json.loads((Path(__file__).resolve().parents[1] / "config" / "split_rules.json").read_text())


def build_manifest() -> Path:
    OUTPUT.mkdir(exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads=1")  # preserve CSV row order for stable row IDs
    connection.execute("SET memory_limit='2GB'")
    connection.execute(f"SET temp_directory='{(OUTPUT / 'split_tmp').as_posix()}'")
    uk = (DATA / "uk_geo_dataset.csv").as_posix()
    ru = (DATA / "ru_geo_dataset.csv").as_posix()
    seed = int(RULES["seed"])
    uk_rate = int(RULES["uk_additional_holdout_per_10000"])
    ru_rate = int(RULES["ru_document_holdout_per_10000"])
    if not 0 <= uk_rate <= 10000 or not 0 <= ru_rate <= 10000:
        raise ValueError("Holdout thresholds must be between 0 and 10,000")
    connection.execute(f"""
        CREATE TEMP TABLE records AS
        SELECT 'uk' AS source, row_number() OVER () - 1 AS row_id,
               NULL::VARCHAR AS doc_id, md5(text) AS text_key,
               md5(loc_markers) AS marker_key, loc_markers != '[]' AS has_location,
               CASE WHEN is_valid = '1' THEN 'uk_valid'
                    WHEN (('0x' || substr(md5('{seed}:uk:' || text), 1, 8))::UBIGINT % 10000) < {uk_rate}
                    THEN 'uk_holdout' ELSE 'train' END AS base_split
        FROM read_csv(?, all_varchar=true)
        UNION ALL
        SELECT 'ru' AS source, row_number() OVER () - 1 AS row_id,
               doc_id, md5(text) AS text_key,
               md5(loc_markers) AS marker_key, loc_markers != '[]' AS has_location,
               CASE WHEN (('0x' || substr(md5('{seed}:ru:' || doc_id), 1, 8))::UBIGINT % 10000) < {ru_rate}
                    THEN 'ru_holdout' ELSE 'train' END AS base_split
        FROM read_csv(?, all_varchar=true)
    """, [uk, ru])
    connection.execute("""
        CREATE TEMP TABLE duplicate_info AS
        SELECT text_key,
               bool_or(base_split != 'train') AS appears_in_validation,
               count(DISTINCT marker_key) > 1 AS conflicting_annotations
        FROM records GROUP BY text_key HAVING count(*) > 1
    """)
    path = OUTPUT / "split_manifest.parquet"
    connection.execute(f"""
        COPY (
            SELECT r.source, r.row_id, r.doc_id, r.text_key, r.has_location,
                   r.base_split,
                   CASE WHEN d.conflicting_annotations
                        THEN 'exclude_conflict'
                        WHEN r.base_split = 'train' AND d.appears_in_validation
                        THEN 'exclude_duplicate' ELSE r.base_split END AS split,
                   coalesce(d.conflicting_annotations, false) AS conflicting_annotations
            FROM records r LEFT JOIN duplicate_info d USING (text_key)
            ORDER BY r.source, r.row_id
        ) TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    rows = connection.execute("""
        SELECT source, split, count(*) AS rows, sum(has_location::INT) AS location_rows,
               sum(conflicting_annotations::INT) AS annotation_conflict_rows
        FROM read_parquet(?) GROUP BY source, split ORDER BY source, split
    """, [path.as_posix()]).fetchall()
    identity = connection.execute("""
        SELECT source, count(*), count(DISTINCT row_id), min(row_id), max(row_id)
        FROM read_parquet(?) GROUP BY source
    """, [path.as_posix()]).fetchall()
    for _, total, distinct, first, last in identity:
        if distinct != total or first != 0 or last != total - 1:
            raise ValueError("Manifest row IDs do not cover a source CSV exactly once")
    overlap = connection.execute("""
        SELECT count(*) FROM (
            SELECT text_key FROM read_parquet(?)
            GROUP BY text_key
            HAVING bool_or(split = 'train') AND bool_or(split IN ('uk_valid','uk_holdout','ru_holdout'))
        )
    """, [path.as_posix()]).fetchone()[0]
    ru_group_violations = connection.execute("""
        SELECT count(*) FROM (
            SELECT doc_id FROM read_parquet(?) WHERE source='ru' AND split IN ('train','ru_holdout')
            GROUP BY doc_id HAVING count(DISTINCT split)>1
        )
    """, [path.as_posix()]).fetchone()[0]
    ru_holdout_documents = connection.execute("""
        SELECT count(DISTINCT doc_id) FROM read_parquet(?) WHERE split='ru_holdout'
    """, [path.as_posix()]).fetchone()[0]
    summary = {"rules": RULES, "columns": ["source", "split", "rows", "location_rows", "annotation_conflict_rows"],
               "counts": rows, "train_validation_text_overlap": overlap,
               "russian_document_split_violations": ru_group_violations,
               "russian_holdout_documents": ru_holdout_documents,
               "input_bytes": {"uk": (DATA / "uk_geo_dataset.csv").stat().st_size,
                               "ru": (DATA / "ru_geo_dataset.csv").stat().st_size}}
    summary["input_sha256"] = {}
    for source in ("uk", "ru"):
        with (DATA / f"{source}_geo_dataset.csv").open("rb") as handle:
            summary["input_sha256"][source] = hashlib.file_digest(handle, "sha256").hexdigest()
    (OUTPUT / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    connection.close()
    return path


if __name__ == "__main__":
    print(build_manifest())
