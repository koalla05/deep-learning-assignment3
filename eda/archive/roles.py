"""Materialize the agreed evaluation roles without replacing the base manifest."""

import hashlib
import json
import duckdb
from eda.core.analysis import OUTPUT


def build_roles():
    db = duckdb.connect()
    db.execute("SET threads=1")
    db.execute("SET memory_limit='2GB'")
    db.read_parquet(str(OUTPUT / 'split_manifest.parquet')).create_view('base')
    docs = db.execute("SELECT DISTINCT doc_id FROM base WHERE split='ru_holdout'").fetchall()
    parent = {d: d for (d,) in docs}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (linked,) in db.execute("""
        SELECT list(DISTINCT doc_id) FROM base WHERE split='ru_holdout'
        GROUP BY text_key HAVING count(DISTINCT doc_id)>1
    """).fetchall():
        for doc in linked[1:]:
            a, b = root(linked[0]), root(doc)
            parent[max(a, b)] = min(a, b)
    db.execute('CREATE TABLE components(doc_id VARCHAR, group_id VARCHAR, role VARCHAR)')
    db.executemany('INSERT INTO components VALUES (?, ?, ?)', [
        (doc, root(doc), 'final' if int(hashlib.md5(
            f'3012026:ru-final:{root(doc)}'.encode()).hexdigest()[:8], 16) % 10000 < 2000
         else 'development') for doc in parent])
    db.execute("""CREATE TEMP TABLE assigned AS
        SELECT b.*, c.group_id AS holdout_group,
          CASE WHEN split='train' THEN 'train'
               WHEN split='uk_valid' THEN 'final'
               WHEN split='uk_holdout' AND text_key IN
                    (SELECT text_key FROM base WHERE split='uk_valid')
                    THEN 'exclude_final_duplicate'
               WHEN split='uk_holdout' THEN 'development'
               WHEN split='ru_holdout' THEN c.role
               ELSE split END AS role
        FROM base b LEFT JOIN components c ON b.source='ru' AND b.doc_id=c.doc_id
    """)
    counts = db.execute('SELECT source, role, count(*) FROM assigned GROUP BY ALL ORDER BY ALL').fetchall()
    overlap = db.execute("""SELECT count(*) FROM (SELECT text_key FROM assigned
        WHERE role IN ('train','development','final') GROUP BY text_key
        HAVING count(DISTINCT role)>1)""").fetchone()[0]
    doc_overlap = db.execute("""SELECT count(*) FROM (SELECT doc_id FROM assigned
        WHERE source='ru' AND role IN ('train','development','final') GROUP BY doc_id
        HAVING count(DISTINCT role)>1)""").fetchone()[0]
    total, unique = db.execute("SELECT count(*), count(DISTINCT (source,row_id)) FROM assigned").fetchone()
    assert total == unique == 9038840, (total, unique)
    assert overlap == doc_overlap == 0, (overlap, doc_overlap)
    expected = {('uk','train'):960035, ('uk','development'):39913, ('uk','final'):10000,
                ('ru','train'):7747985, ('ru','development'):127671, ('ru','final'):32014,
                ('uk','exclude_final_duplicate'):4}
    actual = {(s,r):n for s,r,n in counts}
    assert all(actual[k] == v for k,v in expected.items()), counts
    path = OUTPUT / 'role_manifest_v1.parquet'
    db.execute("COPY (SELECT * FROM assigned ORDER BY source,row_id) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    with (OUTPUT / 'split_manifest.parquet').open('rb') as f:
        fingerprint = hashlib.file_digest(f, 'sha256').hexdigest()
    summary = dict(version=1, seed=3012026, base_sha256=fingerprint, counts=counts,
                   text_overlap=overlap, document_overlap=doc_overlap, total_rows=total,
                   russian_holdout_components=len({root(d) for d in parent}))
    (OUTPUT / 'role_summary_v1.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    print(json.dumps(build_roles(), indent=2))
