"""Freeze a reproducible grouped three-fold extractor comparison sample."""

import hashlib
import json
from collections import defaultdict

import duckdb

from eda.core.analysis import OUTPUT


SEED = 3012026
THRESHOLDS = {'uk': 310, 'ru': 48}  # per 10,000 groups, ~30k rows/source


def bucket(key):
    return int(hashlib.md5(f'{SEED}:cv:{key}'.encode()).hexdigest()[:8], 16) % 10000


def build():
    db = duckdb.connect()
    db.execute('SET threads=1')
    db.execute("SET memory_limit='2GB'")
    path = OUTPUT / 'role_manifest_v2.parquet'
    db.read_parquet(str(path)).create_view('roles')
    db.create_function('cv_bucket', bucket, ['VARCHAR'], 'INTEGER')
    db.execute("""CREATE TEMP TABLE sampled AS
        SELECT source,row_id,doc_id,text_key,has_location,group_id,role_v2
        FROM roles WHERE role_v2 IN ('train','development') AND
            cv_bucket(group_id)<CASE WHEN source='uk' THEN 310 ELSE 48 END""")
    groups = db.execute("""SELECT source,group_id,count(*) AS rows,
        sum(has_location::INT)::BIGINT AS positive_rows FROM sampled
        GROUP BY source,group_id ORDER BY source,rows DESC,group_id""").fetchall()
    assignments=[]
    balance={s:[[0,0] for _ in range(3)] for s in THRESHOLDS}
    totals=defaultdict(lambda:[0,0])
    for source,_,rows,positive in groups:
        totals[source][0]+=rows;totals[source][1]+=positive
    for source,group_id,rows,positive in groups:
        target_rows,target_pos=(x/3 for x in totals[source])
        best=min(range(3),key=lambda fold:(
            (((balance[source][fold][0]+rows)**2-balance[source][fold][0]**2)/max(target_rows,1)**2)+
            (((balance[source][fold][1]+positive)**2-balance[source][fold][1]**2)/max(target_pos,1)**2),
            balance[source][fold][0],
            bucket(f'{group_id}:tie:{fold}')))
        balance[source][best][0]+=rows
        balance[source][best][1]+=positive
        assignments.append((source,group_id,best))
    db.execute('CREATE TEMP TABLE fold_groups(source VARCHAR, group_id VARCHAR, fold INTEGER)')
    db.executemany('INSERT INTO fold_groups VALUES (?,?,?)',assignments)
    db.execute("""CREATE TEMP TABLE folds AS SELECT s.*,g.fold
        FROM sampled s JOIN fold_groups g USING(source,group_id)""")
    overlap = db.execute("""SELECT count(*) FROM (
        SELECT group_id FROM folds GROUP BY group_id HAVING count(DISTINCT fold)>1)""").fetchone()[0]
    documents = db.execute("""SELECT count(*) FROM (
        SELECT doc_id FROM folds WHERE source='ru' GROUP BY doc_id
        HAVING count(DISTINCT fold)>1)""").fetchone()[0]
    text = db.execute("""SELECT count(*) FROM (
        SELECT source,text_key FROM folds GROUP BY source,text_key
        HAVING count(DISTINCT fold)>1)""").fetchone()[0]
    assert overlap == documents == text == 0
    counts=db.execute("""SELECT source,fold,count(*) AS rows,
        sum(has_location::INT)::BIGINT AS positive_rows,
        count(DISTINCT group_id) AS groups,
        count(DISTINCT doc_id) AS documents FROM folds GROUP BY source,fold
        ORDER BY source,fold""").fetchall()
    assert len(counts)==6
    out=OUTPUT/'cv_manifest_v1.parquet'
    db.execute('COPY (SELECT * FROM folds ORDER BY source,row_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(out)])
    with path.open('rb') as f:
        parent=hashlib.file_digest(f,'sha256').hexdigest()
    summary=dict(seed=SEED,thresholds_per_10000=THRESHOLDS,
                 source_manifest_sha256=parent,
                 sampled_rows=sum(row[2] for row in counts),
                 columns=['source','fold','rows','positive_rows','groups','documents'],
                 counts=counts,group_overlap=overlap,document_overlap=documents,
                 exact_text_overlap=text,
                 assignment='Group hash sampling, then largest groups assigned to folds to balance rows and positive rows within source',
                 caveat='Approximate stratification; folds are not confidence intervals. Learned preprocessing must be fitted within each training fold.')
    (OUTPUT/'cv_summary_v1.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
