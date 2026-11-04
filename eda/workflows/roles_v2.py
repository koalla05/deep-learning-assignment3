"""Conservative role overlay using the audited duplicate components."""

import hashlib
import json
from pathlib import Path

import duckdb

from eda.core.analysis import OUTPUT
from eda.core.coverage import CACHE, records


DECISIONS = Path(__file__).resolve().parents[1] / 'config' / 'near_duplicate_decisions.json'


def build():
    decisions = json.loads(DECISIONS.read_text())
    with duckdb.connect(str(CACHE)) as db:
        db.execute('SET threads=1')
        db.execute("SET memory_limit='2GB'")
        db.read_parquet(str(OUTPUT / 'role_manifest_v1.parquet')).create_view('base_roles')
        # Accepted sampled edit links extend the equality-based components.
        for item in decisions:
            if not item['accepted']:
                continue
            a, b = db.execute("""SELECT group_id FROM proposed_groups
                WHERE source=? AND row_id IN (?,?) ORDER BY row_id""",
                [item['source'],item['development_id'],item['training_id']]).fetchall()
            if a != b:
                target, other = min(a[0], b[0]), max(a[0], b[0])
                db.execute('UPDATE proposed_groups SET group_id=? WHERE group_id=?', [target,other])
        db.execute("""CREATE OR REPLACE TEMP TABLE priority AS
            SELECT group_id,bool_or(role='final') AS has_final,
                bool_or(role='development') AS has_development
            FROM proposed_groups WHERE role IN ('development','final') GROUP BY group_id""")
        db.execute("""CREATE OR REPLACE TEMP TABLE assigned_v2 AS
            SELECT b.*,p.group_id,
                CASE WHEN b.role='train' AND q.has_final THEN 'exclude_final_group'
                     WHEN b.role='train' AND q.has_development THEN 'exclude_development_group'
                     WHEN b.role='development' AND q.has_final THEN 'exclude_final_group'
                     ELSE b.role END AS role_v2
            FROM base_roles b LEFT JOIN proposed_groups p USING(source,row_id)
            LEFT JOIN priority q USING(group_id)""")
        counts = records(db,"""SELECT source,role_v2,count(*) AS rows,
            sum(has_location::INT)::BIGINT AS location_rows
            FROM assigned_v2 GROUP BY ALL ORDER BY ALL""")
        checks = records(db,"""SELECT count(*) AS total_rows,
            count(DISTINCT (source,row_id)) AS unique_rows FROM assigned_v2""")[0]
        assert checks['total_rows'] == checks['unique_rows'] == 9038840
        group_overlap = db.execute("""SELECT count(*) FROM (
            SELECT group_id FROM assigned_v2 WHERE role_v2 IN ('train','development','final')
            GROUP BY group_id HAVING count(DISTINCT role_v2)>1)""").fetchone()[0]
        assert group_overlap == 0, group_overlap
        doc_overlap = db.execute("""SELECT count(*) FROM (
            SELECT doc_id FROM assigned_v2 WHERE source='ru'
                AND role_v2 IN ('train','development','final')
            GROUP BY doc_id HAVING count(DISTINCT role_v2)>1)""").fetchone()[0]
        assert doc_overlap == 0, doc_overlap
        text_overlap = db.execute("""SELECT count(*) FROM (
            SELECT text_key FROM assigned_v2 WHERE role_v2 IN ('train','development','final')
            GROUP BY text_key HAVING count(DISTINCT role_v2)>1)""").fetchone()[0]
        assert text_overlap == 0, text_overlap
        final_change = db.execute("""SELECT count(*) FROM assigned_v2
            WHERE (role='final') != (role_v2='final')""").fetchone()[0]
        assert final_change == 0
        path = OUTPUT / 'role_manifest_v2.parquet'
        db.execute("COPY (SELECT * FROM assigned_v2 ORDER BY source,row_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)", [str(path)])
        with (OUTPUT/'role_manifest_v1.parquet').open('rb') as f:
            parent_hash = hashlib.file_digest(f,'sha256').hexdigest()
        with DECISIONS.open('rb') as f:
            decision_hash = hashlib.file_digest(f,'sha256').hexdigest()
        summary = dict(version=2,parent_manifest_sha256=parent_hash,
                       near_duplicate_decisions_sha256=decision_hash,counts=counts,
                       total_rows=checks['total_rows'],group_overlap=group_overlap,
                       document_overlap=doc_overlap,text_overlap=text_overlap,
                       final_membership_changes=final_change,
                       rule='Exclude lower-priority active rows in a document/equality/accepted-edit component; final > development > train',
                       caveat='Repeated generic snippets create a giant Russian component; characterize excluded-data bias before training.')
        (OUTPUT/'role_summary_v2.json').write_text(json.dumps(summary,indent=2))
        return summary


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
