"""Measure connected document groups under exact and orthographic equality.

Final text is used only to compute predefined overlap keys, never exported for
example review. This script proposes role changes; it does not apply them.
"""
import hashlib
import json

import duckdb

from eda.core.analysis import DATA,OUTPUT
from eda.core.coverage import CACHE,records


def run():
    with duckdb.connect(str(CACHE)) as db:
        db.execute('SET threads=1')
        db.execute("SET memory_limit='2GB'")
        db.read_parquet(str(OUTPUT/'role_manifest_v1.parquet')).create_view('base_roles')
        db.execute("""CREATE OR REPLACE TABLE group_keys AS
            SELECT source,row_id,CASE WHEN length(normalized)>=20 THEN md5(normalized)
                ELSE text_key END AS key FROM all_rows WHERE role IN ('train','development')""")
        for source in ('uk','ru'):
            db.execute("""INSERT INTO group_keys
                WITH raw AS (SELECT row_number() OVER ()-1 AS row_id,text FROM read_csv(?,all_varchar=true)),
                final AS (SELECT r.row_id,m.text_key,trim(regexp_replace(lower(nfc_normalize(text)),
                    '[^\\p{L}\\p{N}]+',' ','g')) AS normalized
                    FROM raw r JOIN base_roles m USING(row_id) WHERE m.source=? AND m.role='final')
                SELECT ?,row_id,CASE WHEN length(normalized)>=20 THEN md5(normalized) ELSE text_key END FROM final
            """,[str(DATA/f'{source}_geo_dataset.csv'),source,source])
        docs=db.execute("SELECT DISTINCT doc_id FROM base_roles WHERE source='ru' AND role IN ('train','development','final')").fetchall()
        parent={d:d for (d,) in docs}
        def root(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]];x=parent[x]
            return x
        links=db.execute("""SELECT list(DISTINCT doc_id) FROM group_keys k
            JOIN base_roles r USING(source,row_id) WHERE source='ru'
            GROUP BY k.key HAVING count(DISTINCT doc_id)>1""").fetchall()
        for (members,) in links:
            for doc in members[1:]:
                a,b=root(members[0]),root(doc)
                parent[max(a,b)]=min(a,b)
        import pyarrow as pa
        components=pa.table({'doc_id':list(parent),'component':[root(d) for d in parent]})
        db.register('components_arrow',components)
        db.execute('CREATE OR REPLACE TABLE document_components AS SELECT * FROM components_arrow')
        db.execute("""CREATE OR REPLACE TABLE proposed_groups AS
            SELECT r.source,r.row_id,r.role,
                CASE WHEN r.source='ru' THEN 'ru:'||c.component ELSE 'uk:'||k.key END AS group_id
            FROM base_roles r JOIN group_keys k USING(source,row_id)
            LEFT JOIN document_components c ON r.source='ru' AND r.doc_id=c.doc_id""")
        summary={}
        summary['components']=records(db,"""WITH sizes AS (
            SELECT source,group_id,count(*) AS rows FROM proposed_groups GROUP BY source,group_id)
            SELECT source,count(*) AS groups,max(rows) AS largest_rows,
                quantile_disc(rows,[0.5,0.9,0.99]) AS row_quantiles FROM sizes GROUP BY source""")
        summary['largest']=records(db,"""SELECT group_id,count(*) AS rows,
            sum(role='train')::BIGINT AS training_rows,sum(role='development')::BIGINT AS development_rows,
            sum(role='final')::BIGINT AS final_rows FROM proposed_groups
            GROUP BY group_id ORDER BY rows DESC LIMIT 5""")
        summary['conservative_exclusions']=records(db,"""WITH priority AS (
            SELECT group_id,bool_or(role='final') AS has_final,bool_or(role='development') AS has_development
            FROM proposed_groups GROUP BY group_id)
            SELECT source,role,count(*) AS rows FROM proposed_groups JOIN priority USING(group_id)
            WHERE (role='train' AND (has_final OR has_development)) OR (role='development' AND has_final)
            GROUP BY source,role ORDER BY source,role""")
        summary['rule']='NFC/lowercase/alphanumeric word sequence equality >=20 characters; exact equality for shorter rows; all Russian document members unioned transitively'
        summary['status']='Diagnostic only; role version 1 is not changed'
        (OUTPUT/'group_audit.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
