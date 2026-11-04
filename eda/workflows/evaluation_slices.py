"""Training-only entity frequencies and development evaluation slices for v2."""

import json

import duckdb

from eda.core.analysis import OUTPUT
from eda.core.coverage import CACHE,records


def run():
    role=(OUTPUT/'role_manifest_v2.parquet').as_posix()
    with duckdb.connect(str(CACHE)) as db:
        db.execute('SET threads=1')
        db.execute("SET memory_limit='2GB'")
        db.execute(f"""CREATE OR REPLACE TABLE entity_frequency_v2 AS
            SELECT d.source,unnest(d.entities) AS entity,count(*) AS mentions
            FROM all_rows d JOIN read_parquet('{role}') r USING(source,row_id)
            WHERE r.role_v2='train' GROUP BY d.source,entity""")
        result={}
        result['entity_coverage']=records(db,f"""WITH development AS (
            SELECT d.source,d.row_id,unnest(d.entities) AS entity FROM all_rows d
            JOIN read_parquet('{role}') r USING(source,row_id)
            WHERE r.role_v2='development')
            SELECT d.source,count(*) AS mentions,count(DISTINCT d.entity) AS unique_strings,
                sum(f.entity IS NULL)::BIGINT AS unseen_mentions,
                count(DISTINCT d.entity) FILTER(WHERE f.entity IS NULL) AS unseen_unique_strings,
                sum(coalesce(f.mentions,0) BETWEEN 1 AND 5)::BIGINT AS rare_seen_mentions
            FROM development d LEFT JOIN entity_frequency_v2 f USING(source,entity)
            GROUP BY d.source""")
        result['development_rows']=records(db,f"""SELECT d.source,
            CASE WHEN d.chars<50 THEN '<50' WHEN d.chars<100 THEN '50-99'
                 WHEN d.chars<200 THEN '100-199' ELSE '200+' END AS length_bin,
            d.has_location,count(*) AS rows FROM all_rows d
            JOIN read_parquet('{role}') r USING(source,row_id)
            WHERE r.role_v2='development' GROUP BY ALL ORDER BY ALL""")
        result['scope']='Version 2 active training entity strings and active development only; Unicode NFC/lowercase/collapsed whitespace, source-specific'
        (OUTPUT/'evaluation_slices_v2.json').write_text(json.dumps(result,indent=2))
        return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
