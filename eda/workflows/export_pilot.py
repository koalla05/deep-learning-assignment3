"""Export a bounded, final-free extractor experiment bundle for Colab."""

import gzip
import hashlib
import json

import duckdb

from eda.core.analysis import OUTPUT, parse_markers
from eda.core.coverage import CACHE


def export():
    path=OUTPUT/'extractor_bundle_v1.jsonl.gz'
    counts={}
    with duckdb.connect(str(CACHE),read_only=True) as db, gzip.open(path,'wt',encoding='utf-8') as output:
        db.execute("SET memory_limit='2GB'")
        roles=(OUTPUT/'role_manifest_v2.parquet').as_posix()
        folds=(OUTPUT/'cv_manifest_v1.parquet').as_posix()
        for source in ('uk','ru'):
            for partition,limit in [('pilot_train',15000),('pilot_development',1000)]:
                role='train' if partition=='pilot_train' else 'development'
                query=f"""SELECT d.row_id,d.text,d.loc_markers,d.doc_id
                    FROM all_rows d JOIN read_parquet('{roles}') r USING(source,row_id)
                    WHERE d.source=? AND r.role_v2=?
                    ORDER BY md5('3012026:extractor:'||d.row_id::VARCHAR) LIMIT {limit}"""
                rows=db.execute(query,[source,role]).fetchall()
                counts[f'{partition}:{source}']=len(rows)
                assert len(rows)==limit
                for row_id,text,markers,doc_id in rows:
                    output.write(json.dumps(dict(partition=partition,source=source,row_id=row_id,
                        doc_id=doc_id,text=text,markers=parse_markers(markers)),ensure_ascii=False)+'\n')
        query=f"""SELECT d.source,d.row_id,d.text,d.loc_markers,d.doc_id,f.group_id,f.fold
            FROM all_rows d JOIN read_parquet('{folds}') f USING(source,row_id)
            ORDER BY d.source,d.row_id"""
        result=db.execute(query)
        count=0
        while rows:=result.fetchmany(2000):
            for source,row_id,text,markers,doc_id,group_id,fold in rows:
                output.write(json.dumps(dict(partition='cv',source=source,row_id=row_id,
                    doc_id=doc_id,group_id=group_id,fold=fold,text=text,
                    markers=parse_markers(markers)),ensure_ascii=False)+'\n')
                count+=1
        counts['cv']=count
        assert count==63388
    with path.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    summary=dict(path=str(path),sha256=digest,counts=counts,
        selection='Deterministic rank within v2 train/development; frozen v2 CV manifest; no final rows')
    (OUTPUT/'extractor_bundle_v1.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    print(json.dumps(export(),indent=2))
