"""Select a reproducible training-only review and audit duplicate annotations."""

import json
from collections import defaultdict
from pathlib import Path
import duckdb
from eda.core.analysis import DATA, OUTPUT, parse_markers


def run():
    db = duckdb.connect()
    db.execute("SET threads=1")
    db.execute("SET memory_limit='2GB'")
    db.read_parquet(str(OUTPUT / 'split_manifest.parquet')).create_view('manifest')
    boundary_ids = [r['row_id'] for r in json.loads((OUTPUT / 'boundary_cases.json').read_text())]
    db.execute('CREATE TABLE boundaries(row_id BIGINT)')
    db.executemany('INSERT INTO boundaries VALUES (?)', [(i,) for i in boundary_ids])
    sample, canonical_summary = [], {}
    for source in ('uk', 'ru'):
        db.execute("""CREATE OR REPLACE TEMP TABLE raw AS
            SELECT row_number() OVER () - 1 AS row_id, text, loc_markers
            FROM read_csv(?, all_varchar=true)""", [str(DATA / f'{source}_geo_dataset.csv')])
        db.execute("""CREATE OR REPLACE TEMP VIEW training AS
            SELECT r.*, m.has_location, md5('3012026:review:' || r.row_id::VARCHAR) AS priority
            FROM raw r JOIN manifest m USING(row_id)
            WHERE m.source='""" + source + "' AND m.split='train'")
        selected = set()
        def take(where, n, method):
            omitted = ','.join(str(i) for i in selected) or '-1'
            rows = db.execute(f"SELECT row_id,text,loc_markers FROM training WHERE ({where}) AND row_id NOT IN ({omitted}) ORDER BY priority LIMIT {n}").fetchall()
            added = 0
            for row_id, text, raw in rows:
                if row_id in selected:
                    continue
                selected.add(row_id)
                sample.append(dict(source=source, row_id=row_id, selection=method, text=text,
                                   markers=parse_markers(raw), reviewer='', verdict='pending', notes=''))
                added += 1
                if added == n:
                    break
            if added != n:
                raise ValueError(f'Insufficient candidates: {source} {method}: {added}/{n}')
        take('has_location', 15, 'random_positive')
        take('NOT has_location', 15, 'random_negative')
        if source == 'uk':
            take('row_id IN (SELECT row_id FROM boundaries)', 10, 'targeted_boundary')
        take("NOT has_location AND regexp_matches(text, '(вул[.]|улиц|міст|город|Київ|Киев|Москв|Льв|район|област)')",
             10, 'targeted_negative_cue')
        if source == 'ru':
            take("has_location AND regexp_matches(text, '(улиц|город|район|област|университет|компани|завод)')",
                 10, 'targeted_category')
        groups = defaultdict(set)
        rows = db.execute("""SELECT m.text_key, r.loc_markers FROM raw r JOIN manifest m USING(row_id)
            WHERE m.source=? AND m.conflicting_annotations""", [source]).fetchall()
        for key, markers in rows:
            groups[key].add(tuple(sorted(set(parse_markers(markers)))))
        canonical_summary[source] = dict(raw_conflict_groups=len(groups),
            semantic_conflict_groups=sum(len(v)>1 for v in groups.values()),
            serialization_only_groups=sum(len(v)==1 for v in groups.values()), rows=len(rows))
    assert len(sample) == 100
    decisions_path = Path(__file__).resolve().parents[1] / 'config' / 'review_decisions.json'
    if decisions_path.exists():
        decisions = {(r['source'], r['row_id']): r for r in json.loads(decisions_path.read_text())}
        assert set(decisions) == {(r['source'], r['row_id']) for r in sample}
        for row in sample:
            row.update(decisions[(row['source'], row['row_id'])])
    (OUTPUT / 'annotation_review.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2))
    lines = ['# Annotation review', '',
             'Assistant first pass. Judgments are provisional; original labels are unchanged. '
             'Random and targeted samples must be interpreted separately.', '']
    for row in sample:
        lines.extend([f"## {row['source']} row {row['row_id']} — {row['selection']}", '',
                      row['text'], '', 'Marked: ' + repr([row['text'][a:b] for a,b in row['markers']]),
                      '', row['verdict'] + ': ' + row['notes'], ''])
    (OUTPUT / 'annotation_review.md').write_text('\n'.join(lines))
    (OUTPUT / 'canonical_conflicts.json').write_text(json.dumps(canonical_summary, indent=2))
    split_summary = json.loads((OUTPUT / 'split_summary.json').read_text())
    metadata = dict(seed=3012026, selection='Lowest MD5 ranks within each stratum; training only',
                    random_rows=60, targeted_rows=40,
                    input_sha256=split_summary['input_sha256'],
                    limitation='Equal stratum allocation and targeted retrieval do not estimate unweighted corpus prevalence')
    (OUTPUT / 'annotation_review_metadata.json').write_text(json.dumps(metadata, indent=2))
    return canonical_summary


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
