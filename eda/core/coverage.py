"""Full-corpus development coverage and indexed orthographic-overlap audit.

The cache is local and rebuildable. Final-evaluation text is omitted: this stage
investigates model-selection data, not final-evaluation examples.
"""

import hashlib
import json
import unicodedata

import duckdb

from eda.core.analysis import DATA, OUTPUT, parse_markers


CACHE = OUTPUT / 'coverage.duckdb'


def entity_strings(text, raw):
    return [' '.join(unicodedata.normalize('NFC', text[a:b]).lower().split())
            for a, b in sorted(set(parse_markers(raw)))]


def records(db, query, parameters=None):
    result = db.execute(query, parameters or [])
    return [dict(zip([c[0] for c in result.description], row)) for row in result.fetchall()]


def build_cache(db):
    db.create_function('entity_strings', entity_strings, ['VARCHAR','VARCHAR'], 'VARCHAR[]')
    db.read_parquet(str(OUTPUT / 'role_manifest_v1.parquet')).create_view('roles')
    for source in ('uk', 'ru'):
        print(f'Building {source} diagnostic cache', flush=True)
        db.execute(f"""CREATE OR REPLACE TABLE {source}_rows AS
            WITH raw AS (SELECT row_number() OVER ()-1 AS row_id,text,loc_markers
                         FROM read_csv(?,all_varchar=true))
            SELECT r.*,m.role,m.doc_id,m.text_key,m.holdout_group,
                   length(text) AS chars, m.has_location,
                   trim(regexp_replace(lower(nfc_normalize(text)), '[^\\p{{L}}\\p{{N}}]+',' ','g')) AS normalized,
                   entity_strings(text,loc_markers) AS entities
            FROM raw r JOIN roles m USING(row_id)
            WHERE m.source=? AND m.role!='final'
        """, [str(DATA / f'{source}_geo_dataset.csv'), source])
    db.execute("""CREATE OR REPLACE VIEW all_rows AS
        SELECT 'uk' AS source,* FROM uk_rows UNION ALL SELECT 'ru' AS source,* FROM ru_rows""")


def audit(db):
    summary = {}
    summary['profiles'] = records(db, """SELECT source,role,count(*) AS rows,
        count(DISTINCT text_key) AS unique_texts,avg(has_location::INT) AS positive_fraction,
        avg(chars) AS mean_characters,quantile_disc(chars,[0.5,0.9,0.99]) AS character_quantiles,
        avg(len(entities)) AS entities_per_row
        FROM all_rows GROUP BY source,role ORDER BY source,role""")
    summary['development_groups'] = records(db, """SELECT source,count(*) AS rows,
        count(DISTINCT CASE WHEN source='ru' THEN holdout_group ELSE text_key END) AS independent_exact_groups,
        count(DISTINCT doc_id) AS documents FROM all_rows WHERE role='development' GROUP BY source""")
    print('Building training-only entity frequency reference', flush=True)
    db.execute("""CREATE OR REPLACE TABLE entity_frequency AS
        SELECT source,unnest(entities) AS entity,count(*) AS mentions
        FROM all_rows WHERE role='train' GROUP BY source,entity""")
    summary['development_entities'] = records(db, """WITH mentions AS (
        SELECT source,row_id,unnest(entities) AS entity FROM all_rows WHERE role='development')
        SELECT d.source,count(*) AS mentions,count(DISTINCT d.entity) AS unique_strings,
            sum(f.entity IS NULL)::BIGINT AS unseen_mentions,
            count(DISTINCT d.entity) FILTER(WHERE f.entity IS NULL) AS unseen_unique_strings,
            sum(coalesce(f.mentions,0) BETWEEN 1 AND 5)::BIGINT AS rare_seen_mentions,
            quantile_disc(length(d.entity),[0.5,0.9,0.99]) AS entity_character_quantiles
        FROM mentions d LEFT JOIN entity_frequency f USING(source,entity) GROUP BY d.source""")
    summary['language_cues'] = records(db, """SELECT source,role,
        CASE WHEN regexp_matches(text,'[іїєґІЇЄҐ]') AND regexp_matches(text,'[ыэъёЫЭЪЁ]') THEN 'both_scripts'
             WHEN regexp_matches(text,'[іїєґІЇЄҐ]') THEN 'uk_specific_letters'
             WHEN regexp_matches(text,'[ыэъёЫЭЪЁ]') THEN 'ru_specific_letters'
             ELSE 'indeterminate' END AS cue,count(*) AS rows
        FROM all_rows WHERE role IN ('train','development') GROUP BY ALL ORDER BY ALL""")
    # Equality after conservative orthographic normalization is an indexed candidate
    # retrieval rule, not proof of semantic equivalence or exhaustive near-dup search.
    db.execute("""CREATE OR REPLACE TABLE orthographic_links AS
        WITH development_keys AS (SELECT DISTINCT source,normalized FROM all_rows
            WHERE role='development' AND length(normalized)>=20),
        training AS (SELECT r.source,r.normalized,count(*) AS train_rows,min(row_id) AS example_train_id
            FROM all_rows r JOIN development_keys k USING(source,normalized)
            WHERE role='train' GROUP BY r.source,r.normalized)
        SELECT d.source,d.row_id AS development_id,t.example_train_id,t.train_rows,d.normalized
        FROM all_rows d JOIN training t USING(source,normalized) WHERE d.role='development'""")
    summary['orthographic_overlap'] = records(db, """SELECT source,count(*) AS development_rows,
        count(DISTINCT normalized) AS normalized_groups,sum(train_rows)::BIGINT AS candidate_pairs
        FROM orthographic_links GROUP BY source""")
    examples = records(db, """SELECT l.source,l.development_id,l.example_train_id,
        d.text AS development_text,t.text AS training_text,d.loc_markers AS development_markers,
        t.loc_markers AS training_markers,l.train_rows
        FROM orthographic_links l JOIN all_rows d ON d.source=l.source AND d.row_id=l.development_id
        JOIN all_rows t ON t.source=l.source AND t.row_id=l.example_train_id
        QUALIFY row_number() OVER (PARTITION BY l.source ORDER BY md5('3012026:'||l.development_id::VARCHAR))<=12
        ORDER BY source,development_id""")
    (OUTPUT / 'orthographic_review.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    summary['scope'] = 'All retained training/development rows; exclusions profiled; final rows omitted'
    summary['entity_matching'] = 'Per source; Unicode NFC, lowercase, collapsed whitespace; preserves digits and punctuation'
    summary['retrieval'] = 'Full-corpus hash-joined normalized equality, >=20 normalized characters; does not retrieve arbitrary edits or paraphrases'
    summary['input_sha256'] = json.loads((OUTPUT / 'split_summary.json').read_text())['input_sha256']
    with (OUTPUT / 'role_manifest_v1.parquet').open('rb') as f:
        summary['role_manifest_sha256'] = hashlib.file_digest(f,'sha256').hexdigest()
    (OUTPUT / 'coverage.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__ == '__main__':
    with duckdb.connect(str(CACHE)) as db:
        db.execute('SET threads=1')  # stable CSV row numbering
        db.execute("SET memory_limit='2GB'")
        build_cache(db)
        print(json.dumps(audit(db),ensure_ascii=False,indent=2),flush=True)
