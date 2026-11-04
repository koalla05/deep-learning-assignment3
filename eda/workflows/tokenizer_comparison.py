"""Matched tokenizer coverage and window diagnostics; no extraction training."""

import hashlib
import json
import re
from collections import Counter

import duckdb
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from eda.core.analysis import DATA, OUTPUT, parse_markers
from eda.core.coverage import CACHE, records
from eda.core.reconstruction import reconstruct


WORDS = re.compile(r'\w+',re.UNICODE)
SIZES = (1000, 10000, 50000)  # per source, nested deterministic subsets
VOCABS = (8000, 16000)


def sample_data(role_version=1):
    with duckdb.connect(str(CACHE),read_only=True) as db:
        db.execute("SET memory_limit='2GB'")
        training, development = [], []
        for source in ('uk','ru'):
            for role,n,destination in [('train',max(SIZES),training),('development',1000,development)]:
                if role_version == 2:
                    query=f"""SELECT d.source,d.row_id,d.doc_id,d.text,d.loc_markers
                        FROM all_rows d JOIN read_parquet('{(OUTPUT / 'role_manifest_v2.parquet').as_posix()}') r USING(source,row_id)
                        WHERE d.source=? AND r.role_v2=?
                        ORDER BY md5('3012026:tokenizer:'||d.row_id::VARCHAR) LIMIT {n}"""
                else:
                    query=f"""SELECT source,row_id,doc_id,text,loc_markers
                        FROM all_rows WHERE source=? AND role=?
                        ORDER BY md5('3012026:tokenizer:'||row_id::VARCHAR) LIMIT {n}"""
                destination.extend(records(db,query,[source,role]))
    import csv
    with (DATA/'test.csv').open(newline='') as f:
        test=[dict(source='test',row_id=i,text=r['text'],loc_markers='[]') for i,r in enumerate(csv.DictReader(f))]
    result=dict(training=training,development=development,test=test)
    (OUTPUT/f'tokenizer_samples_v{role_version}.json').write_text(json.dumps(result,ensure_ascii=False))
    return result


def fit(kind, vocabulary, rows):
    model = models.WordLevel(unk_token='[UNK]') if kind=='word' else models.BPE(unk_token='[UNK]')
    tokenizer=Tokenizer(model)
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    trainer_type = trainers.WordLevelTrainer if kind=='word' else trainers.BpeTrainer
    tokenizer.train_from_iterator((r['text'] for r in rows), trainer=trainer_type(
        vocab_size=vocabulary,special_tokens=['[PAD]','[UNK]'],show_progress=False))
    return tokenizer


def measure(tokenizer, rows):
    counts=Counter()
    lengths=[]
    entity_pieces=[]
    unknown_id=tokenizer.token_to_id('[UNK]')
    for row in rows:
        text=row['text']
        encoding=tokenizer.encode(text)
        offsets=encoding.offsets
        unknown=[span for span,idx in zip(offsets,encoding.ids) if idx==unknown_id]
        lengths.append(len(offsets))
        words=[m.span() for m in WORDS.finditer(text)]
        markers=sorted(set(parse_markers(row['loc_markers'])))
        counts.update(rows=1,pieces=len(offsets),unknown_pieces=len(unknown),words=len(words),
                      unknown_words=sum(any(s<b and a<e for s,e in unknown) for a,b in words),
                      characters=sum(not c.isspace() for c in text),
                      unknown_characters=len({i for a,b in unknown for i in range(a,b) if not text[i].isspace()}),
                      entities=len(markers),unknown_entities=sum(any(s<b and a<e for s,e in unknown) for a,b in markers))
        predicted=reconstruct(offsets,markers,True)
        actual=set(markers); pred=set(predicted)
        counts.update(span_tp=len(actual&pred),span_fp=len(pred-actual),span_fn=len(actual-pred))
        for a,b in markers:
            pieces=[i for i,(s,e) in enumerate(offsets) if s<b and a<e]
            entity_pieces.append(len(pieces))
            for cap in (256,512):
                if pieces:
                    counts[f'entities_beyond_prefix_{cap}'] += pieces[-1]>=cap
                    starts=range(0,max(1,len(offsets)),cap-64)
                    counts[f'entities_not_contained_window_{cap}'] += not any(
                        start<=pieces[0] and pieces[-1]<start+cap for start in starts)
                else:
                    counts[f'entities_beyond_prefix_{cap}'] += 1
                    counts[f'entities_not_contained_window_{cap}'] += 1
        for cap in (256,512):
            counts[f'rows_over_{cap}']+=len(offsets)>cap
    result=dict(counts)
    for name,num,den in [('unknown_word_rate','unknown_words','words'),
                         ('unknown_character_rate','unknown_characters','characters'),
                         ('unknown_piece_rate','unknown_pieces','pieces'),
                         ('unknown_entity_rate','unknown_entities','entities')]:
        result[name]=counts[num]/counts[den] if counts[den] else None
    denominator=2*counts['span_tp']+counts['span_fp']+counts['span_fn']
    result['bio_span_f1']=2*counts['span_tp']/denominator if denominator else None
    result['length_quantiles']=np.quantile(lengths,[.5,.9,.99]).tolist()
    result['pieces_per_entity']=float(np.mean(entity_pieces)) if entity_pieces else None
    return result


def synthetic_posts(rows):
    """Concatenate source sentences and translate offsets; boundary stress only."""
    result=[]
    text=''; markers=[]
    for row in rows:
        shift=len(text)
        text+=row['text']+'\n'
        markers.extend((a+shift,b+shift) for a,b in parse_markers(row['loc_markers']))
        if len(text)>=1200:
            result.append(dict(text=text,loc_markers=repr(markers)))
            text='';markers=[]
        if len(result)==50:
            break
    return result


def run():
    sample=sample_data(2)
    results=[]
    for size in SIZES:
        training=[r for source in ('uk','ru') for r in [x for x in sample['training'] if x['source']==source][:size]]
        for kind in ('word','bpe'):
            for vocabulary in VOCABS:
                print(f'{kind} vocab={vocabulary} training={2*size}',flush=True)
                tokenizer=fit(kind,vocabulary,training)
                name=f'{kind}_{vocabulary}_{2*size}'
                tokenizer.save(str(OUTPUT/f'tokenizer_{name}.json'))
                for source in ('uk','ru','test','synthetic'):
                    rows=(sample['test'] if source=='test' else synthetic_posts(sample['development'])
                          if source=='synthetic' else [r for r in sample['development'] if r['source']==source])
                    results.append(dict(name=name,kind=kind,requested_vocabulary=vocabulary,
                                        actual_vocabulary=tokenizer.get_vocab_size(),training_rows=2*size,
                                        source=source,**measure(tokenizer,rows)))
    with (OUTPUT/'tokenizer_samples_v2.json').open('rb') as f:
        fingerprint=hashlib.file_digest(f,'sha256').hexdigest()
    report=dict(seed=3012026,source_mixture='50% Ukrainian / 50% Russian by rows',
                sample_sha256=fingerprint,results=results,
                limitations=['Row-balanced mixture is not token-balanced training.',
                             'Synthetic posts are boundary stress tests, not target-domain validation.',
                             'Coverage and gold-span reconstruction are not extraction quality.',
                             'Largest fit uses 100,000 rows; stabilization is assessed, not assumed.'])
    (OUTPUT/'tokenizer_comparison.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    run()
