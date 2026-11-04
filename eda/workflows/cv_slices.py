"""Compare frozen out-of-fold predictions on source, novelty, and length slices."""

import gzip
import json
import unicodedata
from collections import defaultdict

from eda.core.analysis import OUTPUT


def entity(value):
    return ' '.join(unicodedata.normalize('NFC',value).lower().split())


def read_bundle():
    with gzip.open(OUTPUT/'extractor_bundle_v1.jsonl.gz','rt') as f:
        return {(r['source'],r['row_id']):r for line in f
                if (r:=json.loads(line))['partition']=='cv'}


def candidate(name):
    return '_'.join(name.split('_')[-2:])


def bin_length(length):
    if length<50:return '<50'
    if length<100:return '50-99'
    if length<200:return '100-199'
    return '200+'


def run():
    rows=read_bundle()
    seen={}
    for fold in range(3):
        for source in ('uk','ru'):
            seen[(fold,source)]={entity(r['text'][a:b])
                for r in rows.values() if r['fold']!=fold and r['source']==source
                for a,b in r['markers']}
    metrics=defaultdict(lambda:dict(rows=0,tp=0,fp=0,fn=0))
    predictions=defaultdict(list)
    with gzip.open(OUTPUT/'eda_cv_predictions.jsonl.gz','rt') as f:
        for line in f:
            p=json.loads(line)
            predictions[candidate(p['model'])].append(p)
    for model,items in predictions.items():
        assert len(items)==len(rows)
        for p in items:
            r=rows[(p['source'],p['row_id'])]
            gold=[entity(r['text'][a:b]) for a,b in r['markers']]
            if not gold:novelty='negative'
            elif any(x not in seen[(r['fold'],r['source'])] for x in gold):novelty='contains_unseen'
            else:novelty='all_seen'
            actual=set(p['actual_strings']);guess=set(p['predicted_strings'])
            tp=len(actual&guess);fp=len(guess-actual);fn=len(actual-guess)
            for kind,value in [('novelty',novelty),('length',bin_length(len(r['text']))),
                               ('all','all')]:
                entry=metrics[(model,r['source'],kind,value)]
                entry['rows']+=1;entry['tp']+=tp;entry['fp']+=fp;entry['fn']+=fn
    results=[]
    for (model,source,kind,value),v in sorted(metrics.items()):
        tp,fp,fn=v['tp'],v['fp'],v['fn']
        results.append(dict(model=model,source=source,slice_kind=kind,slice=value,**v,
            precision=tp/(tp+fp) if tp+fp else int(fn==0),
            recall=tp/(tp+fn) if tp+fn else int(fp==0),
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1,
            false_positives_per_1000_rows=1000*fp/v['rows']))
    report=dict(results=results,
        novelty='Entity string NFC/lowercase/collapsed whitespace, source-specific, fitted on the other two folds only',
        limitation='A row with any unseen gold entity enters contains_unseen; negative-row F1 is not informative, use false-positive rate.')
    (OUTPUT/'eda_cv_slices.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
