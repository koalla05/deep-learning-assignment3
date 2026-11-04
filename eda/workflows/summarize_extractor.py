"""Summarize pilot and three-fold extractor results with grouped paired uncertainty."""

import gzip
import json
from collections import defaultdict

import numpy as np

from eda.core.analysis import OUTPUT


def counts(row,prefix):
    a=set(row['actual_strings'] if prefix=='string' else map(tuple,row['actual_spans']))
    p=set(row['predicted_strings'] if prefix=='string' else map(tuple,row['predicted_spans']))
    return np.array([len(a&p),len(p-a),len(a-p)],dtype=np.int64)


def f1(vector):
    tp,fp,fn=map(int,vector)
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.


def read_predictions(mode):
    path=OUTPUT/f'eda_{mode}_predictions.jsonl.gz'
    with gzip.open(path,'rt') as file:
        return [json.loads(line) for line in file]


def candidate(model):
    parts=model.split('_')
    return parts[-2]+'_'+parts[-1]


def summarize(mode='cv'):
    raw=json.loads((OUTPUT/f'eda_{mode}_results.json').read_text())
    rows=read_predictions(mode)
    by_candidate=defaultdict(list)
    for row in rows:by_candidate[candidate(row['model'])].append(row)
    result=dict(mode=mode,source_results={})
    for name,records in by_candidate.items():
        expected=63388 if mode=='cv' else 2000
        assert len(records)==expected,(name,len(records))
        keys={(r['source'],r['row_id']) for r in records}
        assert len(keys)==len(records)
        source_scores={}
        for source in ('uk','ru','combined'):
            selected=records if source=='combined' else [r for r in records if r['source']==source]
            sums={prefix:sum((counts(r,prefix) for r in selected),np.zeros(3,dtype=np.int64))
                  for prefix in ('string','span')}
            source_scores[source]={prefix:dict(tp=int(v[0]),fp=int(v[1]),fn=int(v[2]),f1=f1(v))
                                   for prefix,v in sums.items()}
        folds=[r for r in raw['results'] if candidate(r['name'])==name]
        source_scores['fold_scores']={source:[r['metrics'][source]['string_f1'] for r in folds]
                                      for source in ('uk','ru','combined')}
        source_scores['fold_mean_spread']={source:dict(mean=float(np.mean(scores)),
            min=float(min(scores)),max=float(max(scores)),sample_sd=float(np.std(scores,ddof=1)) if len(scores)>1 else None)
            for source,scores in source_scores['fold_scores'].items()}
        result['source_results'][name]=source_scores
    if mode=='cv':
        rng=np.random.default_rng(3012026)
        candidates=sorted(by_candidate)
        paired={}
        for left,right in [('bpe_mixed','bpe_uk'),('bpe_mixed','word_mixed')]:
            maps={name:{(r['source'],r['row_id']):r for r in by_candidate[name]} for name in (left,right)}
            assert maps[left].keys()==maps[right].keys()
            for source in ('uk','ru','combined'):
                group_contrib=defaultdict(lambda:np.zeros((2,3),dtype=np.int64))
                for key,a in maps[left].items():
                    if source!='combined' and a['source']!=source:continue
                    b=maps[right][key]
                    assert a['group_id']==b['group_id']
                    group_contrib[(a['source'],a['group_id'])][0]+=counts(a,'string')
                    group_contrib[(a['source'],a['group_id'])][1]+=counts(b,'string')
                groups=list(group_contrib)
                values=np.stack([group_contrib[g] for g in groups])
                observed=f1(values[:,0].sum(axis=0))-f1(values[:,1].sum(axis=0))
                draws=[]
                # Source-stratified resampling preserves source sizes in the combined score.
                by_source=defaultdict(list)
                for i,g in enumerate(groups):by_source[g[0]].append(i)
                for _ in range(500):
                    sampled=np.concatenate([rng.choice(indices,len(indices),replace=True)
                                            for indices in by_source.values()])
                    sums=values[sampled].sum(axis=0)
                    draws.append(f1(sums[0])-f1(sums[1]))
                paired[f'{left}_minus_{right}:{source}']=dict(observed=observed,
                    bootstrap_percentile_95=np.quantile(draws,[.025,.975]).tolist(),
                    independent_groups=len(groups))
        result['paired_group_bootstrap']=paired
        result['bootstrap_caveat']='Resamples held-out document/duplicate groups and does not capture retraining variability; three fold scores are not confidence intervals.'
    (OUTPUT/f'eda_{mode}_summary.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    import sys
    print(json.dumps(summarize(sys.argv[1] if len(sys.argv)>1 else 'cv'),indent=2))
