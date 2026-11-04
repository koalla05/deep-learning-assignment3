"""Compare literal published normalization with corrected punctuation syntax."""

import gzip
import json
import re
import string
from collections import defaultdict

from eda.core.analysis import OUTPUT


def corrected(value):
    value=re.sub(rf'[{re.escape(string.punctuation)}]','',value)
    value=re.sub(r'\d','',value)
    value=re.sub(r'\b\w\b\s?','',value)
    return re.sub(r'\s+',' ',value).strip().lower()


def run(mode='pilot'):
    counts=defaultdict(lambda:[0,0,0])
    partition='pilot_development' if mode=='pilot' else 'cv'
    with gzip.open(OUTPUT/'extractor_bundle_v1.jsonl.gz','rt') as f:
        texts={(r['source'],r['row_id']):r['text'] for line in f
               if (r:=json.loads(line))['partition']==partition}
    with gzip.open(OUTPUT/f'eda_{mode}_predictions.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            name='_'.join(r['model'].split('_')[-2:])
            text=texts[(r['source'],r['row_id'])]
            a={corrected(text[start:end]) for start,end in r['actual_spans']}
            p={corrected(text[start:end]) for start,end in r['predicted_spans']}
            for source in (r['source'],'combined'):
                c=counts[(name,source)]
                c[0]+=len(a&p);c[1]+=len(p-a);c[2]+=len(a-p)
    result={f'{name}:{source}':dict(tp=tp,fp=fp,fn=fn,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1)
        for (name,source),(tp,fp,fn) in counts.items()}
    report=dict(mode=mode,score=result,
        limitation='Diagnostic correction to punctuation syntax, not verified live Kaggle scoring')
    (OUTPUT/f'eda_{mode}_score_sensitivity.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    import sys
    print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else 'pilot'),indent=2))
