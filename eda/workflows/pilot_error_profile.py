"""Pilot error categories on fixed development rows; no label corrections."""

import gzip
import json
from collections import defaultdict

from eda.core.analysis import OUTPUT


def run():
    profile=defaultdict(lambda:defaultdict(int))
    with gzip.open(OUTPUT/'eda_pilot_predictions.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            model='_'.join(r['model'].split('_')[-2:])
            key=(model,r['source'])
            gold=set(r['actual_strings']);guess=set(r['predicted_strings'])
            actual_spans=[tuple(x) for x in r['actual_spans']]
            predicted_spans=[tuple(x) for x in r['predicted_spans']]
            v=profile[key]
            v['rows']+=1
            v['negative_rows']+=not gold
            v['positive_rows']+=bool(gold)
            v['negative_with_prediction']+=not gold and bool(guess)
            v['positive_with_no_prediction']+=bool(gold) and not guess
            v['positive_exact_string_set']+=bool(gold) and gold==guess
            v['positive_with_overlapping_wrong_span']+=bool(gold) and any(
                a<d and c<b and (a,b)!=(c,d) for a,b in actual_spans
                for c,d in predicted_spans)
            v['positive_with_3plus_gold_strings']+=len(gold)>=3
            v['positive_with_3plus_and_miss']+=len(gold)>=3 and bool(gold-guess)
    report=dict(profile=[dict(model=m,source=s,**data) for (m,s),data in profile.items()],
        caveat='Overlapping wrong spans are a broad boundary indicator; raw labels include defects. Categories are not mutually exclusive.')
    (OUTPUT/'pilot_error_profile.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
