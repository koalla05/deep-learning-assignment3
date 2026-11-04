"""Export the fixed 26 competition-style labels for one qualitative check."""

import csv
import json

from eda.core.analysis import DATA, OUTPUT, parse_markers


def export():
    with (DATA/'labeling_sample.csv').open(newline='',encoding='utf-8') as f:
        rows=[dict(source='qualitative',row_id=i,doc_id=None,text=r['text'],
                   markers=parse_markers(r['loc_markers']))
              for i,r in enumerate(csv.DictReader(f))]
    assert len(rows)==26
    path=OUTPUT/'qualitative_sample.json'
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    return path


if __name__=='__main__':
    print(export())
