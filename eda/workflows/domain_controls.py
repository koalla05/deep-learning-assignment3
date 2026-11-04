"""Paired, grouped controls for the source-versus-competition GRU diagnostic."""

import hashlib
import json
import random
import re
from collections import Counter

import duckdb
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from eda.core.adversarial import DomainGRU, _encode, _vocabulary
from eda.core.analysis import OUTPUT


SEEDS=(3012026,3012027,3012028)
EPOCHS=6


def hash_rank(value):
    return hashlib.md5(f'3012026:domain:{value}'.encode()).hexdigest()


def template(text):
    return re.sub(r'\d+','#',re.sub(r'https?://\S+|www\.\S+','URL',text.lower())).strip()


def mask_cues(text):
    text=re.sub(r'https?://\S+|www\.\S+',' ',text)
    text=re.sub(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]',' ',text)
    return text.replace('\n',' ').replace('\r',' ')


def select(audit,source):
    candidates=audit[source]['sample']
    ids=','.join(str(r['row_id']) for r in candidates)
    with duckdb.connect() as db:
        active={r[0] for r in db.execute(f"""SELECT row_id FROM read_parquet(?)
          WHERE source=? AND role_v2 IN ('train','development') AND row_id IN ({ids})""",
          [str(OUTPUT/'role_manifest_v2.parquet'),source]).fetchall()}
    candidates=sorted((r for r in candidates if r['row_id'] in active and r['text']),
                      key=lambda r:hash_rank(f'{source}:{r["row_id"]}'))
    unique=[];seen_doc=set();seen_text=set()
    for r in candidates:
        group=r['doc_id'] if source=='ru' else r['text']
        if group in seen_doc or r['text'] in seen_text:continue
        seen_doc.add(group);seen_text.add(r['text']);unique.append(r)
    tests=sorted(enumerate(audit['test']['rows']),key=lambda v:hash_rank(f'test:{v[0]}'))
    test_unique=[];seen_template=set()
    for i,r in tests:
        key=template(r['text'])
        if key in seen_template:continue
        seen_template.add(key);test_unique.append(dict(row_id=i,text=r['text'],group=key))
    full_source=unique[:400];full_test=test_unique[:400]
    assert len(full_source)==len(full_test)==400
    matched_source=[];matched_test=[];used_docs=set();used_ids=set()
    for test in test_unique:
        if not 40 <= len(test['text']) <= 200:continue
        options=(r for r in unique if r['row_id'] not in used_ids and
                 (r['doc_id'] if source=='ru' else r['text']) not in used_docs)
        best=min(options,key=lambda r:abs(len(r['text'])-len(test['text'])),default=None)
        if best is None or abs(len(best['text'])-len(test['text']))>15:continue
        used_docs.add(best['doc_id'] if source=='ru' else best['text'])
        used_ids.add(best['row_id'])
        matched_source.append(best);matched_test.append(test)
        if len(matched_source)==160:break
    assert len(matched_source)>=120,(source,len(matched_source))
    return [(full_source,full_test),(matched_source,matched_test)]


def split_groups(rows,source):
    # Stable membership across seeds and diagnostic conditions.
    ordered=sorted(rows,key=lambda r:hash_rank(str(r['doc_id'] if source=='ru' else r['text'])))
    cutoff=int(len(ordered)*.75)
    return ordered[:cutoff],ordered[cutoff:]


def train_eval(source_rows,test_rows,source,seed,masked):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    src_train,src_hold=split_groups(source_rows,source)
    tst_train,tst_hold=split_groups(test_rows,'test')
    transform=mask_cues if masked else lambda x:x
    train=[transform(r['text']) for r in src_train+tst_train]
    hold=[transform(r['text']) for r in src_hold+tst_hold]
    labels=torch.tensor([0]*len(src_train)+[1]*len(tst_train),dtype=torch.float32)
    y=np.array([0]*len(src_hold)+[1]*len(tst_hold))
    vocab=_vocabulary(train)
    x,ln=_encode(train,vocab);xh,lnh=_encode(hold,vocab)
    model=DomainGRU(len(vocab)+2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        model.train()
        order=torch.randperm(len(train))
        for positions in order.split(64):
            optimizer.zero_grad()
            loss=loss_fn(model(x[positions],ln[positions]),labels[positions])
            loss.backward();optimizer.step()
    model.eval()
    with torch.no_grad():pred=torch.sigmoid(model(xh,lnh)).numpy()
    return dict(auc=float(roc_auc_score(y,pred)),heldout_source=len(src_hold),
                heldout_test=len(tst_hold),train_source=len(src_train),train_test=len(tst_train),
                heldout_probabilities=pred.tolist(),heldout_labels=y.tolist(),
                heldout_source_ids=[r['row_id'] for r in src_hold],
                heldout_test_ids=[r['row_id'] for r in tst_hold])


def run():
    audit=json.loads((OUTPUT/'audit.json').read_text())
    results=[];sizes=[]
    for source in ('uk','ru'):
        (full_src,full_test),(matched_src,matched_test)=select(audit,source)
        sizes.append(dict(source=source,full_per_class=400,matched_per_class=len(matched_src),
            matched_source_mean_length=float(np.mean([len(r['text']) for r in matched_src])),
            matched_test_mean_length=float(np.mean([len(r['text']) for r in matched_test])),
            matched_mean_absolute_length_gap=float(np.mean([abs(len(a['text'])-len(b['text']))
                                                        for a,b in zip(matched_src,matched_test)]))))
        for seed in SEEDS:
            for condition,src,tst,masked in [
                ('full',full_src,full_test,False),
                ('length_matched',matched_src,matched_test,False),
                ('length_matched_cues_masked',matched_src,matched_test,True)]:
                print(source,seed,condition,flush=True)
                results.append(dict(source=source,seed=seed,condition=condition,
                                    **train_eval(src,tst,source,seed,masked)))
    summary=dict(seeds=SEEDS,epochs=EPOCHS,max_characters=256,sizes=sizes,results=results,
        split='Fixed group-hash 75/25 per class; source documents/texts and test templates stay in one partition',
        caveats=['Matched selection uses source audit samples and test length <=200; it does not represent full competition posts.',
                 'Cue masking changes model inputs, not source data or labels.',
                 'AUC differences are descriptive and do not establish effects on extraction performance.'])
    (OUTPUT/'domain_controls.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    run()
