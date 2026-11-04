"""Bounded recurrent BIO extraction pilot and grouped outer-fold comparison.

Run through the Colab CLI after uploading extractor_bundle_v1.jsonl.gz. The
bundle contains no final-evaluation rows. EDA_MODE=benchmark|pilot|cv selects
the stage. Each outer fold trains its tokenizer on that fold's training rows.
"""

import gzip
import hashlib
import json
import os
import random
import re
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


SEED=3012026
EPOCHS=3
TOKEN_BUDGET=300000
WINDOW=256
STRIDE=192
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA=Path('/content/extractor_bundle_v1.jsonl.gz')
OUT=Path('/content')


def load_rows():
    with gzip.open(DATA,'rt',encoding='utf-8') as f:
        rows=[json.loads(line) for line in f]
    assert all(r['partition'] in ('pilot_train','pilot_development','cv') for r in rows)
    return rows


def fit_tokenizer(rows,kind):
    model=models.WordLevel(unk_token='[UNK]') if kind=='word' else models.BPE(unk_token='[UNK]')
    tokenizer=Tokenizer(model)
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    trainer=trainers.WordLevelTrainer if kind=='word' else trainers.BpeTrainer
    tokenizer.train_from_iterator((r['text'] for r in rows),trainer=trainer(
        vocab_size=16000,special_tokens=['[PAD]','[UNK]'],show_progress=False))
    return tokenizer


def representable(offsets,markers):
    starts={a for a,b in offsets};ends={b for a,b in offsets}
    return all(a in starts and b in ends for a,b in markers)


def windows(row,tokenizer,training):
    encoded=tokenizer.encode(row['text'])
    ids=encoded.ids;offsets=encoded.offsets
    markers=sorted(set(tuple(p) for p in row['markers']))
    if training and not representable(offsets,markers):return [],False
    owners=[next((i for i,(a,b) in enumerate(markers) if s<b and a<e),None)
            for s,e in offsets]
    tags=[];prev=None
    for owner in owners:
        tags.append(0 if owner is None else 1 if owner!=prev else 2)
        prev=owner
    result=[]
    for start in range(0,max(1,len(ids)),STRIDE):
        end=min(start+WINDOW,len(ids))
        if start>=end:break
        window_tags=tags[start:end].copy()
        if training:
            for i,owner in enumerate(owners[start:end]):
                if owner is None:continue
                a,b=markers[owner]
                if not (offsets[start][0]<=a and b<=offsets[end-1][1]):
                    window_tags[i]=-100
        left_index=start+(0 if start==0 else (WINDOW-STRIDE)//2)
        right_index=end-(0 if end==len(ids) else (WINDOW-STRIDE)//2)
        core_left=offsets[left_index][0]
        core_right=offsets[right_index-1][1]
        result.append((ids[start:end],window_tags,offsets[start:end],row,core_left,core_right))
        if end==len(ids):break
    return result,True


class BiGRU(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.emb=nn.Embedding(vocab,96,padding_idx=0)
        self.drop=nn.Dropout(.15)
        self.gru=nn.GRU(96,96,batch_first=True,bidirectional=True)
        self.out=nn.Linear(192,3)

    def forward(self,x,lengths):
        embedded=self.drop(self.emb(x))
        packed=nn.utils.rnn.pack_padded_sequence(embedded,lengths.cpu(),
                                                  batch_first=True,enforce_sorted=False)
        result,_=self.gru(packed)
        dense,_=nn.utils.rnn.pad_packed_sequence(result,batch_first=True,
                                                 total_length=x.shape[1])
        return self.out(self.drop(dense))


def collate(batch):
    maxlen=max(len(x[0]) for x in batch)
    ids=torch.zeros((len(batch),maxlen),dtype=torch.long)
    tags=torch.full((len(batch),maxlen),-100,dtype=torch.long)
    lengths=torch.empty(len(batch),dtype=torch.long)
    for i,(tokens,labels,*_) in enumerate(batch):
        lengths[i]=len(tokens)
        ids[i,:len(tokens)]=torch.tensor(tokens)
        tags[i,:len(tokens)]=torch.tensor(labels)
    return ids.to(DEVICE),tags.to(DEVICE),lengths


def rank(row):
    return hashlib.md5(f'{SEED}:extractor-order:{row["source"]}:{row["row_id"]}'.encode()).hexdigest()


def choose_training(rows,tokenizer,source_mix):
    by_source={s:sorted((r for r in rows if r['source']==s),key=rank) for s in ('uk','ru')}
    budgets={'uk':TOKEN_BUDGET if source_mix=='uk' else TOKEN_BUDGET//2,
             'ru':0 if source_mix=='uk' else TOKEN_BUDGET//2}
    chosen=[];spent=defaultdict(int);excluded=defaultdict(int)
    for source,budget in budgets.items():
        if budget==0:continue
        for row in by_source[source]:
            items,ok=windows(row,tokenizer,True)
            if not ok:excluded[source]+=1;continue
            cost=sum(len(item[0]) for item in items)
            if spent[source]+cost>budget and spent[source]>0:break
            chosen.extend(items);spent[source]+=cost
    return chosen,dict(spent),dict(excluded)


def train_model(rows,tokenizer,source_mix,seed):
    torch.manual_seed(seed);random.seed(seed)
    examples,spent,excluded=choose_training(rows,tokenizer,source_mix)
    assert examples
    model=BiGRU(tokenizer.get_vocab_size()).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    weights=torch.tensor([1.,3.,2.],device=DEVICE)
    loss_fn=nn.CrossEntropyLoss(weight=weights,ignore_index=-100)
    losses=[];started=time.monotonic()
    for epoch in range(EPOCHS):
        random.shuffle(examples)
        model.train();total=0;count=0
        for start in range(0,len(examples),32):
            x,y,lengths=collate(examples[start:start+32])
            optimizer.zero_grad()
            logits=model(x,lengths)
            loss=loss_fn(logits.reshape(-1,3),y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total+=loss.item();count+=1
        losses.append(total/count)
        print('epoch',epoch+1,'loss',round(losses[-1],4),flush=True)
    return model,dict(token_budget=TOKEN_BUDGET,spent_tokens=spent,
                      excluded_unrepresentable=excluded,training_windows=len(examples),
                      losses=losses,training_seconds=time.monotonic()-started)


def decode(tags,offsets,core_left,core_right):
    spans=[];start=None;end=None
    for label,(a,b) in zip(tags,offsets):
        if label==0:
            if start is not None:spans.append((start,end));start=end=None
            continue
        if label==1 or start is None:
            if start is not None:spans.append((start,end))
            start=a;end=b
        else:end=b
    if start is not None:spans.append((start,end))
    return [(a,b) for a,b in spans if core_left <= (a+b)/2 < core_right]


def predict(model,tokenizer,rows):
    model.eval();predictions={}
    with torch.no_grad():
        for row in rows:
            items,_=windows(row,tokenizer,False)
            spans=set()
            for start in range(0,len(items),32):
                batch=items[start:start+32]
                x,_,lengths=collate(batch)
                labels=model(x,lengths).argmax(-1).cpu().tolist()
                for item,tags in zip(batch,labels):
                    ids,_,offsets,_,left,right=item
                    spans.update(decode(tags[:len(ids)],offsets,left,right))
            predictions[(row['source'],row['row_id'])]=sorted(spans)
    return predictions


def normalize(value):
    value=re.sub(r'[{re.escape(string.punctuation)}]','',value)
    value=re.sub(r'\d','',value)
    value=re.sub(r'\b\w\b\s?','',value)
    return re.sub(r'\s+',' ',value).strip().lower()


import string


def score(rows,predictions):
    count=defaultdict(lambda:defaultdict(int));details=[]
    for row in rows:
        key=(row['source'],row['row_id'])
        actual=set(tuple(p) for p in row['markers'])
        predicted=set(predictions[key])
        text=row['text']
        actual_strings={normalize(text[a:b]) for a,b in actual}
        predicted_strings={normalize(text[a:b]) for a,b in predicted}
        detail=dict(source=row['source'],row_id=row['row_id'],group_id=row.get('group_id',
                    row['doc_id'] or row['row_id']),actual_spans=sorted(actual),
                    predicted_spans=sorted(predicted),
                    actual_strings=sorted(actual_strings),predicted_strings=sorted(predicted_strings))
        details.append(detail)
        for source in (row['source'],'combined'):
            for prefix,gold,guess in [('string',actual_strings,predicted_strings),
                                      ('span',actual,predicted)]:
                count[source][prefix+'_tp']+=len(gold&guess)
                count[source][prefix+'_fp']+=len(guess-gold)
                count[source][prefix+'_fn']+=len(gold-guess)
    metrics={}
    for source,values in count.items():
        item=dict(values)
        for prefix in ('string','span'):
            tp=item[prefix+'_tp'];fp=item[prefix+'_fp'];fn=item[prefix+'_fn']
            item[prefix+'_precision']=tp/(tp+fp) if tp+fp else int(fn==0)
            item[prefix+'_recall']=tp/(tp+fn) if tp+fn else int(fp==0)
            item[prefix+'_f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
        metrics[source]=item
    return metrics,details


def checkpoint(model,tokenizer,name):
    folder=OUT/'eda_checkpoints';folder.mkdir(exist_ok=True)
    torch.save(model.cpu().state_dict(),folder/f'{name}.pt')
    model.to(DEVICE)
    tokenizer.save(str(folder/f'{name}.json'))


def bundle_checkpoints():
    with tarfile.open(OUT/'eda_checkpoints.tar.gz','w:gz') as tar:
        tar.add(OUT/'eda_checkpoints',arcname='eda_checkpoints')


def benchmark():
    assert DATA.exists()
    torch.manual_seed(SEED)
    model=BiGRU(16000).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    x=torch.randint(0,16000,(32,96),device=DEVICE)
    y=torch.randint(0,3,(32,96),device=DEVICE)
    lengths=torch.full((32,),96,dtype=torch.long)
    for i in range(55):
        if DEVICE.type=='cuda':torch.cuda.synchronize()
        if i==5:started=time.monotonic()
        optimizer.zero_grad()
        loss=nn.functional.cross_entropy(model(x,lengths).reshape(-1,3),y.reshape(-1))
        loss.backward();optimizer.step()
    if DEVICE.type=='cuda':torch.cuda.synchronize()
    result=dict(device=str(DEVICE),gpu=torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,
                seconds_per_step=(time.monotonic()-started)/50,
                projected_steps=3*EPOCHS*TOKEN_BUDGET/(32*96),
                projected_training_seconds=(time.monotonic()-started)/50*3*EPOCHS*TOKEN_BUDGET/(32*96),
                caveat='Excludes tokenization, evaluation, setup, checkpoint transfer, and CV; re-estimate after pilot')
    (OUT/'eda_benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


def experiments(mode):
    rows=load_rows()
    results=[];predictions=[]
    scenarios=[('bpe','mixed'),('bpe','uk'),('word','mixed')]
    if mode=='pilot':
        training=[r for r in rows if r['partition']=='pilot_train']
        evaluation=[r for r in rows if r['partition']=='pilot_development']
        folds=[('pilot',training,evaluation)]
    else:
        foldrows=[r for r in rows if r['partition']=='cv']
        folds=[(str(f),[r for r in foldrows if r['fold']!=f],
                      [r for r in foldrows if r['fold']==f]) for f in range(3)]
    for fold,training,evaluation in folds:
        tokenizers={kind:fit_tokenizer(training,kind) for kind in ('bpe','word')}
        for kind,mix in scenarios:
            name=f'{mode}_{fold}_{kind}_{mix}'
            print('experiment',name,'train',len(training),'eval',len(evaluation),flush=True)
            started=time.monotonic()
            model,diagnostics=train_model(training,tokenizers[kind],mix,SEED)
            guessed=predict(model,tokenizers[kind],evaluation)
            metrics,details=score(evaluation,guessed)
            checkpoint(model,tokenizers[kind],name)
            results.append(dict(name=name,fold=fold,kind=kind,mix=mix,
                                training=diagnostics,metrics=metrics,
                                total_seconds=time.monotonic()-started,
                                evaluation_rows=len(evaluation)))
            for detail in details:detail['model']=name
            predictions.extend(details)
            print('score',name,metrics['combined']['string_f1'],flush=True)
    with DATA.open('rb') as f:bundle_hash=hashlib.file_digest(f,'sha256').hexdigest()
    report=dict(mode=mode,seed=SEED,epochs=EPOCHS,token_budget_per_epoch=TOKEN_BUDGET,
        hardware=dict(device=str(DEVICE),gpu=torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None),
        bundle_sha256=bundle_hash,results=results,
        caveats=['Fixed argmax decoding and three epochs; no outer-fold early stopping or threshold tuning.',
                 'Published score is reproduced literally, including its punctuation-expression ambiguity.',
                 'Word and BPE candidates receive equal token budgets but different row counts.',
                 'Pilot development and CV outer-fold scores are distinct comparisons.'])
    (OUT/f'eda_{mode}_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with gzip.open(OUT/f'eda_{mode}_predictions.jsonl.gz','wt',encoding='utf-8') as f:
        for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    bundle_checkpoints()
    print('saved',mode,'results and checkpoints',flush=True)


if __name__=='__main__':
    mode=os.environ.get('EDA_MODE','benchmark')
    if mode=='benchmark':benchmark()
    elif mode in ('pilot','cv'):experiments(mode)
    else:raise ValueError(mode)
