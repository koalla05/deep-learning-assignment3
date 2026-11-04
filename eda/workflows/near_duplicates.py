"""Bounded inverted-shingle retrieval for edits beyond orthographic equality."""

from collections import defaultdict, Counter
from difflib import SequenceMatcher
import json
import re
import unicodedata

from eda.core.analysis import OUTPUT
from eda.workflows.tokenizer_comparison import sample_data


def words(text):
    return re.findall(r'\w+',unicodedata.normalize('NFC',text).lower())


def shingles(tokens):
    return {tuple(tokens[i:i+3]) for i in range(len(tokens)-2)}


def run():
    data=sample_data()
    matches=[]
    coverage=[]
    for source in ('uk','ru'):
        training=[r for r in data['training'] if r['source']==source]
        development=[r for r in data['development'] if r['source']==source]
        index=defaultdict(list)
        training_words=[]
        for i,row in enumerate(training):
            tokens=words(row['text']);training_words.append(tokens)
            for shingle in shingles(tokens):
                index[shingle].append(i)
        scored=0
        queries_with_candidates=0
        for row in development:
            tokens=words(row['text'])
            keys=sorted((s for s in shingles(tokens) if s in index),key=lambda s:(len(index[s]),s))[:3]
            candidates=Counter(i for key in keys for i in index[key])
            if candidates:queries_with_candidates+=1
            for i,_ in candidates.most_common(200):
                other=training_words[i]
                if min(len(tokens),len(other))/max(len(tokens),len(other),1)<.7:
                    continue
                scored+=1
                ratio=SequenceMatcher(None,tokens,other,autojunk=False).ratio()
                if ratio>=.85:
                    matches.append(dict(source=source,development_id=row['row_id'],
                                        training_id=training[i]['row_id'],word_sequence_ratio=ratio,
                                        development_text=row['text'],training_text=training[i]['text']))
        coverage.append(dict(source=source,training_rows=len(training),development_queries=len(development),
                             queries_with_candidates=queries_with_candidates,scored_pairs=scored))
    matches.sort(key=lambda r:(r['source'],-r['word_sequence_ratio'],r['development_id'],r['training_id']))
    result=dict(coverage=coverage,matches=matches,
                method='Inverted contiguous three-word shingles; three rarest query shingles; at most 200 candidates; word sequence ratio >=0.85',
                limitations='Sampled retrieval; misses paraphrases, short texts, candidates without a common shingle, and truncated candidate lists. Not an exhaustive leakage guarantee.')
    (OUTPUT/'near_duplicates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(coverage=coverage,matched_pairs=len(matches)),indent=2))


if __name__=='__main__':
    run()
