"""One fixed qualitative application of the selected pilot checkpoint."""

import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

if __package__:
    from .colab_extractor import BiGRU, predict, score, DEVICE
else:
    from colab_extractor import BiGRU, predict, score, DEVICE


def main():
    name=os.environ.get('EDA_CHECKPOINT','pilot_pilot_word_mixed')
    root=Path('/content')
    rows=json.loads((root/'qualitative_sample.json').read_text())
    tokenizer=Tokenizer.from_file(str(root/'eda_checkpoints'/f'{name}.json'))
    model=BiGRU(tokenizer.get_vocab_size()).to(DEVICE)
    model.load_state_dict(torch.load(root/'eda_checkpoints'/f'{name}.pt',map_location=DEVICE,weights_only=True))
    predictions=predict(model,tokenizer,rows)
    _,details=score(rows,predictions)
    for detail,row in zip(details,rows):
        detail['text']=row['text']
    report=dict(checkpoint=name,rows=details,
        limitation='Fixed 26-item qualitative set; no tuning, training, or prevalence estimate')
    (root/'eda_qualitative.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('Qualitative rows saved:',len(rows))


if __name__ == '__main__':
    main()
