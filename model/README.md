# GRU/LSTM experiments

Local notebooks that answer the design questions before the final Colab run. Run them in order. After each one, write your decisions into `nerkit/decisions.py`, because the next notebook starts from them.

| Notebook | Answers |
| --- | --- |
| `00_setup_data_cv` | Does the data join? Is the baseline reproduced? Can I run CV, and on which device? Is there a label gap? |
| `01_tokenization_embeddings` | Word or BPE? What vocabulary size? Which pretrained embeddings? |
| `02_architecture` | GRU or LSTM? Layers, width, direction, CRF? |
| `03_training_setup` | Optimizer, learning rate, scheduler, batch size, regularization, class weights? |
| `04_data_strategy` | How many rows, which language mix, do post-style training and newline tokens help? |
| `05_decoding_logits_ensembling` | Thresholds, merging rules, saved logits, ensembles? |

## Setup

From the repository root:

```sh
git lfs pull
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r model/requirements.txt
.venv/bin/python -m ipykernel install --user --name ai301-ner
```

Put `uk_geo_dataset.csv`, `ru_geo_dataset.csv`, `test.csv` and `labeling_sample.csv` in `data/kse-ua-location-extraction-2026/`.

## How runs work

- Finished runs are saved in `results/<notebook>.jsonl`, so rerunning a notebook skips them and an interrupted notebook resumes. Changing any setting makes it a new run.
- Each run saves logits for `inner`, `heldout`, `posts`, `labeling` and `test` in `artifacts/<run key>/`, plus the tokenizer and model weights. Notebook 05 describes the format and how to combine them.
- For W&B logging, set `NER_WANDB=online` or `offline` before starting Jupyter. It's off by default.
- `final` rows aren't in the cache, and competition text is never used for fitting.

## Layout

`nerkit/` holds the shared code: `data` (leak-safe selection), `tokenization`, `embeddings`, `models`, `engine` (Lightning training), `decoding`, `metrics`, `artifacts` (logits), `experiment` (one full run), `diagnostics` and `plots`.
