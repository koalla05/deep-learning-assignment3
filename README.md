# Ukrainian and Russian location extraction

This project finds location names in Ukrainian and Russian text and returns them as strings for a Kaggle competition. This page summarizes what we learned from exploratory analysis and small model experiments, and the rules future experiments must follow. The detailed evidence, numbers, and reproduction steps are in the [EDA guide](eda/README.md).

## Main findings

- **Word tokenization is the strongest baseline so far.** It beats BPE with both GRU and LSTM models, in both languages and every fold, when all models train on the same examples.
- **Word BiLSTM has a small lead over word BiGRU** (combined F1 0.646 versus 0.628). It was tested with one seed and a short training run, so it needs confirming.
- **Unfamiliar names remain difficult.** About 40% of labeled location mentions contain a word outside the learned vocabulary. BPE covers more tokens but often predicts name fragments. On Ukrainian names absent from training, the leading models recover only about a quarter.
- **Competition posts differ from the training sentences.** They are longer and contain lists, line breaks, and footers. A classifier still tells the two collections apart after controlling for length and formatting. Good scores on the source data may not carry over to Kaggle.
- **Labels and duplicates need care.** Some labels are questionable, and many texts repeat across the data. We keep the supplied labels, record doubtful examples, and keep linked texts on the same side of every split.

## What we did

1. **Audited the data.** We checked label quality, repeated and near-identical texts, and how well tokenization preserves location boundaries. [Details](eda/README.md#data-and-labels)
2. **Built leak-free splits.** Linked texts and whole Russian documents stay together; rows that would leak between training and evaluation are excluded. A final evaluation set is reserved. [Details](eda/README.md#how-the-splits-were-built)
3. **Compared small extraction models.** Four combinations of tokenizer (word or BPE) and recurrent layer (GRU or LSTM), plus a Ukrainian-only control, each trained from scratch on three cross-validation folds. [Details](eda/README.md#extraction-experiments)
4. **Measured how different competition posts are.** We trained classifiers to tell source sentences from competition posts (adversarial validation). [Details](eda/README.md#how-different-are-competition-posts)

## Key results

Scores combine the three held-out folds. F1 balances correct predictions against missed locations and compares the location strings found in each text.

| Model | Ukrainian F1 | Russian F1 | Combined F1 |
| --- | ---: | ---: | ---: |
| Word BiGRU | 0.581 | 0.664 | 0.628 |
| BPE BiGRU | 0.513 | 0.576 | 0.548 |
| **Word BiLSTM** | 0.600 | 0.681 | **0.646** |
| BPE BiLSTM | 0.519 | 0.581 | 0.553 |
| Word BiGRU, Ukrainian labels only | 0.631 | 0.165 | 0.324 |

Training only on Ukrainian labels helps Ukrainian a little but collapses Russian. These are small, bounded experiments, not estimates of the Kaggle score.

How well a classifier separates source text from competition posts (AUC: 0.5 is chance, 1.0 is perfect separation), using the word BiLSTM architecture:

| Condition | Ukrainian source | Russian source |
| --- | ---: | ---: |
| Length matched, URLs/emoji/line breaks masked | 0.839 | 0.934 |
| Unmasked texts | 0.937 | 0.967 |

The two collections remain clearly distinguishable. For Russian, much of this comes simply from language: competition posts are mostly Ukrainian. The Ukrainian comparison points to differences beyond language. Results vary across seeds and samples are small; see the [detailed interpretation](eda/README.md#what-the-results-mean).

## Required split and exclusion policy for future experiments

**Every new experiment must use the frozen split.** Use `eda/output/role_manifest_v2.parquet` and its `role_v2` column, not a fresh random split of the raw CSVs. Join it to the original source rows by `(source, row_id)`. The manifests are shared through Git LFS; the [dataset guide](data/kse-ua-location-extraction-2026/README.md#team-setup-and-experiment-splits) explains how to download and join them.

| `role_v2` | Ukrainian rows | Russian rows | Meaning and permitted use |
| --- | ---: | ---: | --- |
| `train` | 959,912 | 6,415,583 | Eligible pool for training and tokenizer fitting; experiments may use smaller samples. |
| `development` | 39,912 | 119,789 | Model selection in fixed-split experiments. Enters training only through grouped CV or an explicitly approved final refit. |
| `final` | 10,000 | 32,014 | Reserved evaluation. Use only after the recipe is frozen and evaluation is approved; never for training or model selection. |
| `exclude_conflict` | 0 | 22,562 | The same text appears with different location labels. Excluded rather than guessing the right label. |
| `exclude_duplicate` | 48 | 98,608 | Training row whose exact text also appears in development or final. |
| `exclude_final_duplicate` | 4 | 0 | Ukrainian development row whose exact text also appears in final. |
| `exclude_development_group` | 97 | 15,169 | Training row linked to a development group. |
| `exclude_final_group` | 27 | 1,325,115 | Training or development row linked to a group containing final rows. |

Each row has exactly one role, so the exclusion counts do not overlap. **Never use any `exclude_*` row for training, tokenizer fitting, or evaluation**, even in larger future runs. The raw files still contain these rows; their presence is not permission to use them. Always select the allowed roles explicitly.

1. **Fixed holdout:** train the model and tokenizer on `train` rows only; select models on `development`.
2. **Cross-validation:** reuse `eda/output/cv_manifest_v1.parquet` (built from role version 2). Train on two folds, evaluate on the third, and fit the tokenizer on the training folds only.
3. **Final set:** keep it out of training, tokenizer fitting, and model selection until the recipe is frozen and final evaluation is approved. The 26 labeled competition examples are for qualitative review only.
4. **Before each experiment:** confirm that no excluded rows are selected and that no group, Russian document, or exact text appears on both the training and evaluation sides.
5. **Record** manifest hashes, selected row IDs, role counts, and those checks. Any split change needs a new, audited version, never a silent fallback.

## Next steps

1. Keep word BiGRU as the reproducible baseline; retest word BiLSTM with more seeds and longer training.
2. Vary the amount of Russian data while keeping Ukrainian examples fixed, and report each language separately.
3. Test pretrained word embeddings against random initialization, focusing on unfamiliar names.
4. Review doubtful labels and long-post errors before changing annotation or decoding rules.
5. Freeze the training recipe before using the final evaluation set or making a new submission.

## Repository guide

| Path | Contents |
| --- | --- |
| [`eda/README.md`](eda/README.md) | Detailed analysis, experiment setup, and reproduction steps |
| [`eda/notebooks/`](eda/notebooks/) | Executed data audit, extraction, and domain analyses |
| [`eda/model_plan.md`](eda/model_plan.md) | Scope of the GRU/LSTM comparison |
| [`eda/competition_like_validation_plan.md`](eda/competition_like_validation_plan.md) | Plan for reliable submissions and a competition-like validation set |
| [`data/kse-ua-location-extraction-2026/README.md`](data/kse-ua-location-extraction-2026/README.md) | Dataset description and team setup |

Raw data, checkpoints, generated outputs, and submissions stay outside Git; only the two split manifests are shared, through Git LFS.
