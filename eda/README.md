# Exploratory analysis

This directory holds the detailed analysis behind the [project README](../README.md). The notebooks show the results; the Python modules make the data checks and experiments reproducible. The project README gives the short, actionable summary; this guide gives the evidence, the setup, and the caveats.

## Where to read the analysis

| Notebook | Questions it answers |
| --- | --- |
| [Data audit](notebooks/data_audit.ipynb) | Are the labels usable? Which texts repeat? How should we split the data? What does tokenization lose? |
| [Extraction experiments](notebooks/extraction_pilot.ipynb) | How were the models compared? Which differences hold across folds? What errors remain? |
| [Source and competition differences](notebooks/domain_analysis.ipynb) | Can a classifier tell the collections apart after controlling for length and formatting? |

The notebooks keep their earlier executed tables and plots as experiment history. The newest BiLSTM domain-result cells have not been executed yet; their saved results are summarized [below](#adversarial-validation-with-the-word-bilstm). The [comparison protocol](model_plan.md) records the scope of the extraction comparison.

## Data and labels

| Data | Rows | Purpose |
| --- | ---: | --- |
| Ukrainian source | 1,010,000 | Labeled source text |
| Russian source | 8,028,840 | Labeled sentences with document IDs |
| `test.csv` | 477 | Kaggle prediction and submission |
| `labeling_sample.csv` | 26 | Separate labeled examples for manual inspection |

Every `locations` field in `test.csv` is the placeholder `[]`; the real labels are hidden. The 26 labeled examples are a separate file with no exact text overlap with the test file, so they cannot tell us the test-set F1.

The Ukrainian Parquet is a processed view of the same source rows. Its flat location tags lose the boundary between adjacent names. Converting the original character positions to BIO tags (beginning, inside, or outside a location) raises reconstruction F1 from 96.57% to 99.49% on retained Ukrainian training rows. This measures how well the label format preserves locations, before any model is trained.

## How the splits were built

The policy that future experiments must follow is in the [project README](../README.md#required-split-and-exclusion-policy-for-future-experiments). This section explains how the split was made and what it costs.

| Set | Ukrainian rows | Russian rows |
| --- | ---: | ---: |
| Training pool | 959,912 | 6,415,583 |
| Development | 39,912 | 119,789 |
| Final local evaluation | 10,000 | 32,014 |

**Where the sets come from.** The supplied Ukrainian validation flag defines its 10,000 final rows. Development adds examples taken from the original training data. Split sizes are practical choices, not the result of a statistical power calculation.

**Finding repeated texts.** Both languages were checked for exact duplicates, for texts that match after normalization, and for a sample of near-duplicates with manually reviewed decisions. Normalization lowercases the text, unifies Unicode forms, and replaces punctuation and symbols with spaces. Normalized matching applies only to texts of at least 20 normalized characters; shorter texts must match exactly. The initial audit found 251,830 groups of repeated texts covering 777,785 rows.

**Keeping linked texts together.** Linked Ukrainian texts stay in the same set. Ukrainian rows have no document IDs, and its largest group of matching texts has four rows. Russian grouping also keeps whole documents together and follows links transitively: if document A shares a sentence with B, and B with C, all three end up in one group.

**What that costs.** About 1.34 million Russian rows were excluded from training and development so that linked examples never cross sets. Generic news snippets connect one especially large group of about 1.32 million rows. These are connected document rows, not 1.32 million identical sentences. We kept this conservative choice rather than reclaiming rows linked only by boilerplate. It changes the retained data slightly: excluded rows linked to final evaluation contain locations more often than retained training rows (33.8% versus 30.2%).

**Limits of the checks.** Exact-overlap checks passed. The near-duplicate search was sampled, so it cannot rule out every paraphrase. The final set was included in early descriptive data checks before being reserved, but no candidate model has been scored on it.

**Cross-validation sample.** For the experiments we sampled 63,388 rows from the training and development pools and divided them into three folds. Related rows stay in the same fold, and each fold has similar row and positive-example counts within each language. Each model trains on two folds and is evaluated on the third. Final evaluation rows are excluded throughout.

## Extraction experiments

### How the models were trained

All extraction models started from random weights, including their word embeddings. We used PyTorch's standard bidirectional GRU and LSTM layers; “bidirectional” means the model reads context on both sides of each token. A final layer predicts one of the three BIO tags, which are converted back to character spans and location strings.

The latest comparison contains **15 fits**: word and BPE tokenization with both recurrent layers, plus a word BiGRU trained on Ukrainian labels only, each on three folds.

| Setting | Value |
| --- | --- |
| Hardware | Colab Tesla T4 GPU |
| Training rows per mixed-language fit | 5,287–5,293 per language; identical rows for all four models |
| Ukrainian-only control | Same total number of rows, all Ukrainian |
| Vocabulary | Up to 16,000 entries, learned separately for each fold |
| Model size | 96 embedding dimensions; one recurrent layer, 96 hidden units per direction |
| Training duration | Three epochs; batch size 32 |
| Optimizer | AdamW; learning rate 0.003, weight decay 0.01 |
| Loss | Cross-entropy, with weights 1/3/2 for outside/beginning/inside tags |
| Regularization | Dropout 0.15; gradient clipping at 1.0 |
| Long texts | Windows of 256 tokens, overlapping by 64 |
| Random seed | 3012026 for each fit |

**Tokenizers.** Each tokenizer learns from text in the two training folds, including rows outside the smaller supervised sample. It never sees the held-out fold or competition texts. The Ukrainian-only control shares the mixed-language vocabulary: “Ukrainian only” describes its labeled examples, not its vocabulary.

**Row selection and decoding.** We use rows whose location boundaries both tokenizers can represent. Held-out examples keep their original labels. In training windows, labels for locations cut by a window edge are ignored. Predictions take the highest-scoring tag and merge overlapping windows with a fixed rule. We use the model after epoch three; validation scores do not choose checkpoints or decoding settings.

**What is and is not held equal.** Every fit stays below 300,000 training tokens per epoch. On the same rows, word models process about 201,000 tokens and BPE models about 286,000. GRU has 1,648,323 parameters and LSTM has 1,685,571. The comparison holds examples and hidden width constant, but not token count or parameter count. Checkpoints, tokenizers, selected row IDs, predictions, losses, and timings are saved locally.

### Results

Scores combine predictions from all three held-out folds. F1 balances precision (how many predictions are correct) and recall (how many labeled locations are found). The main score compares normalized sets of location strings within each text. Exact-span F1 also requires the character boundaries to match.

| Configuration | Ukrainian F1 | Russian F1 | Combined F1 | Exact-span F1, combined |
| --- | ---: | ---: | ---: | ---: |
| Mixed, word BiGRU | 0.581 | 0.664 | 0.628 | 0.626 |
| Mixed, BPE BiGRU | 0.513 | 0.576 | 0.548 | 0.538 |
| **Mixed, word BiLSTM** | 0.600 | 0.681 | 0.646 | 0.644 |
| Mixed, BPE BiLSTM | 0.519 | 0.581 | 0.553 | 0.546 |
| Ukrainian labels only, word BiGRU | 0.631 | 0.165 | 0.324 | 0.323 |

**Word versus BPE.** Word tokenization wins on both languages in every fold. Its combined advantage is 0.080 with GRU and 0.093 with LSTM; resampling held-out groups gives 95% intervals of 0.073–0.087 and 0.086–0.100.

**LSTM versus GRU.** The word LSTM leads word GRU by 0.018 (interval 0.014–0.021). These intervals describe the saved models; they do not account for training with another seed.

**Ukrainian-only control.** Replacing Russian examples with Ukrainian ones improves Ukrainian F1 by 0.050 but lowers Russian F1 by 0.499. It also doubles Ukrainian supervision and changes which names the model sees, so it does not prove that Russian data hurts when Ukrainian exposure is held fixed.

**Reading the combined score.** It pools correct, extra, and missed locations across both languages. It is not an equal average of the two languages' F1 scores, nor an estimate of Kaggle performance.

**Earlier experiment.** An earlier run gave models equal token budgets instead of equal examples, so word GRU saw about 15,800 examples per fold versus 11,100 for BPE. It scored 0.655 versus 0.559. The latest comparison confirms the word advantage persists with equal examples. Because the samples also changed, the gap between the two experiments does not isolate the benefit of seeing more rows.

### Unfamiliar names

With the word vocabulary, about 40% of labeled location mentions in held-out folds contain at least one word the vocabulary does not know (40.9% Ukrainian, 39.8% Russian). BPE splits every word into known pieces, so it has no unknown words, yet it still scores lower: it often predicts fragments or extra names.

Names that never appeared in a fold's training pool are hard for every model. The four mixed-language models recover 24–25% of such Ukrainian names and 30–35% of such Russian names, compared with 76–87% for names that appeared in the labeled training sample. Names present in the training pool but outside that sample fall in between (30–41%). Seeing a name's text is not proof the model learned it, since a string may also appear unlabeled elsewhere.

### What the errors show

- BPE sometimes predicts fragments such as `Ста` or `Чер`.
- All models miss some lists of places.
- On rows without labeled locations, word LSTM makes fewer extra predictions than word GRU, and both make fewer than BPE GRU.
- Some apparent errors are questionable labels: one reviewed Russian sentence mentions `Румынии` without a location label.
- One run of the earlier word pilot disagreed with the 26 separate labeled examples on 17 rows. Missed streets, neighborhoods, and lists, and predictions in footers, recur. This small, non-random sample shows error types; 17/26 is not a general error rate.

The saved error slices are in `output/eda_cv_slices.json`. The name-exposure analysis is in `output/eda_cv_exposure.json`; it reconstructs exactly which rows each model trained on. There, “unseen” means a name was absent from that fold's training pool, while “available but unsampled” means it was in the pool but outside the model's training sample.

**Scoring caveat.** The competition's scoring code removes digits and repeated strings within a text, so it can hide address-number errors. Its punctuation expression also appears malformed. Correcting it changes the latest scores by about 0.002–0.006 without changing the ranking. We report the supplied version and keep exact-span scores alongside it.

## How different are competition posts?

Competition posts average about 395 characters, versus 102–119 in the sampled source data. With the tested 16,000-piece BPE tokenizer, 36 of 477 posts exceed 256 pieces and five exceed 512. Synthetic tests confirm the windows cover long texts, but do not measure extraction quality on real long posts.

### Adversarial validation with the word BiLSTM

We trained classifiers to tell source text from competition text. This is called adversarial validation. AUC measures how well the classifier ranks the two groups: 0.5 is chance, 1.0 is perfect separation. The classifier uses the same word BiLSTM design as the leading extractor, but outputs one yes/no answer per text instead of location tags. It starts from random weights; it does not reuse an extraction model or evaluate location predictions.

The experiment has **12 BiLSTM fits**: two source languages × two conditions × three seeds (3012026–3012028).

| Condition | Training texts per class | Held-out texts per class | What is controlled |
| --- | ---: | ---: | --- |
| Primary control | 120 | 40 | Lengths matched; URLs, emoji, and line breaks masked |
| Broader check | 300 | 100 | Nothing; unmasked texts |

Every fit uses six epochs, a vocabulary learned only from its own training texts, and at most the first 256 word tokens. Training ran locally on CPU and took about 88 seconds in total.

| Condition | Ukrainian AUC, mean (seed range) | Russian AUC, mean (seed range) |
| --- | ---: | ---: |
| Length matched, cues masked | 0.839 (0.689–0.915) | 0.934 (0.869–0.990) |
| Broader, unmasked texts | 0.937 (0.926–0.946) | 0.967 (0.941–0.982) |

### What the results mean

**The collections differ, even after the controls.** This confirms the domain warning with the leading extraction architecture. The Ukrainian controlled result varies a lot across seeds, so its mean is not precise. The two conditions differ in both texts and training size, so the gap between them does not isolate the effect of length or masking. Two of the 100 competition posts in each broader holdout were cut at 256 tokens; this checks the start of each text, not full-post extraction.

**Neither architecture is consistently better at this task.** BiGRU and BiLSTM both separate the collections. BiLSTM scored higher in two of three Ukrainian runs; BiGRU scored higher in all three Russian runs. This comparison checks whether the collections differ; it does not tell us which model extracts locations better.

**Language explains much of the Russian result.** Competition posts are mostly Ukrainian, so a simple check for Ukrainian-specific letters already separates Russian source sentences from them (AUC 0.938, close to BiLSTM's 0.934). The same check fails on Ukrainian source sentences (AUC 0.450), yet BiLSTM still separates them. The Ukrainian comparison therefore gives stronger evidence of differences beyond language, although it does not show which features the model uses.

**Uncertainty is larger than the tables suggest.** The samples are small. Repeated runs use the same training and evaluation texts; only the seed changes, which affects starting weights, dropout, and batch order. So seed ranges show training variability, not variability from new text samples.

Good extraction scores on the source data may not carry over to competition posts. These results show a transfer risk; they do not measure how much extraction F1 drops or estimate the Kaggle score.

### Checks and supporting numbers

| Controlled comparison | BiGRU AUC range | BiLSTM minus BiGRU AUC, paired seed range | Ukrainian-letter check AUC |
| --- | ---: | ---: | ---: |
| Ukrainian source versus competition | 0.734–0.873 | −0.045 to +0.094 | 0.450 |
| Russian source versus competition | 0.894–0.992 | −0.036 to −0.002 | 0.938 |

- Positive differences favor BiLSTM; negative ones favor BiGRU. The GRU/LSTM comparison uses the same evaluation texts and the same word-to-ID mapping.
- Replaying the six earlier GRU runs reproduced their saved probabilities to within 1e-6.
- Audits found no training/evaluation overlap in source groups, documents, exact texts, normalized templates, or encoded text beginnings.
- AUC measures ranking, not accuracy at a fixed 0.5 cut-off. One controlled Russian BiLSTM run had AUC 0.869 but balanced accuracy of only 0.500. The cut-off was not tuned.
- Paired uncertainty intervals resample the same held-out groups for both architectures. They describe the saved models, not variation from new samples or other seeds. The [domain notebook](notebooks/domain_analysis.ipynb) records per-run comparisons and intervals.
- These classifiers are separate from the location extractors. Their vocabularies include their training competition posts; extractor vocabularies do not. No extraction model was retrained, and the final evaluation set stayed reserved. Per-run scores, predictions, and audits are saved locally in `output/eda_comparison/domain_word_bilstm/`.

## Kaggle submission results

The Kaggle CLI showed the following completed submissions on 4 November 2026. These are **public leaderboard scores**, calculated on hidden competition labels. They are separate from the source-data cross-validation scores above.

| File | Submission ID | Public F1 | What it contains |
| --- | ---: | ---: | --- |
| `submission_bilstm_word_vote3.csv` | 57999555 | **0.41086** | Word BiLSTM predictions accepted by at least two of three fold models |
| `submission_bigru_pilot_v2.csv` | 57999600 | 0.40449 | Earlier word BiGRU pilot predictions |
| `submission_format_probe.csv` | 57999516 | 0.00000 | Empty lists for every post; a format check |

“Three fold models” means three independent BiLSTM fits from the grouped cross-validation experiment. Model 0 trains on folds 1 and 2, model 1 on folds 0 and 2, and model 2 on folds 0 and 1. Each model has its own word vocabulary and saw about 10,500 labeled training rows. For the submission, all three predict the same 477 competition posts. A location string is kept if at least two predict that exact wording. This is a vote over three models, not a model trained three times longer or on all source rows.

The BiLSTM vote leads the BiGRU pilot by 0.00637 public F1, but the submissions differ in training sample and voting as well as architecture. They are not a controlled GRU-versus-LSTM comparison. The single-model BiLSTM CSV exists locally but has no recorded completed Kaggle score. Private leaderboard scores are still blank.

The local CSVs for the two scored model entries contain 477 rows and parseable location lists. The same `v2` BiGRU filename and empty-list probe had earlier failed uploads, then completed on 4 November; the CLI does not identify the backend change that allowed them through. The later `submission_bigru_pilot_v3.csv` still has an error status: 130 of its 477 rows do not parse as CSV location lists because lists containing commas are not quoted as one cell. A completed submission shows the format is accepted; its public F1 is the only competition performance evidence available here.

## Repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/` | Executed data, domain, and extraction analyses |
| `core/` | Shared parsing, scoring, split, and domain code |
| `workflows/` | Runnable audits, exports, and result summaries |
| `colab/` | Self-contained scripts uploaded for GPU experiments |
| `config/` | Split rules and recorded review decisions |
| `archive/` | The earlier role assignment and the original experiment plan |
| `output/` | Local artifacts, ignored by Git except the two split manifests (Git LFS) |
| `submissions/` | Generated Kaggle submissions, ignored by Git |

The current roles come from `workflows/roles_v2.py`. `archive/roles.py` is still needed to rebuild its version 1 input; do not use version 1 as the current split. The [dataset guide](../data/kse-ua-location-extraction-2026/README.md#team-setup-and-experiment-splits) explains how to use the shared manifests and the files in `config/`.

## Split requirements for new experiments

Follow the [required split and exclusion policy](../README.md#required-split-and-exclusion-policy-for-future-experiments). In short: use `output/role_manifest_v2.parquet`, column `role_v2`; keep every `exclude_*` row out of training, tokenizer fitting, and evaluation in both languages; and never recover extra training data by reading the raw CSVs without joining the manifest.

The grouped CV manifest, `output/cv_manifest_v1.parquet`, was built from role version 2 and samples only `train` and `development` rows; its “v1” does not refer to role version 1. If a manifest is missing, download it through Git LFS or rebuild it through the audited workflow below; never substitute a fresh random split.

## Reproducing the analysis

### Run locally

From the project root, place the supplied files in `data/kse-ua-location-extraction-2026/`, install `eda/requirements.txt`, and run workflows as modules:

```sh
python -m eda.workflows.cv_slices
python -m eda.workflows.cv_exposure
python -m eda.workflows.run_notebook eda/notebooks/extraction_pilot.ipynb
```

The notebook runner uses a local Unix socket. The first data audit scans about nine million source rows and takes several minutes. The extraction notebook reads saved results and does not retrain models. Notebook code reads local artifacts from `output/`, so those must be regenerated before every cell can be rerun.

To rebuild everything from raw data, run the full audit and original split, then the role and grouping audits, version 2 role assignment, fold export, and extraction exports. The [original experiment plan](archive/followup_plan.md) records that sequence and early decisions; its counts are historical, and this guide holds the current ones.

### Extraction comparison

Generate the frozen row selection from the existing CV bundle:

```sh
EDA_COMPARE_DATA=eda/output/extractor_bundle_v1.jsonl.gz \
EDA_COMPARE_OUT=eda/output/eda_comparison \
python -m eda.colab.eda_comparison
python -m eda.workflows.representation_comparison
```

The selection records the exact training row IDs for each fold. On Colab, upload the bundle, `colab_extractor.py`, and `eda_comparison.py` to `/content`, and place `selection.json` at `/content/eda_comparison/selection.json`. Set `EDA_COMPARE_MODE=run` and run `eda_comparison.py`; completed fits are recorded in `/content/eda_comparison/results.json`, so the script can resume after an interruption. After downloading results and predictions to `eda/output/eda_comparison/`, summarize them:

```sh
python -m eda.workflows.summarize_comparison
python -m eda.workflows.comparison_error_analysis
python -m eda.workflows.comparison_score_sensitivity
```

The extraction comparison uses only the frozen CV rows; it never reads the final evaluation set.

### Domain checks

The earlier word BiGRU check runs with `python -m eda.workflows.domain_recurrent_control`, followed by `python -m eda.workflows.domain_cue_baselines` to compare it with simple language and length cues. The word BiLSTM experiment runs with `python -m eda.workflows.domain_bilstm`; use `--max-new-runs 1` to time the first fit. It resumes only if inputs, code, and settings are unchanged.

## Make a Kaggle submission

From the project root, with the supplied competition files and local checkpoints in place:

```sh
python -m eda.workflows.make_submission
python -m eda.workflows.make_submission --checkpoint fold0_mixed_lstm_word
```

By default, the three fold models of the word BiLSTM each predict locations, and a location is kept when at least two agree on its exact wording. Each model was trained on about 10,500 labeled rows, **not** on the full training data. The second command runs just one model. Use `--output path/to/submission.csv` to choose another CSV path.

The commands save a Kaggle-ready CSV and a neighboring compressed raw JSONL file with each model's spans and confidence scores; a neighboring JSON file records the checkpoints and voting rule. They also print and save a check on the 26 labeled examples in `eda/output/submission_sanity_labeling_sample.json`. This small qualitative sample is not a leaderboard score; report any Kaggle results explicitly as leaderboard scores.

`eda/submissions/` and `eda/output/` are ignored by Git (apart from the shared split manifests). Checkpoints are local only; a fresh clone must obtain them separately. No command here uploads a submission.
