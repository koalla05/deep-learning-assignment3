
## EDA Main findings

- **Word-wise tokenization is the strongest baseline in these experiments.** It beats BPE, with both GRU and LSTM, including when all models trained on the same examples.
- **Word BiLSTM has a small lead over word BiGRU.** Its combined F1 is 0.646 versus 0.628. This deserves a further test with another seed and more training.
- **Unfamiliar names remain difficult.** About 40% of labeled location mentions contain a word outside the learned vocabulary. BPE covers the tokens better but often predicts fragments or extra names. On Ukrainian location names absent from the training folds, the leading mixed-language models recover only about a quarter.
- **Competition posts differ substantially from the training sentences.** They are longer and contain lists, line breaks, and footers. The completed word BiLSTM domain check still separates the collections after length and formatting controls. Good source-data scores do not establish good competition performance.
- **Annotation and duplicate problems affect evaluation.** We preserve the supplied labels, record questionable examples, and keep linked documents together when splitting data.

## Where to read the analysis

| Notebook | Questions it answers |
| --- | --- |
| [Data audit](eda/notebooks/data_audit.ipynb) | Labels usability. Which texts repeat. How should we split the data? What does tokenization lose? |
| [Extraction experiments](eda/notebooks/extraction_pilot.ipynb) | How were the models compared? Which differences persist across folds? What errors remain? |
| [Source and competition differences](eda/notebooks/domain_analysis.ipynb) | Can a classifier distinguish the collections after controlling for length and formatting? |

The notebooks retain earlier executed tables and plots; the newest BiLSTM domain-result cells have not yet been executed, but their saved experiment results are summarized below. Earlier checks are kept as experiment history. The [EDA guide](eda/README.md) contains run instructions, and the [comparison protocol](eda/model_plan.md) records the extraction comparison's scope.

## Data and splits

| Data | Rows | Purpose |
| --- | ---: | --- |
| Ukrainian source | 1,010,000 | Labeled source text |
| Russian source | 8,028,840 | Labeled sentences with document IDs |
| `test.csv` | 477 | Kaggle prediction and submission |
| `labeling_sample.csv` | 26 | Separate labeled examples for manual inspection |

Every `locations` field in `test.csv` is the placeholder `[]`. The real labels are hidden. The 26 labeled examples are a separate file, with no exact text overlap with the test file; they cannot tell us the test-set F1.

The Ukrainian Parquet is a processed view of the same source rows. Its flat location tags lose the distinction between adjacent names. Converting original character offsets to BIO tags—beginning, inside, or outside a location—raises reconstruction F1 from 96.57% to 99.49% on retained Ukrainian training rows. This checks how well the representation preserves labels, before training a model.

### Training, development, and final evaluation

| Set | Ukrainian rows | Russian rows |
| --- | ---: | ---: |
| Training pool | 959,912 | 6,415,583 |
| Development | 39,912 | 119,789 |
| Final local evaluation | 10,000 | 32,014 |

The supplied Ukrainian validation flag defines its 10,000 final evaluation rows. Development adds examples from the original training data. Both languages were checked for exact duplicates, normalized word-sequence equality, and sampled near-duplicates with reviewed decisions. Linked Ukrainian texts stay together; Russian grouping additionally joins whole documents and follows duplicate links transitively. About 1.34 million Russian rows were excluded from training and development to prevent linked examples crossing sets. Generic news snippets connect one particularly large group of 1.32 million rows; these are connected document rows, not 1.32 million identical sentences. This conservative choice changes the retained data: excluded rows linked to final evaluation contain locations more often than retained training rows (33.8% versus 30.2%).

These sizes are practical allocations, not a statistical power calculation. The final set was included in descriptive data checks before being reserved, but the candidate models have not been scored on it. Exact overlap checks passed; the sampled search for similar texts cannot rule out every paraphrase.

For the experiments, we sampled **63,388 rows from the training and development pools** and divided them into three folds. Related rows stay together, and each fold has similar row and positive-example counts within each language. Each model trains using two folds and is evaluated on the third. This is grouped cross-validation with approximate balancing, rather than a random sentence split. Final evaluation rows are excluded throughout.

### Required split and exclusion policy for future experiments

**Note:** For future experiments, use `eda/output/role_manifest_v2.parquet` and its `role_v2` assignments as the source of truth, not a fresh sentence-level split of the raw CSVs. Join it to the original source rows by `(source, row_id)`. Row IDs refer to the unchanged source CSV ordering; verify source-file fingerprints before reusing them.

| `role_v2` | Ukrainian rows | Russian rows | Meaning and permitted use |
| --- | ---: | ---: | --- |
| `train` | 959,912 | 6,415,583 | Eligible pool for supervised training and tokenizer fitting; experiments may use smaller samples. |
| `development` | 39,912 | 119,789 | Hold out for model selection in fixed-split experiments. May enter fitting only through grouped CV or an explicitly authorized final refit. |
| `final` | 10,000 | 32,014 | Reserved evaluation after the recipe is frozen and evaluation is authorized; never fitting or model selection. |
| `exclude_conflict` | 0 | 22,562 | Identical text has different supplied location-marker fields. Excluded rather than resolving labels silently. |
| `exclude_duplicate` | 48 | 98,608 | Originally assigned to training, but the exact text also appears in a supplied or constructed holdout. |
| `exclude_final_duplicate` | 4 | 0 | Ukrainian development candidate whose exact text appears in the supplied final evaluation split. |
| `exclude_development_group` | 97 | 15,169 | Training row connected to a development group, with no final-evaluation member taking priority. |
| `exclude_final_group` | 27 | 1,325,115 | Training or development row connected to a group containing final-evaluation rows. |

Each row has one role; the exclusion counts are separate, not cumulative. **Every `exclude_*` role remains ineligible for supervised training, tokenizer/vocabulary fitting, and quantitative evaluation, including in stage 2.** Raw files retain these rows for provenance; their presence is not permission to reuse them. Select eligible roles explicitly rather than treating every non-final row as training data.

1. For a fixed development holdout, fit the extractor and tokenizer on eligible `train` rows only. For the existing grouped CV, reuse `eda/output/cv_manifest_v1.parquet`, which derives from role version 2, and fit both only on the two training folds. Development rows enter training only through that grouped CV protocol or a separately authorized final refit.
2. Keep accepted duplicate groups and Russian documents intact. Before each new experiment export, verify zero group, document, and exact-text overlap across its training and evaluation partitions, and verify that no excluded rows enter either partition.
3. Keep `final` rows out of extractor training, tokenizer fitting, and model selection. Score them only after the training recipe is frozen and final evaluation is explicitly authorized. Keep the 26 labeled examples qualitative; competition text is used only in the separate domain diagnostics, not extraction training or vocabulary fitting.
4. Record the role/fold manifest hashes, selected row IDs, role counts, and overlap checks with each experiment. Any partition change requires an explicit new version and audit, not a silent fallback to the original split. Changing language mixture or training budget does not relax these exclusions.

## How the experimental models were trained

All extraction models started from random weights, including their word embeddings. We used PyTorch's standard bidirectional GRU and LSTM layers. “Bidirectional” means that the model reads context on both sides of each token. A final layer predicts one of the three BIO tags, which are converted back to character spans and location strings.

The latest comparison contains **15 fits**: word and BPE tokenization with both recurrent layers, plus a word BiGRU trained on Ukrainian labels only, each repeated across three folds.

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

Each tokenizer sees text from the two training folds, including rows outside the smaller supervised sample. It never sees the held-out fold or competition texts. The Ukrainian-only control shares the mixed-language vocabulary: “Ukrainian only” describes its supervised examples, not its vocabulary.

We select rows whose location boundaries both tokenizers can represent. Held-out examples retain their original labels. Within training windows, labels for locations cut by a window edge are ignored. Predictions use the highest-scoring tag and a fixed window-merging rule. We use the model after epoch three; validation scores do not choose checkpoints or decoding settings.

Every fit stays below 300,000 training tokens per epoch. With the same mixed-language rows, word models process about 201,000 tokens and BPE models about 286,000. GRU has 1,648,323 parameters and LSTM has 1,685,571. The comparison therefore holds examples and hidden width constant, but not token count or parameter count. Checkpoints, tokenizers, selected row IDs, predictions, losses, and timings are saved locally.

## Extraction results

Scores below combine predictions from all three held-out folds. F1 balances precision (how many predictions are correct) and recall (how many labeled locations are recovered). The main score compares normalized sets of location strings within each text. Exact-span F1 also requires the character boundaries to match.

| Configuration | Ukrainian F1 | Russian F1 | Combined F1 | Exact-span F1, combined |
| --- | ---: | ---: | ---: | ---: |
| Mixed, word BiGRU | 0.581 | 0.664 | 0.628 | 0.626 |
| Mixed, BPE BiGRU | 0.513 | 0.576 | 0.548 | 0.538 |
| **Mixed, word BiLSTM** | 0.600 | 0.681 | 0.646 | 0.644 |
| Mixed, BPE BiLSTM | 0.519 | 0.581 | 0.553 | 0.546 |
| Ukrainian labels only, word BiGRU | 0.631 | 0.165 | 0.324 | 0.323 |

Word tokenization wins on both languages in every fold. Its combined advantage is 0.080 with GRU and 0.093 with LSTM. Resampling held-out groups gives 95% intervals of 0.073–0.087 and 0.086–0.100. The word LSTM advantage over word GRU is 0.018, with an interval of 0.014–0.021. These intervals describe the saved models; they do not account for another training seed.

Replacing Russian examples with Ukrainian examples improves Ukrainian F1 by 0.050 but reduces Russian F1 by 0.499. This also doubles Ukrainian supervision and changes which names the model sees. It does not prove that adding Russian data is harmful when Ukrainian exposure is kept fixed. The combined score pools correct, extra, and missed locations across both languages; it is not an equal average of their F1 scores or an estimate of Kaggle performance.

An earlier experiment gave models equal token budgets, allowing word GRU to see about 15,800 examples per fold versus 11,100 for BPE. It scored 0.655 versus 0.559. The latest comparison confirms that the word advantage persists with equal examples. Because the samples also changed, the difference between those experiments does not isolate the benefit of seeing more rows.

### What the errors show

BPE sometimes predicts fragments such as `Ста` or `Чер`. All models miss some lists of places. Word LSTM makes fewer extra predictions on rows without labeled locations than word GRU, and both make fewer than BPE GRU. Some apparent errors are questionable annotations: one reviewed Russian sentence mentions `Румынии` without a location label.

A single application of the earlier word pilot disagreed with the 26 separate labeled examples on 17 rows. Missed streets, neighborhoods, lists, and footer predictions recur. This small, nonrandom sample helps identify failure types; 17/26 is not a general error rate.

Note: The scoring code supplied for the competition removes digits and repeated strings within a text, so it can hide address-number errors. Its punctuation expression also appears malformed. Correcting that expression changes the latest scores by about 0.002–0.006 without changing the ranking. We report the literal supplied version and keep exact-span scores alongside it. 

## How different are competition posts?

Competition posts average about 395 characters, versus 102–119 in the sampled source data. With the tested 16,000-piece BPE tokenizer, 36 of 477 posts exceed 256 pieces and five exceed 512. Synthetic tests confirm window coverage, but do not measure extraction quality on real long posts.

### Adversarial validation with the word BiLSTM

We trained classifiers to distinguish source text from competition text. This is adversarial validation: an AUC of 0.5 means chance-level ranking, while 1.0 means perfect separation. The completed follow-up uses the word BiLSTM encoder architecture of the leading mixed-language extractor, with a binary classification head instead of token-level location tags. It starts from random weights; it does not reuse an extraction checkpoint or evaluate location predictions.

The experiment contains **12 BiLSTM fits**: two source languages, two conditions, and three seeds (3012026–3012028). The primary control matches text lengths and masks URLs, emoji, and line breaks, using 120 texts per class for training and 40 per class held out. The broader diagnostic uses unmasked texts, with 300 per class for training and 100 held out. Every fit uses six epochs, a vocabulary learned only from its domain-training texts, and at most the first 256 word tokens. Training ran locally on CPU and took about 88 seconds in total.

| BiLSTM condition | Ukrainian AUC, mean (seed range) | Russian AUC, mean (seed range) |
| --- | ---: | ---: |
| Length matched, cues masked | 0.839 (0.689–0.915) | 0.934 (0.869–0.990) |
| Broader, unmasked texts | 0.937 (0.926–0.946) | 0.967 (0.941–0.982) |

Separation remains after the controls, confirming the domain warning with the leading extraction architecture. The Ukrainian controlled result varies substantially across seeds; its mean is not a precise estimate. The broader and controlled conditions differ in both membership and training size, so their AUC difference does not isolate the effect of length or masking. Two of the 100 competition posts in each broader holdout were truncated; this checks text prefixes, not full-post extraction or window merging.

#### What the results mean

Both BiGRU and BiLSTM can tell source sentences apart from competition posts. BiLSTM scored higher in two of the three Ukrainian runs, while BiGRU scored higher in all three Russian runs. Neither is consistently better across both languages and repeated runs. This comparison checks whether the collections differ; it does not tell us which model extracts locations better.

Language provides an easy clue when comparing Russian source sentences with the mostly Ukrainian competition posts. A simple check for Ukrainian-specific letters already separates them well. That same check does not usefully distinguish Ukrainian source sentences from competition posts, but BiLSTM still does. The Ukrainian comparison therefore gives stronger evidence of differences beyond this simple language clue; it does not identify exactly which features the model uses.

The evaluation samples are small, and repeated runs reuse the same evaluation texts rather than testing new samples. The seed was used for training init state. **Good extraction scores on the source corpora may not carry over to competition posts.** 

#### Checks and experiment records

The controlled GRU/LSTM comparison uses the same evaluation texts and token-to-ID mapping. Replaying the six earlier GRU runs reproduced their saved probabilities within 1e-6. Audits found no training/evaluation overlap in accepted source groups, documents, exact texts, normalized templates, or encoded input prefixes.

The numerical checks behind the interpretation are:

| Controlled comparison | BiGRU AUC range | BiLSTM minus BiGRU AUC, paired seed range | Ukrainian-letter indicator AUC |
| --- | ---: | ---: | ---: |
| Ukrainian source versus competition | 0.734–0.873 | -0.045 to +0.094 | 0.450 |
| Russian source versus competition | 0.894–0.992 | -0.036 to -0.002 | 0.938 |

Positive paired differences favor BiLSTM; negative ones favor BiGRU. The Russian letter-indicator score is close to the BiLSTM mean of 0.934, making the language clue an important qualification to that result.

AUC measures ranking, not accuracy at the fixed 0.5 decision threshold. For example, one controlled Russian BiLSTM run had AUC 0.869 but balanced accuracy of only 0.500. The threshold was not tuned. Paired uncertainty intervals resample the same held-out groups for both architectures; they describe the saved models, not variation from new training samples or retraining with other seeds. The [domain notebook](eda/notebooks/domain_analysis.ipynb) records the per-run comparisons and intervals.

These classifiers were trained separately from the location extractors. Their vocabularies include their training competition posts, but extractor vocabularies do not. No extraction model was retrained, and the final local evaluation set remains reserved. Per-run scores, predictions, and audits are saved locally under `eda/output/eda_comparison/domain_word_bilstm/`.

## What to carry into the next stage

1. Keep word BiGRU as the reproducible baseline and retest word BiLSTM with another seed and a larger budget.
2. Vary Russian exposure while keeping Ukrainian training examples fixed. Report each language separately.
3. Test pretrained word embeddings against random initialization, measuring coverage and recall on unfamiliar names. No pretrained weights were used here.
4. Review ambiguous labels and long-post errors before changing annotation or decoding rules. Preserve the original labels for comparison.
5. Freeze the training recipe before using the final local evaluation set or making a new submission.

Raw data, generated artifacts, checkpoints, and submissions stay outside Git. See the [EDA guide](eda/README.md) to reproduce the checks and read the notebooks from their saved results.
