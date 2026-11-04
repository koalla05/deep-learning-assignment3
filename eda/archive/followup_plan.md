> Historical plan and experiment record. Current findings and file paths are in the [project README](../../README.md) and [EDA guide](../README.md).

# Further EDA experiments

Plan agreed on 2026-11-03: targeted review of about 100 examples, controlled diagnostics, a bounded extraction pilot, and three-fold grouped cross-validation of shortlisted approaches. Local analysis may take a few hours; Colab is the preferred option for training. Longer local training requires a separate agreement.

Progress: annotation reconstruction and review, duplicate and coverage audits, matched tokenizer diagnostics, controlled domain analysis, version 2 roles, the bounded extractor pilot, and three grouped CV folds are complete. A one-time review used the 26 labeled competition-style examples. Results and limitations are summarized in the root README. Detailed CV error slices remain to be analyzed. The Colab runtime was released after artifacts were saved. Full training and reserved final evaluation remain for the next stage.

The agreed annotation policy preserves original labels and flags uncertain categories. Exclude clearly unrepresentable rows only from pilot training, with a tokenizer-specific exclusion artifact; development is scored against original labels. The supplied-token audit identifies 1,638 such Ukrainian training rows. Other tokenizers must establish their own representability rather than inherit this list blindly.

The first EDA established structural annotation validity, removed exact training–validation text overlap, and found substantial differences between source sentences and competition posts. The next round should determine which data and representation choices are worth carrying into recurrent model training.

## Training, model selection, final evaluation, and submission

The existing manifest distinguishes training from holdouts. Before fitting an extractor, add a versioned role assignment that separates development from final local evaluation:

| Role | Ukrainian rows | Russian rows | Purpose |
| --- | ---: | ---: | --- |
| Training pool | 960,035 | 7,747,985 | Fit model parameters and tokenizers; draw smaller pilot samples from here |
| Development | 39,913 | 127,671 | Choose preprocessing, model settings, stopping point, and decoding |
| Final local evaluation | 10,000 | 32,014 | Evaluate the selected configuration after model-selection decisions are fixed |
| Competition submission | 477 total competition texts | — | Generate predictions for the supplied unlabeled `test.csv` |

These are available pool sizes, not a recommended training mixture. The Russian corpus should not receive eight times the training weight merely because it is larger. The holdout sizes are practical starting allocations, not the result of a statistical power calculation: the initial sampling used approximately 4% of Ukrainian training texts and 2% of Russian documents. Assess adequacy through independent group counts, coverage of difficult cases, and uncertainty in measured model differences. Stratification improves balance but does not establish statistical power.

The supplied 10,000 Ukrainian validation rows become final local evaluation. Four texts occur in both Ukrainian holdouts; exclude their four copies from development, leaving 39,913 rows. The current exact train–holdout overlap remains zero.

Subdivide the Russian holdout by connected groups: all sentences from a document stay together, and documents sharing an exact text stay together. A read-only simulation assigning about 20% of those groups to final evaluation yields 32,014 rows from 2,963 documents, with the other 127,671 rows from 11,768 documents assigned to development. The simulation verifies zero exact-text and document overlap between those roles. It uses seed 3012026, the lexicographically smallest document ID as each component's key, and the first eight hexadecimal digits of MD5(`3012026:ru-final:<key>`) modulo 10,000, selecting values below 2,000 for final evaluation.

Preserve the existing manifest and materialize these roles as a new version before pilot metrics are computed. Record the four Ukrainian exclusions. Any later annotation or near-duplicate findings that justify membership changes require another explicit version.

The final evaluation rows have already contributed to descriptive EDA; they are reserved from model selection from this point onward, not an unseen external benchmark. Further example review and tokenizer comparisons should use training and development data. Report Ukrainian and Russian scores separately. Source scores do not directly estimate performance on competition posts.

The 26 supplied labeled examples remain a qualitative sample. The competition test set is not a random slice of our labeled data: it is the separate 477-text file with hidden labels. Submit predictions from the selected checkpoint; public leaderboard feedback must be reported as such. If we later refit on training plus development, freeze the recipe first and keep final evaluation excluded, then report the score of the actual refitted checkpoint.

The assignment requires consistent splits and local validation plus leaderboard results, but does not specify a number of cross-validation folds. Report fixed-holdout pilot results and the three-fold results separately. EDA diagnostics without extraction predictions do not have a meaningful leaderboard score.

## Cross-validation protocol

Use the fixed development set for initial feasibility checks and narrowing the candidate approaches. Then compare the shortlist with three-fold cross-validation on one reproducible, manageable subset of the combined training and development pool. Final local evaluation and the 26 qualitative examples are excluded. Within this subset, fresh fold assignments replace the earlier training/development roles.

Build groups before assigning folds. All Russian sentences from one document remain together, and documents linked by duplicate text form one connected group. Ukrainian duplicate texts remain together. Include any near-duplicate links accepted by the preceding audit. Balance source and presence of a location approximately across folds, then inspect entity counts, length distributions, and rare/unseen-name coverage. Group integrity takes precedence over exact proportions; report any imbalance caused by large groups.

Freeze subset membership, group IDs, fold IDs, and seed before computing comparison scores. Use the same evaluation rows in each fold for every candidate. An approach using only Ukrainian training data still receives the same evaluation slices as an approach using both sources. Match training token budgets for corpus-mixing comparisons.

For each fold, fit vocabularies, tokenizers, frequency references, and other learned preprocessing using only that fold's training rows. Use the pilot's fixed stopping recipe, or a separate grouped inner validation sample from the fold's training rows, for early stopping and threshold selection. The outer evaluation fold must not select its own checkpoint or decoding threshold.

Report the three fold scores, their mean and spread, and pooled out-of-fold precision, recall, and F1, separately by source. A spread across three folds is not itself a confidence interval. For uncertainty in paired model differences, resample independent document/duplicate groups in the held-out predictions and state that this does not capture all training variability.

Cross-validation selects the configuration. Train the selected configuration using a fixed recipe, then evaluate it on the reserved final local set. Do not return to tuning on that final score. Cross-validation within source corpora does not resolve the source–competition domain difference.

## Review and compute budget

Start with 60 randomly selected training examples stratified by source and presence of location markers (15 per stratum), plus 40 targeted training examples covering boundary disputes, suspicious negatives, and category ambiguity. Keep the selection method and reviewer judgments alongside each example. Stratified and targeted samples must not be pooled into an unweighted estimate of corpus error prevalence.

The assistant will perform the initial review and send uncertain cases to the user. Review judgments remain separate from original labels; they are not automatically treated as corrected ground truth. The user can review the full sample if needed.

Colab CLI 0.7.4 is installed and authenticated. On 2026-11-03, the account reported 103.23 compute units and no active sessions. GPU availability has not been tested. Use local execution for streaming audits and short checks, then benchmark a small pilot batch on a named Colab GPU runtime before sizing the run. Start with a T4 request, record actual hardware and throughput, retain checkpoints and results locally, and release the runtime after retrieval. Do not upgrade hardware silently if allocation fails.

Target a bounded pilot and shortlist comparison completing within about an hour after environment setup. Begin with a short timing run and project the cost of all three folds for each candidate before launching them. Set sample size and shortlist size from that estimate, before examining comparison results. Reduce the workload or discuss a longer run if the estimate exceeds the budget. Longer local training remains subject to agreement. No runtime was allocated during planning.

## What the existing results support

- Valid offsets do not establish correct or complete annotations. Missing locations in rows labeled negative remain largely unexamined.
- The Parquet audit found 1,527 token disagreements and 1,527 unaligned raw spans. Equal counts do not establish a one-to-one relationship or identify which annotation view is correct.
- The word and BPE tokenizers were fitted on fewer than 10,000 source rows. Their unknown rates use different units, so the current comparison does not establish an extraction advantage for BPE.
- Domain AUC shows that sources are distinguishable. Length, formatting, and language may explain much of the difference; a larger AUC for Russian does not demonstrate that Russian training examples are harmful.
- Exact leakage removal does not make validation rows independent. Repeated texts and sentences from the same document can still concentrate evaluation weight.
- Truncation affects 9.9% of competition texts at 256 BPE pieces. Without test labels, this does not tell us how many locations would be lost.

## Experiment sequence

### 1. Establish annotation and reconstruction reliability

Review a mixture of randomly selected positive and negative rows from each source, plus targeted cases involving raw/Parquet disagreements, addresses, landmarks, organization names, and footers. Record the reason for each disagreement. Report random-sample findings separately from deliberately selected difficult cases; the latter cannot estimate corpus error rates. Keep the 26 supplied labeled examples qualitative.

Join token disagreements to the actual unaligned spans by row and offsets. Canonicalize parsed marker sets to distinguish semantic annotation differences from serialization or ordering differences. The current count of 4,786 conflicting duplicate texts is based on raw marker strings.

Run gold span → token tags → reconstructed span checks for flat `LOC` tags and BIO tags. Measure exact-span recovery and the published normalized-string score separately, and classify every failure type. Preserve the original character offsets throughout.

**Decision:** choose a defensible training target and decoding scheme. Proceed when failures are explained and the handling policy is explicit; do not silently repair source labels.

### 2. Measure validation coverage and repeated-text influence

Materialize the agreed roles first and verify row coverage, the four Ukrainian exclusions, and zero exact-text/document overlap across roles. Use development data to investigate repeated-text concentration, entity lengths, and rare or previously unseen location strings. Build entity-frequency references from training rows only. Limit final-set checks to predefined integrity and coverage checks; reserve detailed error inspection for the final evaluation. Separate source provenance from actual language; ambiguous and mixed-language texts need their own category and manual checks.

Expand near-duplicate retrieval beyond the current small train sample, using an indexed candidate search and manual review of likely matches. Record the searched corpus size and retrieval limitations. Compare retained and excluded rows to check whether exclusions disproportionately remove particular kinds of examples.

**Decision:** define evaluation slices and the groups used for folds and uncertainty estimates. Preserve the original manifest. Any justified change to the agreed role membership must produce a separately versioned manifest and an explanation of changed counts before model comparisons begin.

### 3. Compare tokenizers with consistent measurements

Fit word and BPE tokenizers on the same training subsets and source mixtures. Increase training sample size until coverage stabilizes or the agreed budget is reached. Compare a small set of vocabulary sizes rather than a large grid.

Measure the fraction of original words containing an unknown token, character coverage, and labeled entities affected by unknowns. Also measure pieces per entity, sequence-length percentiles, and reconstruction failures. Retain each tokenizer's native unknown rate as a diagnostic with its denominator stated.

Test overlapping windows and larger length caps on labeled source examples. If long examples are scarce, use clearly identified synthetic concatenations as a boundary stress test; they do not measure performance on real competition posts.

**Decision:** shortlist tokenization and window settings using coverage, recoverable boundaries, and sequence cost. Extraction quality remains to be tested with a model.

### 4. Explain the domain-classification result

Repeat the GRU domain comparison across a small number of seeds with shared, grouped evaluation membership. Check repeated competition texts and templates before splitting. Compare the original inputs with length-matched inputs, then controlled removal of newline, URL, and emoji cues. Apply these transformations only to diagnostic copies. Inspect language composition before interpreting source-specific differences.

Report paired AUC changes, variation across runs, and the number of examples retained after matching. Avoid claiming that a lower AUC proves a better training corpus.

**Decision:** identify whether the observed separation is mainly explained by presentation, length, language, or differences that remain after those controls. Use this to select a small number of training-data hypotheses.

### 5. Test the data hypotheses with a small extractor

Hold the recurrent architecture, tokenizer, training token budget, and development evaluation fixed while comparing Ukrainian-only training with a controlled Ukrainian/Russian mixture. Compare tokenization separately so source mixing and representation effects are not confounded. Reserve final local evaluation until those choices are fixed.

Report normalized-string precision, recall, and F1, with exact-span F1 and the slices from experiment 2. Examine missed and spurious locations, including errors on negative rows. Use the 26 supplied examples only for a fixed qualitative review, not repeated model selection. Any confidence intervals must respect document and duplicate grouping.

Run the shortlisted approaches through the agreed three-fold protocol, with tokenization and source-mixture comparisons kept distinct. Record any inability to finish all folds; partial runs must not be reported as completed cross-validation.

**Decision:** choose the first full training configuration from measured extraction behavior and fold consistency. Stop after the bounded comparison; further tuning belongs to the training stage. Final local evaluation follows the chosen training run, not every EDA experiment.

## Deliverables and stopping rule

Extend the data audit with annotation, validation, and tokenizer experiments. Keep controlled domain experiments in the domain notebook and put the extraction pilot and cross-validation results in a separate notebook. Retain versioned role/fold manifests and a review table with reviewer decisions. Every experiment should retain its sample definition, input fingerprints, measurements, limitations, and resulting decision. Data, checkpoints, and generated caches stay out of Git.

Update the root findings only after results exist. Finish this round when annotation handling is explicit, span-reconstruction failures are explained, evaluation roles and folds pass overlap checks, and the bounded comparisons support a first full training configuration. Report unresolved questions alongside the evidence rather than expanding into open-ended tuning. An unresolved live-scorer punctuation question should remain explicit rather than block unrelated checks.
