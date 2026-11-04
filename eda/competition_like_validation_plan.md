# Plan: competition-like validation

Status: part 1 (the submission script) is done. Part 2 hasn't been started yet, so please check with the team before running any of it.

## Why we need this

In cross-validation the word BiLSTM gets about 0.65 F1. The catch is that we validate on the same kind of text we train on, and the competition posts are quite different. They average about 395 characters, while our source sentences average 102–119. Most of them have several lines, a fair number end with links or channel footers, and nearly all are in Ukrainian (428 of the 477 posts have Ukrainian-specific letters).

The adversarial validation in the [EDA README](README.md#how-different-are-competition-posts) shows the same thing: a classifier can tell our text from competition text with AUC between 0.84 and 0.97. We also ran the model on the 26 competition examples that have hand labels and got about 0.23 F1. Twenty-six posts is not much, but it's a long way from 0.65.

In other words, a good CV score doesn't tell us much about the leaderboard. We want two things:

1. A submission script we can trust, which also keeps the raw model outputs.
2. A validation set that looks more like the competition, so we can check whether training on more competition-like data actually helps.

A few terms we use below:

- *Competition-likeness* is a score for each source sentence showing how much it resembles a competition post. It comes from a classifier trained to separate the two.
- The *competition-like validation set* is made of the source rows that score highest. They still have our location labels, and we only evaluate on them.
- *Competition-mix F1* weights the languages the way the competition does, roughly 90% Ukrainian and 10% Russian.

## Rules

- Follow the [split policy](../README.md#required-split-and-exclusion-policy-for-future-experiments). Don't use `final` rows or anything marked `exclude_*`, and don't train on the full data.
- The 477 competition posts are only used to train the competition-likeness scorer. They must not end up in the extractor's training data or in its tokenizer.
- The 26 labeled examples (`data/kse-ua-location-extraction-2026/labeling_sample.csv`) are for reporting and sanity checks. Don't tune anything on them.
- We don't pick models by leaderboard score, only by local validation.
- Don't commit, push or upload to Kaggle unless the team has agreed.

## Part 1: submission script (done)

We rewrote `eda/workflows/make_submission.py`. How to run it is described in [Make a Kaggle submission](README.md#make-a-kaggle-submission).

By default it uses the three cross-validation word BiLSTMs (`fold0`, `fold1`, `fold2`) and keeps a location if at least two of them predict exactly the same string. To run one model only, pass `--checkpoint NAME`. Predictions have to be exact substrings of the post. Duplicates are removed, the order of first appearance is kept, and nothing else is filtered.

Next to every CSV the script writes `<name>_raw.jsonl.gz`. For each post it stores the spans from each model (start, end, text and mean tag probability) together with the final voted list. A `<name>_raw_meta.json` file stores the checkpoint hashes and settings. With these files we can try different post-processing without running the models again.

Before finishing, the script checks the CSV: it must have 477 rows in test order, and every cell must parse as a JSON list. It also predicts the 26 labeled examples and writes the score to `output/submission_sanity_labeling_sample.json`.

There's still one problem. Kaggle rejects all our 2026 submissions with "Scoring session had non-zero exit code", including a file where every prediction is empty. Nobody has a real score on the 2026 leaderboard, while the 2025 competition uses the same files and scores normally. Most likely the problem is on Kaggle's side, or it expects a format we haven't been told about. We need to ask the instructor.

## Part 2: competition-like validation (not started)

### Step 0: split the spare data

The cross-validation sample doesn't use most `train` rows. We split their groups into two pools with a fixed, seeded hash. One pool is used to train the scorer, and the validation set is drawn from the other (the candidate pool). We split first because a scorer that has already seen a row would rank it unfairly. The Ukrainian candidate pool has to be large enough to give about 20k rows.

### Step 1: train the scorer

We train one scorer per language, separating source rows from the scorer pool from the 477 competition posts. The scoring is cross-fitted: each row is scored by a model trained on other folds, never by one that saw it.

We'll try four scorers, from cheapest to most expensive:

1. character TF-IDF with logistic regression;
2. the same with extra length and formatting features;
3. a word BiLSTM classifier trained on more source rows than our earlier run;
4. pretrained multilingual sentence embeddings (e.g. `multilingual-e5-small`) with logistic regression.

Each is run on raw text and on a controlled version, where lengths are matched and links, emoji and line breaks are masked. We use the raw version unless its scores mostly follow text length (Spearman correlation with length of 0.7 or more), in which case we switch to the controlled one. The winner is the scorer with the best cross-fitted AUC that also passes the checks in step 2.

This scorer is only used for ranking rows. For the assignment we still report the BiLSTM adversarial validation, because the task asks for the same architecture as the main model.

### Step 2: check the scorer

Before trusting the ranking, we check three things:

- Scorers trained on different folds should agree, with at least 70% overlap in their top 25% of rows.
- The 26 labeled competition examples should rank above most source rows.
- We write down the expected interval width and the smallest F1 difference the set can detect, before looking at any results.

### Step 3: build the set (v1)

| Language | Size | Taken from |
| --- | --- | --- |
| Ukrainian | about 20,000 rows | top-ranked `development` rows, plus candidate-pool `train` rows outside CV |
| Russian | top 25%, at most 10,000 rows | `development` rows outside CV |

We take whole groups only and drop any group that overlaps the CV sample. We also build a random set of the same size from the same pool. If scores change on the competition-like set, the random set tells us whether that is about competition-likeness or just about using different rows.

The set is saved as `eda/output/competition_like_v1.parquet` and shared through Git LFS. Its rows get a new evaluation-only role: they can't be used for training, tokenizer fitting or CV sampling, and they don't replace the `final` evaluation.

We made the Ukrainian part big on purpose: its confidence intervals come out about 30% narrower than with half as many rows. Even so, we'll treat F1 differences under about 0.02 as inconclusive.

### Step 4: rescore the models we already have

We score all 15 saved fits on the competition-like set and on the random set, reporting F1 per language, competition-mix F1, group-bootstrap intervals, and whether the ranking differs from CV. This step needs no training.

### Step 5: does competition-like training data help?

We keep the current word BiLSTM recipe (about 10.5k labeled rows, 3 epochs) and only change how the training rows are chosen. Each setup is run with three seeds:

| Setup | How training rows are picked |
| --- | --- |
| Random | random sample, as we do now |
| Language mix | random, but about 90% Ukrainian like the competition |
| Weighted | competition-like rows are more likely to be picked; weights are capped so a few rows can't take over |

Training runs locally on CPU or MPS. We time the first run, and if everything together would take more than an hour we stop and move it to Colab. Every run is evaluated on both sets. The 26 examples are reported too, but only for reference.

### Step 6: new submission (optional)

If one setup is clearly better, by more than the 0.02 noise level, we make a new CSV with the part 1 script. We upload it only after the team agrees.

### Step 7: docs

- Write a short decision record in `docs/adr/0001-competition-like-validation-set.md` explaining why the set exists and what it may be used for.
- Add the evaluation-only role to the split table in the root README and to the LFS file list in the dataset README.
- Put the results in their own sections of `eda/README.md`, and add one line to the root README saying what we should do because of them.
