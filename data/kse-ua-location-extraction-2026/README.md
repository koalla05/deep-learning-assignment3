# Name Entity Recognition dataset

## uk_geo_dataset
Ukrainian dataset [Corpora Ukrainian](https://huggingface.co/datasets/ukr-models/Ukr-Synth). 
Approximately contains 1M text samples with location and organization entities.
Structure:
- text (str): text sample.
- loc_markers (List[tuple]): list of markers of location entities. Left and right indices.
- org_markers (List[tuple]): list of markers of origanization entities. Left and right indices.
- per_markers (List[tuple]): list of markers of person entities. Left and right indices.
- is_valid (int): whether it was taken from the validation split.

## ru_geo_dataset
russian dataset based on [nerus dataset](https://github.com/natasha/nerus). 
Approximately contains 8M text samples with location and organization entities.
Structure:
- text (str): text sample.
- loc_markers (List[tuple]): list of markers of location entities. Left and right indices.
- org_markers (List[tuple]): list of markers of origanization entities. Left and right indices.
- per_markers (List[tuple]): list of markers of person entities. Left and right indices.
- doc_id (int): document id.
- sent_id (int): sentence id within a document.

## Team setup and experiment splits

Run these commands from the repository root after cloning or pulling:

```sh
git lfs install
git lfs pull
```

Obtain the original supplied `uk_geo_dataset.csv` and `ru_geo_dataset.csv` separately and place them beside this README. Raw data is not committed. Keep the original contents and row order: the manifests identify a row by `(source, row_id)`, where `source` is `uk` or `ru` and `row_id` is the **zero-based CSV record index, excluding the header** (not a physical line number). Assign IDs before filtering or shuffling; do not substitute a processed dataset.

The two frozen Parquet files are downloaded through Git LFS into `eda/output/`. They contain row metadata and assignments, **not the source texts**:

| File | How to use it |
| --- | --- |
| `role_manifest_v2.parquet` | Join original rows on `(source, row_id)` and use **`role_v2`**, not the older `role` or `split` columns. For a fixed holdout experiment, fit the extractor and tokenizer on `train` only and select models on `development`. |
| `cv_manifest_v1.parquet` | Reuse the frozen three-fold comparison sample. For each held-out `fold` (0, 1, or 2), train on the other two folds and evaluate on that fold. Fit the tokenizer only on those training folds. This samples eligible v2 `train`/`development` rows; it is not the full training corpus or the old v1 role split. |

**Never use any `exclude_*` row for training, tokenizer fitting, or quantitative evaluation.** Keep `final` reserved until the recipe is frozen and final evaluation is explicitly authorized. Check that joins lose no selected rows and introduce no duplicates, and verify zero accepted-group, Russian-document, and exact-text overlap between training and evaluation. Record source-file and manifest hashes, selected row IDs, role counts, and overlap checks with each experiment.

### What the config files do

The committed files in `eda/config/` record audit inputs and decisions; they are **not a replacement for the frozen manifests**:

| Config | Purpose |
| --- | --- |
| `split_rules.json` | Initial split seed and holdout rules; does not encode the complete v2 grouping/exclusion process. |
| `near_duplicate_decisions.json` | Reviewed candidate links; accepted links contribute to v2 groups and exclusions. |
| `review_decisions.json` | Annotation-review records, not automatic label corrections. |
| `language_review_decisions.json` | Language-review observations, not a new training split. |

For normal experiments, keep these configs unchanged and consume the downloaded manifests. Do not rerun splitting or create a random split if files are missing; resolve the LFS download first. Rebuilding manifests requires the full audited workflow and its intermediate inputs, not just these JSON files. See the [split/exclusion policy and role table](../../README.md#required-split-and-exclusion-policy-for-future-experiments) and [EDA workflow guide](../../eda/README.md).
