# Exploratory analysis

This directory contains the analysis behind the findings in the [project README](../README.md). The notebooks show the results; the Python modules make the data checks and experiments reproducible. Generated tables, model predictions, and checkpoints stay in the ignored `output/` directory. Submission files stay in the ignored `submissions/` directory.

## Layout

| Path | Purpose |
| --- | --- |
| `notebooks/` | Executed data, domain, and extraction analyses |
| `core/` | Shared parsing, scoring, split, and domain code |
| `workflows/` | Runnable audits, exports, and result summaries |
| `colab/` | Self-contained scripts uploaded for GPU experiments |
| `config/` | Split rules and recorded annotation-review decisions |
| `archive/` | The earlier role assignment and the agreed experiment plan |
| `output/` | Local artifacts, excluded from Git |
| `submissions/` | Generated Kaggle submissions, excluded from Git |

The current evaluation roles come from `workflows/roles_v2.py`. `archive/roles.py` is needed to reproduce its version 1 input; do not use version 1 as the current model split. The three notebooks retain their executed outputs. Their code reads local artifacts from `output/`, so those files must be regenerated to rerun every cell. The [EDA comparison protocol](model_plan.md) records the completed bounded GRU/LSTM experiments and their stopping point before stage 2.

## Split requirements for new experiments

Follow the [required split and exclusion policy](../README.md#required-split-and-exclusion-policy-for-future-experiments). The authoritative roles are `output/role_manifest_v2.parquet`, column `role_v2`; all `exclude_*` rows stay out of supervised training and tokenizer fitting, including future larger runs. This applies to both languages, not only the large Russian connected group. Do not recover extra training data by reading the raw CSVs without joining the role manifest.

For the fixed development holdout, use only `train` rows for fitting. The existing grouped CV manifest, `output/cv_manifest_v1.parquet`, was built from role version 2 and samples only `train` and `development` rows; its version number does not refer to role version 1. Fit each tokenizer on the current training folds only. Final evaluation remains reserved. Before exporting a new experiment, check excluded-row membership and group/document/exact-text separation, and save manifest hashes, row IDs, counts, and check results. Rebuild missing manifests through the audited workflow below rather than substituting a fresh random split.

## Run locally

From the project root, place the supplied files in `data/kse-ua-location-extraction-2026/`, install `requirements.txt`, and run a workflow as a module. For example:

```sh
python -m eda.workflows.cv_slices
python -m eda.workflows.cv_exposure
python -m eda.workflows.run_notebook eda/notebooks/extraction_pilot.ipynb
```

The notebook runner uses a local Unix socket. The first data audit scans roughly nine million source rows and takes several minutes. The extraction notebook reads saved results and does not retrain models.

To rebuild the analysis from raw data, run the full audit and original split first, followed by the role and grouping audits, version 2 role assignment, fold export, and extraction exports. The [experiment plan](archive/followup_plan.md) records the original sequence and decisions; the [project README](../README.md) gives the current counts and conclusions. Colab scripts expect their input files under `/content` when uploaded individually.

The saved CV error slices are in `output/eda_cv_slices.json`. The supervised-name exposure analysis is in `output/eda_cv_exposure.json`. The latter reconstructs exactly which rows each bounded model trained on; “unseen” means a string was absent from that fold's available training pool, while “available but unsampled” means it was present in that pool but outside the model's token budget.

## Completed EDA comparison

The completed comparison can be reproduced from the existing CV bundle. Generate the frozen row selection locally:

```sh
EDA_COMPARE_DATA=eda/output/extractor_bundle_v1.jsonl.gz \
EDA_COMPARE_OUT=eda/output/eda_comparison \
python -m eda.colab.eda_comparison
python -m eda.workflows.representation_comparison
```

The selection records the exact training row IDs for each fold. On Colab, upload the bundle, `colab_extractor.py`, `eda_comparison.py`, under `/content`, and place `selection.json` at `/content/eda_comparison/selection.json`. Set `EDA_COMPARE_MODE=run` and run `eda_comparison.py`; completed fits are recorded in `/content/eda_comparison/results.json`, so the script can resume after an interruption. After downloading the results and predictions to `eda/output/eda_comparison/`, summarize them with:

```sh
python -m eda.workflows.summarize_comparison
python -m eda.workflows.comparison_error_analysis
python -m eda.workflows.comparison_score_sensitivity
```

The domain confirmation uses the same word BiGRU encoder dimensions as the extractor. Run `python -m eda.workflows.domain_recurrent_control`, then `python -m eda.workflows.domain_cue_baselines` to compare its AUC with simple language and length cues. The output directory is ignored by Git. The extraction comparison uses only the frozen CV rows; it does not read the final local evaluation set.
