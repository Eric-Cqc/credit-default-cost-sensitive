# Credit Default Prediction with Cost-Sensitive Model Selection

This portfolio mini-project builds a reproducible, leakage-aware credit-default workflow for an imbalanced Lending Club course subset. It emphasizes data quality, honest model comparison, validation-only score-threshold selection, and an explicit business cost matrix. It is a compact demonstration—not a state-of-the-art credit-risk system.

## Problem and target

The binary target is derived from `loan_status`:

- `Fully Paid` → `0`
- `Charged Off`, `Default`, `In Grace Period`, `Late (16-30 days)`, and `Late (31-120 days)` → `1`
- `Current`, blank, and any unmapped status → excluded from the labeled sample

The exact counts, class balance, split sizes, encoded feature count, metrics, score thresholds, and selected model below are generated from `results/metrics.json` after a fresh run.

<!-- METRICS:START -->
### Verified run results

The fresh run labels 1,402 observations from 10,000 raw rows and produces 30 encoded features. Class balance is 951 non-default / 451 default (32.17% positive).

Split sizes: train 840, validation 281, test 281.

All expected costs below are normalized totals under the configured cost matrix; the selected score threshold minimizes validation cost. The test columns use that validation-selected score threshold.
Thresholds and model selection are validation-only; the test set is never used for selection.

| Model | Status | Val ROC-AUC | Val PR-AUC | Val F1 @ cost score threshold | Val expected cost | Test ROC-AUC | Test PR-AUC | Test F1 @ selected score threshold | Test expected cost | Cost score threshold | F1 score threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision tree | complete | 0.6973 | 0.4680 | 0.5824 | 165.00 | 0.5884 | 0.3986 | 0.4848 | 255.00 | 0.39 | 0.52 |
| Logistic regression | complete | 0.6777 | 0.4712 | 0.5484 | 160.00 | 0.6760 | 0.5225 | 0.5315 | 190.00 | 0.37 | 0.37 |
| Random forest | complete | 0.6691 | 0.4584 | 0.5497 | 164.00 | 0.6560 | 0.4543 | 0.5376 | 189.00 | 0.40 | 0.42 |
| XGBoost | complete | 0.6176 | 0.4237 | 0.4865 | 190.00 | 0.6317 | 0.4618 | 0.4905 | 187.00 | 0.01 | 0.14 |

**Selected model:** Logistic regression. **Rationale:** minimum validation expected cost (160.00 normalized units) under cost_fn=5.0 and cost_fp=1.0.
<!-- METRICS:END -->

## Methodology

Feature engineering follows the intent of the original coursework while moving every learned statistic into the training-only sklearn pipeline:

- parse `emp_length` into years and add `emp_unknown`;
- add `has_delinq` and train-fitted missing-month handling for `mths_since_last_delinq`;
- convert `earliest_cr_line` into credit age using 2011 as the anchor year;
- convert `int_rate` percent text to a numeric proportion;
- winsorize at the 1st/99th percentiles, then apply `log1p` to `annual_inc` and `revol_bal`;
- winsorize other skewed numeric variables;
- consolidate `home_ownership`, `purpose`, and `addr_state` into region groups;
- one-hot encode categorical features and add `pay_burden_income = installment / (annual income / 12)`;
- impute numeric/categorical missing values and scale numeric variables.

The pipeline is fitted independently for each model. Winsor bounds, date modes, delinquency sentinels, imputers, one-hot categories, and the scaler are all learned from the training partition only.

The split is stratified and deterministic: an 80/20 development/test split followed by a 75/25 train/validation split inside development, yielding 60/20/20. The dataset has no `issue_d`, so a temporal split is infeasible; this is a documented limitation.

The compared models use fixed, documented hyperparameters from `configs/config.yaml`:

| Model | Configuration |
|---|---|
| Logistic regression | `C=1.0`, `solver=liblinear`, `class_weight=balanced`, `max_iter=1000` |
| Decision tree | `max_depth=6`, `min_samples_leaf=50`, `class_weight=balanced` |
| Random forest | `n_estimators=300`, `max_depth=10`, `min_samples_leaf=20`, `class_weight=balanced` |
| XGBoost | `n_estimators=500`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, training-derived `scale_pos_weight` |

Every model uses `random_state=42`. XGBoost is a base dependency for the full comparison. On macOS, install the OpenMP runtime with `brew install libomp` if its native library is unavailable; the pipeline still retains a clear import/fit skip fallback for other environments.

The canonical configuration sets `n_jobs=1` for strict deterministic reruns; it can be raised for speed only when the resulting artifacts are revalidated.

Model scores are uncalibrated; the reported thresholds are score cutoffs, not default probabilities.

## Leakage prevention

The exported index and `id` are excluded. A strict raw-feature allowlist is enforced at the prepared-matrix boundary: only the audited underwriting fields enter `X`; unknown columns are rejected. The boundary also rejects post-origination fields, including `out_prncp`, `total_pymnt`, `total_rec_prncp`, `total_rec_int`, recoveries, collections, last/next payment fields, credit-pull fields, late-fee fields, settlement fields, and their `_inv` aliases. `loan_status` is used only for labeling. The feature builder fails closed if a prohibited column reaches the model pipeline.

No full-sample preprocessing is performed: the feature builder and `ColumnTransformer` sit inside every estimator pipeline and are fitted only on `X_train`.

## Cost-sensitive selection and business interpretation

The configurable default cost matrix is `cost_fn=5` and `cost_fp=1` in normalized units. It is an illustrative assumption that a missed default costs approximately five times as much as wrongly rejecting a good loan; it is not a calibrated lender loss model. Each model gets two score thresholds on validation: the minimum expected-cost score threshold used for selection and the F1-optimal score threshold reported for comparison. The selected model is the one with the lowest validation expected cost. The independent test set is evaluated once at score thresholds chosen on validation and is never used for selection.

The generated cost curve shows how the selected model's validation cost changes with its score threshold. In a real lending setting, the matrix should be replaced with approved economic estimates, and score calibration, reject inference, fairness review, and policy constraints would need separate treatment.

## Limitations

- The positive class combines intermediate delinquency states with charge-off/default, so it is not a pure eventual charge-off label.
- The labeled subset is small relative to production credit datasets.
- A random split is used because `issue_d` is unavailable; temporal generalization is not measured.
- The data is a course-provided Lending Club subset rather than a production portfolio.
- The cost matrix is illustrative and should not be interpreted as a validated financial loss function.
- No claim is made about calibration, causal effects, fairness, or deployment readiness.

## Differences from the original coursework

This repository preserves the useful feature-engineering intent but fixes several audit issues:

1. Learned preprocessing is fitted inside each training-only pipeline instead of being computed before splitting.
2. Post-origination payment/recovery fields, identifiers, and the exported index are explicitly excluded.
3. Unknown statuses are excluded through an auditable mapping, and engineered features are produced without full-sample statistics.
4. Score-threshold and model selection use validation expected cost under a configurable business matrix; F1 is retained only as a comparison score threshold, and the test set is held out for final reporting.

The historical logistic-regression test ROC-AUC of `0.6759` is sourced from the original non-public coursework notebook's embedded outputs and is not reproducible from this repository; it is not directly comparable because leakage-safe preprocessing and cost-based selection change the modeling protocol and operating score threshold.

## Reproduction

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If the system `python3` is too new for an available wheel, use a local Python 3.11 interpreter to create `.venv`. On macOS, install the OpenMP runtime before the full run when needed:

```bash
brew install libomp
```

The pipeline retains a fallback that records an XGBoost import/fit skip in `metrics.json` if the native runtime is unavailable elsewhere.

Obtain the authorized raw CSV and place it at `data/raw/loan.csv`, then run:

```bash
python -m credit_default.pipeline --config configs/config.yaml
python scripts/update_readme_metrics.py
pytest
```

For a data-free smoke/demo run (isolated under `results/demo/`):

```bash
python scripts/make_synthetic_sample.py --output data/sample/loan_sample_synthetic.csv
python -m credit_default.pipeline --config configs/config.yaml --data data/sample/loan_sample_synthetic.csv --metrics results/demo/metrics.json --figures results/demo/figures
```

The real-data run writes `results/metrics.json` and four clean matplotlib figures under `results/figures/`; the demo writes under `results/demo/`. The raw CSV is gitignored and is never part of the repository history.

## Repository layout

```text
configs/config.yaml                    Reproducible split, cost, and model settings
data/README.md                         Schema, provenance, acquisition, and licensing note
data/sample/loan_sample_synthetic.csv Safe synthetic smoke-test data
notebooks/01_exploration.ipynb         Aggregate-only thin EDA
scripts/                               Synthetic-data and README metric helpers
src/credit_default/                    Data, features, models, evaluation, and pipeline code
tests/                                 Mapping, leakage, fit-scope, split, math, and smoke tests
results/                               Fresh-run metrics and figures
```

## License

Code is released under the MIT License. The raw data remains subject to its source permissions and is not redistributed here.
