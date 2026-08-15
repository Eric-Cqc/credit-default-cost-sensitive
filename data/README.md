# Data

## Provenance and acquisition

The project uses a 10,000-row Lending Club subset supplied with the original coursework materials. The raw CSV is intentionally not redistributed in this repository. To reproduce the real-data run, obtain the authorized `loan.csv` source and place it at `data/raw/loan.csv`.

The repository also includes `data/sample/loan_sample_synthetic.csv`. It has the same schema, plausible ranges, and deterministic random values, but contains no real rows. It is safe for tests and a data-free demo.

## Schema and data dictionary

The source CSV has 24 columns. The empty-name field is an exported row index and is ignored. `loan_status` is used only to derive the target and is never passed to a model.

| Column | Type in source | Meaning / modeling role |
|---|---|---|
| `""` | integer-like | Exported row index; excluded. |
| `id` | identifier | Loan identifier; excluded. |
| `loan_amnt` | numeric | Requested loan amount. |
| `funded_amnt` | numeric | Funded loan amount. |
| `term` | categorical text | Repayment term; one-hot encoded. |
| `int_rate` | percent text | Interest rate; converted from percent text to a proportion. |
| `installment` | numeric | Scheduled monthly installment. |
| `emp_length` | categorical text | Employment length; parsed into years plus an unknown indicator. |
| `home_ownership` | categorical text | Home ownership status; rare values consolidated. |
| `annual_inc` | numeric | Self-reported annual income; winsorized and log-transformed. |
| `loan_status` | categorical text | Outcome status used only for binary target mapping. |
| `purpose` | categorical text | Loan purpose; major purposes retained and rare purposes consolidated. |
| `addr_state` | categorical text | Borrower state; converted to a four-region feature. |
| `dti` | numeric | Debt-to-income ratio. |
| `delinq_2yrs` | numeric | Delinquencies in the preceding two years. |
| `earliest_cr_line` | date-like text | Earliest reported credit line; converted to credit age using 2011 as the anchor year. |
| `mths_since_last_delinq` | numeric | Months since most recent delinquency; paired with `has_delinq` and train-fitted missing handling. |
| `open_acc` | numeric | Number of open accounts. |
| `revol_bal` | numeric | Revolving account balance; winsorized and log-transformed. |
| `total_acc` | numeric | Total credit accounts. |
| `out_prncp` | numeric | Post-origination outstanding principal; leakage column, excluded. |
| `total_pymnt` | numeric | Post-origination total payments; leakage column, excluded. |
| `total_rec_prncp` | numeric | Post-origination recovered principal; leakage column, excluded. |
| `total_rec_int` | numeric | Post-origination recovered interest; leakage column, excluded. |

The loader also excludes any present recoveries, collections, last-payment, next-payment, credit-pull, late-fee, or settlement fields. Unknown or blank statuses are excluded from the labeled sample.

## Licensing note

The source is treated as a course-provided dataset. Users must confirm that their own authorized copy may be used for their intended purpose. This repository distributes code and a synthetic sample, not the raw CSV.
