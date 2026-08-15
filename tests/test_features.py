from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from credit_default.data import load_raw_data, prepare_supervised_data
from credit_default.features import FeatureBuilder
from credit_default.models import make_model_pipeline


def _frame(values: list[float]) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "loan_amnt": values,
            "funded_amnt": values,
            "term": ["36 months"] * n,
            "int_rate": ["12.0%"] * n,
            "installment": [100.0] * n,
            "emp_length": ["5 years"] * n,
            "home_ownership": ["RENT"] * n,
            "annual_inc": values,
            "purpose": ["credit_card"] * n,
            "addr_state": ["CA"] * n,
            "dti": [10.0] * n,
            "delinq_2yrs": [0.0] * n,
            "earliest_cr_line": ["2000-01-01 12:00"] * n,
            "mths_since_last_delinq": [np.nan] * n,
            "open_acc": [5.0] * n,
            "revol_bal": values,
            "total_acc": [10.0] * n,
        }
    )


def test_winsor_bounds_are_fit_on_training_subset_only() -> None:
    train = _frame([100.0, 200.0, 300.0, 400.0, 500.0])
    validation = _frame([100.0, 1000.0, 1000000.0])
    builder = FeatureBuilder().fit(train)
    expected_upper = float(train["annual_inc"].quantile(0.99))
    assert builder.winsor_bounds_["annual_inc"][1] == expected_upper
    before = builder.winsor_bounds_.copy()
    transformed = builder.transform(validation)
    assert builder.winsor_bounds_ == before
    assert transformed.shape == (3, 20)


def test_pipeline_fits_preprocessing_from_training_subset() -> None:
    train = _frame([100.0, 200.0, 300.0, 400.0, 500.0])
    y = pd.Series([0, 0, 1, 1, 1])
    pipeline = make_model_pipeline(
        LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    )
    pipeline.fit(train, y)
    builder = pipeline.named_steps["feature_builder"]
    assert builder.winsor_bounds_["annual_inc"][1] == float(
        train["annual_inc"].quantile(0.99)
    )


def test_credit_age_is_derived_before_training_winsorization() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = load_raw_data(root / "data/sample/loan_sample_synthetic.csv")
    X, _ = prepare_supervised_data(raw)

    builder = FeatureBuilder().fit(X)
    transformed = builder.transform(X)

    assert builder.winsor_bounds_["credit_age"] != (0.0, 0.0)
    assert transformed["credit_age"].nunique(dropna=True) > 5
