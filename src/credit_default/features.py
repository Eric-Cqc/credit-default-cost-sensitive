"""Training-fitted feature engineering and sklearn preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .data import find_leakage_columns


CREDIT_ANCHOR_YEAR = 2011
MAJOR_PURPOSES = frozenset(
    {"debt_consolidation", "credit_card", "home_improvement", "small_business"}
)
STATE_TO_REGION = {
    "CT": "NE",
    "ME": "NE",
    "MA": "NE",
    "NH": "NE",
    "RI": "NE",
    "VT": "NE",
    "NJ": "NE",
    "NY": "NE",
    "PA": "NE",
    "IL": "MW",
    "IN": "MW",
    "MI": "MW",
    "OH": "MW",
    "WI": "MW",
    "IA": "MW",
    "KS": "MW",
    "MN": "MW",
    "MO": "MW",
    "NE": "MW",
    "ND": "MW",
    "SD": "MW",
    "DE": "SO",
    "FL": "SO",
    "GA": "SO",
    "MD": "SO",
    "NC": "SO",
    "SC": "SO",
    "VA": "SO",
    "DC": "SO",
    "WV": "SO",
    "AL": "SO",
    "KY": "SO",
    "MS": "SO",
    "TN": "SO",
    "AR": "SO",
    "LA": "SO",
    "OK": "SO",
    "TX": "SO",
    "AZ": "WE",
    "CO": "WE",
    "ID": "WE",
    "MT": "WE",
    "NV": "WE",
    "NM": "WE",
    "UT": "WE",
    "WY": "WE",
    "AK": "WE",
    "CA": "WE",
    "HI": "WE",
    "OR": "WE",
    "WA": "WE",
}

NUMERIC_FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "mths_since_last_delinq",
    "open_acc",
    "revol_bal",
    "total_acc",
    "int_rate_num",
    "credit_age",
    "emp_length_years",
    "emp_unknown",
    "has_delinq",
    "pay_burden_income",
]
CATEGORICAL_FEATURES = ["term", "home_ownership", "purpose", "addr_region"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
WINSOR_COLUMNS = [
    "annual_inc",
    "revol_bal",
    "installment",
    "loan_amnt",
    "funded_amnt",
    "open_acc",
    "total_acc",
    "dti",
    "credit_age",
    "mths_since_last_delinq",
]
LOG_COLUMNS = {"annual_inc", "revol_bal"}


def _series_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).astype(float)


def _parse_emp_length(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.lower().str.strip()
    text = text.replace({"n/a": pd.NA, "none": pd.NA, "": pd.NA})
    digits = text.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(digits, errors="coerce")


def _parse_interest_rate(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce") / 100.0


def _clip(values: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    lower, upper = bounds
    return values.clip(lower=lower, upper=upper)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Build intent-preserving features with every learned statistic fit on train.

    The builder is the first step in each model pipeline. Its fitted attributes
    include winsorization bounds, the credit-year mode, and the delinquency
    missing-value sentinel. Imputers, category vocabularies, and the scaler are
    fitted by the subsequent ``ColumnTransformer`` on the same training subset.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureBuilder":
        self._validate_input(X)
        frame = X.copy()
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)

        mths = _numeric(_series_or_nan(frame, "mths_since_last_delinq"))
        self.delinq_missing_value_ = (
            float(mths.max(skipna=True) + 1.0) if mths.notna().any() else 122.0
        )

        dates = pd.to_datetime(
            _series_or_nan(frame, "earliest_cr_line"), errors="coerce"
        )
        years = dates.dt.year.astype(float)
        year_mode = years.dropna().mode()
        self.credit_year_fill_ = (
            float(year_mode.iloc[0]) if not year_mode.empty else float(CREDIT_ANCHOR_YEAR)
        )
        derived_credit_age = float(CREDIT_ANCHOR_YEAR) - years.fillna(
            self.credit_year_fill_
        )

        has_delinq = mths.notna()
        self.winsor_bounds_: dict[str, tuple[float, float]] = {}
        for column in WINSOR_COLUMNS:
            if column == "credit_age":
                values = derived_credit_age
            else:
                values = _numeric(_series_or_nan(frame, column))
            if column == "mths_since_last_delinq":
                values = values.loc[has_delinq]
            finite = values.dropna()
            if finite.empty:
                self.winsor_bounds_[column] = (0.0, 0.0)
            else:
                self.winsor_bounds_[column] = (
                    float(finite.quantile(0.01)),
                    float(finite.quantile(0.99)),
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            ["feature_names_in_", "winsor_bounds_", "credit_year_fill_"],
        )
        self._validate_input(X)
        frame = X.copy()
        index = frame.index

        raw_values = {
            column: _numeric(_series_or_nan(frame, column))
            for column in WINSOR_COLUMNS
        }
        annual_clipped = _clip(raw_values["annual_inc"], self.winsor_bounds_["annual_inc"])
        revol_clipped = _clip(raw_values["revol_bal"], self.winsor_bounds_["revol_bal"])
        numeric: dict[str, pd.Series] = {}
        numeric["annual_inc"] = np.log1p(annual_clipped.clip(lower=0.0))
        numeric["revol_bal"] = np.log1p(revol_clipped.clip(lower=0.0))

        for column in WINSOR_COLUMNS:
            if column in LOG_COLUMNS:
                continue
            values = raw_values[column]
            if column == "mths_since_last_delinq":
                has_delinq = values.notna()
                values = values.fillna(self.delinq_missing_value_)
                values = values.copy()
                values.loc[has_delinq] = _clip(
                    values.loc[has_delinq], self.winsor_bounds_[column]
                )
            else:
                values = _clip(values, self.winsor_bounds_[column])
            numeric[column] = values

        emp_years = _parse_emp_length(_series_or_nan(frame, "emp_length"))
        numeric["emp_length_years"] = emp_years
        numeric["emp_unknown"] = emp_years.isna().astype(float)

        dates = pd.to_datetime(
            _series_or_nan(frame, "earliest_cr_line"), errors="coerce"
        )
        years = dates.dt.year.astype(float).fillna(self.credit_year_fill_)
        numeric["credit_age"] = _clip(
            float(CREDIT_ANCHOR_YEAR) - years,
            self.winsor_bounds_["credit_age"],
        )
        numeric["int_rate_num"] = _parse_interest_rate(
            _series_or_nan(frame, "int_rate")
        )
        numeric["delinq_2yrs"] = _numeric(_series_or_nan(frame, "delinq_2yrs"))
        numeric["has_delinq"] = _numeric(
            _numeric(_series_or_nan(frame, "mths_since_last_delinq")).notna().astype(float)
        )

        monthly_income = annual_clipped / 12.0
        installment = numeric["installment"]
        burden = installment / monthly_income
        burden = burden.replace([np.inf, -np.inf], np.nan)
        numeric["pay_burden_income"] = burden

        output = pd.DataFrame(index=index)
        for column in NUMERIC_FEATURES:
            output[column] = numeric.get(
                column, pd.Series(np.nan, index=index, dtype="float64")
            )

        term = _series_or_nan(frame, "term").astype("string").str.strip()
        term = term.mask(term.eq(""), pd.NA)
        home = _series_or_nan(frame, "home_ownership").astype("string").str.strip()
        home = home.replace({"NONE": "OTHER", "ANY": "OTHER"})
        purpose = _series_or_nan(frame, "purpose").astype("string").str.strip()
        purpose = purpose.where(
            purpose.isna() | purpose.isin(MAJOR_PURPOSES), "purpose_other"
        )
        state = _series_or_nan(frame, "addr_state").astype("string").str.upper().str.strip()
        region = state.map(STATE_TO_REGION).fillna("OTHER")
        output["term"] = term
        output["home_ownership"] = home
        output["purpose"] = purpose
        output["addr_region"] = region
        return output[ALL_FEATURES]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(ALL_FEATURES, dtype=object)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureBuilder expects a pandas DataFrame")
        leakage = find_leakage_columns(X.columns)
        if leakage:
            raise ValueError(
                "FeatureBuilder received prohibited columns: " + ", ".join(leakage)
            )


def make_preprocessor() -> ColumnTransformer:
    """Create the training-fitted imputation, encoding, and scaling transformer."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=float,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
