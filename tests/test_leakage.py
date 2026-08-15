from __future__ import annotations

import pandas as pd
import pytest

from credit_default.data import EXPLICIT_LEAKAGE_COLUMNS, prepare_supervised_data
from credit_default.features import FeatureBuilder


PATTERN_LEAKAGE_COLUMNS = [
    "recoveries",
    "collection_recovery_fee",
    "settlement_status",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
]
SUFFIX_ALIAS_COLUMNS = [
    "out_prncp_inv",
    "total_pymnt_inv",
    "total_rec_prncp_inv",
    "total_rec_int_inv",
]


def test_prepared_feature_matrix_has_no_banned_columns() -> None:
    frame = pd.DataFrame(
        {
            "": [0, 1],
            "id": ["synthetic-a", "synthetic-b"],
            "loan_status": ["Fully Paid", "Charged Off"],
            "loan_amnt": [1000, 1200],
            "total_pymnt": [1100, 0],
            "out_prncp": [0, 100],
        }
    )
    X, y = prepare_supervised_data(frame)
    assert len(y) == 2
    assert not set(EXPLICIT_LEAKAGE_COLUMNS).intersection(X.columns)
    assert "id" not in X.columns
    assert "__exported_index__" not in X.columns


def test_feature_builder_fails_closed_on_banned_columns() -> None:
    X = pd.DataFrame(
        {
            "loan_amnt": [1000.0, 1200.0],
            "total_pymnt": [1100.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="prohibited"):
        FeatureBuilder().fit(X)


def test_pattern_based_leakage_columns_never_reach_feature_matrix() -> None:
    frame = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off"],
            "loan_amnt": [1000.0, 1200.0],
            **{column: [1.0, 2.0] for column in PATTERN_LEAKAGE_COLUMNS},
        }
    )
    prepared, _ = prepare_supervised_data(frame)

    assert not set(PATTERN_LEAKAGE_COLUMNS).intersection(prepared.columns)
    with pytest.raises(ValueError, match="prohibited"):
        FeatureBuilder().fit(frame.drop(columns="loan_status"))


def test_index_and_post_origination_suffix_aliases_never_reach_feature_matrix() -> None:
    banned = ["index", *SUFFIX_ALIAS_COLUMNS]
    frame = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off"],
            "loan_amnt": [1000.0, 1200.0],
            **{column: [1.0, 2.0] for column in banned},
        }
    )
    prepared, _ = prepare_supervised_data(frame)

    assert not set(banned).intersection(prepared.columns)
    with pytest.raises(ValueError, match="prohibited"):
        FeatureBuilder().fit(frame.drop(columns="loan_status"))


def test_unknown_raw_columns_are_rejected_at_prepared_matrix_boundary() -> None:
    frame = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off"],
            "loan_amnt": [1000.0, 1200.0],
            "unexpected_source_field": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="Unknown raw feature columns"):
        prepare_supervised_data(frame)
