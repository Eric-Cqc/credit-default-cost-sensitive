"""Data loading, status labeling, leakage guards, and deterministic splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


STATUS_MAP: dict[str, int] = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "In Grace Period": 1,
    "Late (16-30 days)": 1,
    "Late (31-120 days)": 1,
}

EXPLICIT_LEAKAGE_COLUMNS = frozenset(
    {
        "out_prncp",
        "total_pymnt",
        "total_rec_prncp",
        "total_rec_int",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "last_credit_pull_d",
        "next_pymnt_d",
        "total_rec_late_fee",
        "settlement_status",
    }
)

# These are the only raw columns allowed to cross the supervised-data boundary.
# The label, identifiers, exported index, and all post-origination fields are
# intentionally absent.  Keeping this list explicit prevents a new source
# column from becoming a model feature by accident.
RAW_FEATURE_COLUMNS = (
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "mths_since_last_delinq",
    "open_acc",
    "revol_bal",
    "total_acc",
)
RAW_FEATURE_ALLOWLIST = frozenset(RAW_FEATURE_COLUMNS)

NON_FEATURE_COLUMNS = frozenset(
    {
        "id",
        "member_id",
        "loan_status",
        "target",
        "index",
        "__exported_index__",
    }
)

LEAKAGE_NAME_PARTS = (
    "recover",
    "collection",
    "last_pymnt",
    "last_credit_pull",
    "next_pymnt",
    "settlement",
    "total_rec_late_fee",
)


@dataclass(frozen=True)
class DataSplits:
    """The three stratified partitions used by the experiment."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and whitespace without learning sample statistics."""

    df = frame.copy()
    rename: dict[object, str] = {}
    for column in df.columns:
        text = str(column)
        if text == "" or text.startswith("Unnamed:"):
            rename[column] = "__exported_index__"
    if rename:
        df = df.rename(columns=rename)

    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].astype("string")
        df[column] = values.str.strip()
    return df


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read a CSV and apply only deterministic string/column normalization."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")
    return clean_data(pd.read_csv(csv_path, low_memory=False))


def label_status(status: pd.Series) -> pd.Series:
    """Map audited loan statuses to binary labels; unknown statuses become ``<NA>``."""

    normalized = status.astype("string").str.strip()
    return normalized.map(STATUS_MAP).astype("Int64")


def find_leakage_columns(columns: Iterable[object]) -> list[str]:
    """Return columns that are identifiers, labels, or post-origination leakage."""

    non_feature_names = {item.lower() for item in NON_FEATURE_COLUMNS}
    explicit_names = {item.lower() for item in EXPLICIT_LEAKAGE_COLUMNS}
    found: list[str] = []
    for column in columns:
        name = str(column)
        normalized = name.strip().lower()
        is_non_feature = normalized in non_feature_names
        is_explicit = normalized in explicit_names
        is_pattern = any(part in normalized for part in LEAKAGE_NAME_PARTS)
        alias_base = normalized[:-4] if normalized.endswith("_inv") else ""
        is_leakage_alias = bool(alias_base) and (
            alias_base in explicit_names
            or any(part in alias_base for part in LEAKAGE_NAME_PARTS)
        )
        if is_non_feature or is_explicit or is_pattern or is_leakage_alias:
            found.append(name)
    return found


def prepare_supervised_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create a labeled feature frame while excluding all known non-feature columns.

    Rows whose status is ``Current``, blank, or otherwise unmapped are excluded.
    The returned feature frame is intentionally free of ``loan_status``, identifiers,
    exported indices, and post-origination fields before any model pipeline sees it.
    """

    df = clean_data(frame)
    if "loan_status" not in df.columns:
        raise ValueError("Input data must contain loan_status for supervised labeling")

    labels = label_status(df["loan_status"])
    keep = labels.notna()
    labeled = df.loc[keep].copy()
    y = labels.loc[keep].astype("int64").rename("target")

    drop_columns = find_leakage_columns(labeled.columns)
    dropped_names = {name.strip().lower() for name in drop_columns}
    unknown_columns = [
        str(column)
        for column in labeled.columns
        if str(column).strip().lower() not in RAW_FEATURE_ALLOWLIST
        and str(column).strip().lower() not in dropped_names
    ]
    if unknown_columns:
        raise ValueError(
            "Unknown raw feature columns are not allowed: "
            + ", ".join(unknown_columns)
        )

    # Preserve the audited feature order while allowing a partial frame in
    # unit tests; FeatureBuilder handles missing allowlisted inputs as NaN.
    X = labeled.loc[
        :, [column for column in RAW_FEATURE_COLUMNS if column in labeled.columns]
    ].copy()
    remaining = find_leakage_columns(X.columns)
    if remaining:
        raise RuntimeError(f"Leakage guard failed to remove columns: {remaining}")

    return X.reset_index(drop=True), y.reset_index(drop=True)


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = 42,
    test_size: float = 0.20,
    validation_size_within_dev: float = 0.25,
) -> DataSplits:
    """Return deterministic stratified 60/20/20-style train/val/test partitions."""

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=validation_size_within_dev,
        random_state=seed,
        stratify=y_dev,
    )

    return DataSplits(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
