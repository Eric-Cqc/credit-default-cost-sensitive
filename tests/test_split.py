from __future__ import annotations

import pandas as pd

from credit_default.data import stratified_train_val_test_split


def test_split_ratios_stratification_and_determinism() -> None:
    X = pd.DataFrame({"value": range(100)})
    y = pd.Series(([0] * 70) + ([1] * 30), name="target")
    first = stratified_train_val_test_split(X, y, seed=42)
    second = stratified_train_val_test_split(X, y, seed=42)

    assert len(first.X_train) == 60
    assert len(first.X_val) == 20
    assert len(first.X_test) == 20
    assert int(first.y_train.sum()) == 18
    assert int(first.y_val.sum()) == 6
    assert int(first.y_test.sum()) == 6
    assert first.X_train.equals(second.X_train)
    assert first.X_val.equals(second.X_val)
    assert first.X_test.equals(second.X_test)
