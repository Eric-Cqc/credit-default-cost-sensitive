from __future__ import annotations

import pandas as pd

from credit_default.data import label_status, prepare_supervised_data


def test_label_mapping_and_exclusions() -> None:
    statuses = pd.Series(
        [
            "Fully Paid",
            "Charged Off",
            "Default",
            "In Grace Period",
            "Late (16-30 days)",
            "Late (31-120 days)",
            "Current",
            "",
            None,
        ]
    )
    labels = label_status(statuses)
    assert labels.iloc[:6].tolist() == [0, 1, 1, 1, 1, 1]
    assert pd.isna(labels.iloc[6])
    assert pd.isna(labels.iloc[7])
    assert pd.isna(labels.iloc[8])


def test_prepare_supervised_data_drops_current_blank_and_unmapped_statuses() -> None:
    frame = pd.DataFrame(
        {
            "": [0, 1, 2, 3, 4],
            "id": ["synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d", "synthetic-e"],
            "loan_status": ["Fully Paid", "Current", "", "Not A Mapped Status", "Charged Off"],
            "loan_amnt": [1000, 1100, 1200, 1300, 1400],
        }
    )
    X, y = prepare_supervised_data(frame)

    assert len(X) == 2
    assert y.tolist() == [0, 1]
