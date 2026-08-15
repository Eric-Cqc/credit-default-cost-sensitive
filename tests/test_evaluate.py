from __future__ import annotations

import numpy as np

from credit_default.evaluate import expected_cost, select_thresholds


def test_expected_cost_math() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    assert expected_cost(y_true, y_pred, cost_fn=5, cost_fp=1) == 6.0


def test_cost_threshold_is_selected_on_toy_arrays() -> None:
    y_true = [0, 0, 1, 1]
    probabilities = [0.10, 0.40, 0.60, 0.90]
    result = select_thresholds(
        y_true,
        probabilities,
        np.array([0.30, 0.50, 0.70]),
        cost_fn=5,
        cost_fp=1,
    )
    assert result["cost_optimal_threshold"] == 0.5
    assert result["f1_optimal_threshold"] == 0.5
    assert len(result["validation_threshold_curve"]) == 3
