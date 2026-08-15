"""Metrics, score-threshold selection, and explicit expected-cost calculations."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def make_threshold_grid(
    minimum: float = 0.01,
    maximum: float = 0.99,
    step: float = 0.01,
) -> np.ndarray:
    """Create a rounded, inclusive score-threshold grid for deterministic selection."""

    if not 0.0 <= minimum < maximum <= 1.0:
        raise ValueError("Threshold bounds must satisfy 0 <= minimum < maximum <= 1")
    if step <= 0:
        raise ValueError("Threshold step must be positive")
    values = np.arange(minimum, maximum + step / 2.0, step, dtype=float)
    return np.round(values, 10)


def expected_cost(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> float:
    """Return total normalized cost: ``cost_fn * FN + cost_fp * FP``."""

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    false_negatives = int(matrix[1, 0])
    false_positives = int(matrix[0, 1])
    return float(cost_fn * false_negatives + cost_fp * false_positives)


def cost_summary(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    cost_fn: float,
    cost_fp: float,
) -> dict[str, float | int | list[list[int]]]:
    """Return confusion counts and total/per-observation expected cost."""

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    total = float(cost_fn * fn + cost_fp * fp)
    n = int(matrix.sum())
    return {
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "total": total,
        "per_sample": float(total / n) if n else 0.0,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def threshold_metrics(
    y_true: Iterable[int],
    scores: Iterable[float],
    threshold: float,
    *,
    cost_fn: float,
    cost_fp: float,
) -> dict[str, float | int | list[list[int]] | dict[str, float | int | list[list[int]]]]:
    """Calculate classification and cost metrics at one score threshold."""

    truth = np.asarray(list(y_true), dtype=int)
    score_values = np.asarray(list(scores), dtype=float)
    predictions = (score_values >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "expected_cost": cost_summary(
            truth, predictions, cost_fn=cost_fn, cost_fp=cost_fp
        ),
    }


def select_thresholds(
    y_true: Iterable[int],
    scores: Iterable[float],
    thresholds: Iterable[float],
    *,
    cost_fn: float,
    cost_fp: float,
) -> dict[str, object]:
    """Select F1- and minimum-cost score thresholds using validation scores only."""

    truth = np.asarray(list(y_true), dtype=int)
    score_values = np.asarray(list(scores), dtype=float)
    grid = np.asarray(list(thresholds), dtype=float)
    if grid.size == 0:
        raise ValueError("At least one threshold is required")

    curve: list[dict[str, float | int]] = []
    f1_values: list[float] = []
    cost_values: list[float] = []
    for threshold in grid:
        metrics = threshold_metrics(
            truth,
            score_values,
            float(threshold),
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )
        cost = metrics["expected_cost"]
        if not isinstance(cost, dict):
            raise TypeError("Internal expected-cost result has an invalid shape")
        f1_value = float(metrics["f1"])
        total_cost = float(cost["total"])
        f1_values.append(f1_value)
        cost_values.append(total_cost)
        curve.append(
            {
                "threshold": float(threshold),
                "f1": f1_value,
                "expected_cost_total": total_cost,
                "expected_cost_per_sample": float(cost["per_sample"]),
            }
        )

    f1_index = int(np.argmax(np.asarray(f1_values)))
    cost_index = int(np.argmin(np.asarray(cost_values)))
    return {
        "f1_optimal_threshold": float(grid[f1_index]),
        "cost_optimal_threshold": float(grid[cost_index]),
        "validation_threshold_curve": curve,
    }


def score_model(
    y_val: Iterable[int],
    val_scores: Iterable[float],
    y_test: Iterable[int],
    test_scores: Iterable[float],
    *,
    thresholds: Iterable[float],
    cost_fn: float,
    cost_fp: float,
) -> dict[str, object]:
    """Evaluate one fitted model without selecting anything from test scores."""

    val_truth = np.asarray(list(y_val), dtype=int)
    validation_scores = np.asarray(list(val_scores), dtype=float)
    test_truth = np.asarray(list(y_test), dtype=int)
    held_out_scores = np.asarray(list(test_scores), dtype=float)
    selection = select_thresholds(
        val_truth,
        validation_scores,
        thresholds,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
    )
    cost_threshold = float(selection["cost_optimal_threshold"])
    f1_threshold = float(selection["f1_optimal_threshold"])

    return {
        "status": "ok",
        "selection": {
            "criterion": "minimum_validation_expected_cost",
            "selected_threshold": cost_threshold,
            "f1_optimal_threshold": f1_threshold,
            "validation_threshold_curve": selection["validation_threshold_curve"],
        },
        "validation": {
            "roc_auc": float(roc_auc_score(val_truth, validation_scores)),
            "pr_auc": float(average_precision_score(val_truth, validation_scores)),
            "at_selected_cost_threshold": threshold_metrics(
                val_truth,
                validation_scores,
                cost_threshold,
                cost_fn=cost_fn,
                cost_fp=cost_fp,
            ),
            "at_f1_optimal_threshold": threshold_metrics(
                val_truth,
                validation_scores,
                f1_threshold,
                cost_fn=cost_fn,
                cost_fp=cost_fp,
            ),
        },
        "test": {
            "roc_auc": float(roc_auc_score(test_truth, held_out_scores)),
            "pr_auc": float(average_precision_score(test_truth, held_out_scores)),
            "at_validation_cost_threshold": threshold_metrics(
                test_truth,
                held_out_scores,
                cost_threshold,
                cost_fn=cost_fn,
                cost_fp=cost_fp,
            ),
            "at_validation_f1_threshold": threshold_metrics(
                test_truth,
                held_out_scores,
                f1_threshold,
                cost_fn=cost_fn,
                cost_fp=cost_fp,
            ),
        },
    }
