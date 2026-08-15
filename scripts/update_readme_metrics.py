#!/usr/bin/env python3
"""Render the metrics block in README.md from results/metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


START = "<!-- METRICS:START -->"
END = "<!-- METRICS:END -->"
LABELS = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _integer(value: Any) -> str:
    return f"{int(value):,}"


def render_metrics(metrics: dict[str, Any]) -> str:
    balance = metrics["class_balance"]["overall"]
    splits = metrics["split_sizes"]
    rows = []
    for name, result in metrics["models"].items():
        label = LABELS.get(name, name)
        if result.get("status") != "ok":
            rows.append(f"| {label} | skipped | — | — | — | — | — | — | — | — |")
            continue
        validation = result["validation"]
        test = result["test"]
        selected = result["selection"]
        val_cost = validation["at_selected_cost_threshold"]
        test_cost = test["at_validation_cost_threshold"]
        rows.append(
            "| {label} | complete | {vroc} | {vpr} | {vf1} | {vcost} | "
            "{troc} | {tpr} | {tf1} | {tcost} | {threshold} | {f1_threshold} |".format(
                label=label,
                vroc=_number(validation["roc_auc"]),
                vpr=_number(validation["pr_auc"]),
                vf1=_number(val_cost["f1"]),
                vcost=_number(val_cost["expected_cost"]["total"], 2),
                troc=_number(test["roc_auc"]),
                tpr=_number(test["pr_auc"]),
                tf1=_number(test_cost["f1"]),
                tcost=_number(test_cost["expected_cost"]["total"], 2),
                threshold=_number(selected["selected_threshold"], 2),
                f1_threshold=_number(selected["f1_optimal_threshold"], 2),
            )
        )

    selected = metrics["selected_model"]
    selected_result = metrics["models"][selected]
    selected_cost = selected_result["validation"]["at_selected_cost_threshold"]["expected_cost"]
    return "\n".join(
        [
            "### Verified run results",
            "",
            f"The fresh run labels {_integer(metrics['n_samples'])} observations from "
            f"{_integer(metrics['n_raw_samples'])} raw rows and produces "
            f"{_integer(metrics['n_features'])} encoded features. Class balance is "
            f"{_integer(balance['counts'].get('0', 0))} non-default / "
            f"{_integer(balance['counts'].get('1', 0))} default ({_number(balance['positive_rate'] * 100, 2)}% positive).",
            "",
            f"Split sizes: train {_integer(splits['train'])}, validation {_integer(splits['validation'])}, "
            f"test {_integer(splits['test'])}.",
            "",
            "All expected costs below are normalized totals under the configured cost matrix; "
            "the selected score threshold minimizes validation cost. The test columns use "
            "that validation-selected score threshold.",
            "Thresholds and model selection are validation-only; the test set is never used for selection.",
            "",
            "| Model | Status | Val ROC-AUC | Val PR-AUC | Val F1 @ cost score threshold | Val expected cost | Test ROC-AUC | Test PR-AUC | Test F1 @ selected score threshold | Test expected cost | Cost score threshold | F1 score threshold |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"**Selected model:** {LABELS.get(selected, selected)}. **Rationale:** minimum validation expected cost "
            f"({selected_cost['total']:.2f} normalized units) under cost_fn={metrics['cost_matrix']['cost_fn']} "
            f"and cost_fp={metrics['cost_matrix']['cost_fp']}.",
        ]
    )


def update_readme(readme_path: Path, metrics_path: Path) -> None:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    text = readme_path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("README is missing metrics markers")
    before, remainder = text.split(START, 1)
    _, after = remainder.split(END, 1)
    replacement = f"{START}\n{render_metrics(metrics)}\n{END}"
    readme_path.write_text(before + replacement + after, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.json"))
    args = parser.parse_args()
    update_readme(args.readme, args.metrics)
    print(f"Updated {args.readme} from {args.metrics}")


if __name__ == "__main__":
    main()
