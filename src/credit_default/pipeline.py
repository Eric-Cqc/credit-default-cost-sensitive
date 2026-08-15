"""Single-command experiment pipeline for the credit-default portfolio."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from .data import (
    DataSplits,
    load_raw_data,
    prepare_supervised_data,
    stratified_train_val_test_split,
)
from .evaluate import make_threshold_grid, score_model
from .models import build_model_pipelines


MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
PALETTE = {
    "logistic_regression": "#0072B2",
    "decision_tree": "#D55E00",
    "random_forest": "#009E73",
    "xgboost": "#CC79A7",
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a top-level mapping")
    return config


def _distribution(y: pd.Series) -> dict[str, Any]:
    counts = y.value_counts().sort_index()
    total = int(len(y))
    return {
        "n": total,
        "counts": {str(int(key)): int(value) for key, value in counts.items()},
        "positive_rate": float((y == 1).mean()) if total else 0.0,
    }


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    distributions = ("numpy", "pandas", "scikit-learn", "matplotlib", "PyYAML", "xgboost")
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def _config_sha256(config_path: Path | None, config: dict[str, Any]) -> str:
    if config_path is not None and config_path.exists():
        return _sha256_file(config_path)
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _thresholds_from_config(config: dict[str, Any]) -> np.ndarray:
    threshold_cfg = config.get("threshold_grid", {})
    return make_threshold_grid(
        minimum=float(threshold_cfg.get("minimum", 0.01)),
        maximum=float(threshold_cfg.get("maximum", 0.99)),
        step=float(threshold_cfg.get("step", 0.01)),
    )


def _feature_importance_rows(pipeline: Any, top_n: int = 20) -> dict[str, Any]:
    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
        importance_type = "feature_importance"
    elif hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_, dtype=float)[0])
        importance_type = "absolute_coefficient"
    else:
        return {"importance_type": "unavailable", "features": []}

    order = np.argsort(values)[::-1][:top_n]
    return {
        "importance_type": importance_type,
        "features": [
            {"feature": str(feature_names[index]), "importance": float(values[index])}
            for index in order
        ],
    }


def _plot_roc_curves(
    y_val: pd.Series,
    y_test: pd.Series,
    probabilities: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for axis, split_name, y_true in (
        (axes[0], "Validation", y_val),
        (axes[1], "Test", y_test),
    ):
        for model_name, scores in probabilities.items():
            values = scores[split_name.lower()]
            fpr, tpr, _ = roc_curve(y_true, values)
            axis.plot(
                fpr,
                tpr,
                lw=2,
                color=PALETTE.get(model_name, "#333333"),
                label=f"{MODEL_LABELS.get(model_name, model_name)} (AUC={auc(fpr, tpr):.3f})",
            )
        axis.plot([0, 1], [0, 1], color="#999999", lw=1, linestyle="--")
        axis.set_title(split_name)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("ROC curves", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_curves(
    y_val: pd.Series,
    y_test: pd.Series,
    probabilities: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for axis, split_name, y_true in (
        (axes[0], "Validation", y_val),
        (axes[1], "Test", y_test),
    ):
        for model_name, scores in probabilities.items():
            values = scores[split_name.lower()]
            precision, recall, _ = precision_recall_curve(y_true, values)
            axis.plot(
                recall,
                precision,
                lw=2,
                color=PALETTE.get(model_name, "#333333"),
                label=MODEL_LABELS.get(model_name, model_name),
            )
        prevalence = float(np.mean(np.asarray(y_true) == 1))
        axis.axhline(prevalence, color="#999999", lw=1, linestyle="--", label="Positive rate")
        axis.set_title(split_name)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Precision–recall curves", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cost_curve(
    model_name: str,
    model_result: dict[str, Any],
    output_path: Path,
) -> None:
    curve = model_result["selection"]["validation_threshold_curve"]
    thresholds = [row["threshold"] for row in curve]
    costs = [row["expected_cost_per_sample"] for row in curve]
    selected = model_result["selection"]["selected_threshold"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(thresholds, costs, color=PALETTE.get(model_name, "#333333"), lw=2)
    axis.axvline(selected, color="#333333", lw=1.2, linestyle="--", label=f"Selected = {selected:.2f}")
    axis.set_title(
        f"Validation cost by score threshold — {MODEL_LABELS.get(model_name, model_name)}"
    )
    axis.set_xlabel("Default score threshold")
    axis.set_ylabel("Expected cost per validation observation")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importances(
    importance: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    available = [(name, value) for name, value in importance.items() if value["features"]]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 6), squeeze=False)
    for axis, (model_name, value) in zip(axes[0], available):
        rows = list(reversed(value["features"][:20]))
        axis.barh(
            [row["feature"] for row in rows],
            [row["importance"] for row in rows],
            color=PALETTE.get(model_name, "#333333"),
        )
        axis.set_title(MODEL_LABELS.get(model_name, model_name))
        axis.set_xlabel(value["importance_type"].replace("_", " ").title())
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Top model feature importances", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run_pipeline(
    config_path: str | Path | None = None,
    *,
    config: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
    data_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete experiment and write metrics/figures to configured paths."""

    if config is None:
        if config_path is None:
            raise ValueError("Provide config_path or config")
        config = load_config(config_path)
    else:
        config = json.loads(json.dumps(config))

    if repo_root is None:
        if config_path is not None:
            repo_root = Path(config_path).resolve().parent.parent
        else:
            repo_root = Path.cwd()
    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve() if config_path is not None else None
    paths = config.get("paths", {})
    raw_path = (
        _resolve_path(root, data_path)
        if data_path is not None
        else _resolve_path(root, paths["raw_data"])
    )
    output_metrics = (
        _resolve_path(root, metrics_path)
        if metrics_path is not None
        else _resolve_path(root, paths.get("metrics", "results/metrics.json"))
    )
    output_figures = (
        _resolve_path(root, figures_dir)
        if figures_dir is not None
        else _resolve_path(root, paths.get("figures", "results/figures"))
    )

    seed = int(config.get("seed", 42))
    split_config = config.get("split", {})
    cost_config = config.get("cost_matrix", {})
    cost_fn = float(cost_config.get("cost_fn", 5.0))
    cost_fp = float(cost_config.get("cost_fp", 1.0))

    raw = load_raw_data(raw_path)
    X, y = prepare_supervised_data(raw)
    splits = stratified_train_val_test_split(
        X,
        y,
        seed=seed,
        test_size=float(split_config.get("test_size", 0.20)),
        validation_size_within_dev=float(
            split_config.get("validation_size_within_dev", 0.25)
        ),
    )
    positive = int((splits.y_train == 1).sum())
    negative = int((splits.y_train == 0).sum())
    if positive == 0 or negative == 0:
        raise ValueError("Training split must contain both classes")
    scale_pos_weight = float(negative / positive)

    model_pipelines, skipped = build_model_pipelines(
        config.get("models", {}),
        scale_pos_weight=scale_pos_weight,
    )
    thresholds = _thresholds_from_config(config)
    model_results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Any] = {}
    probabilities: dict[str, dict[str, np.ndarray]] = {}

    for model_name, model_pipeline in model_pipelines.items():
        print(f"Fitting {MODEL_LABELS.get(model_name, model_name)}...")
        try:
            model_pipeline.fit(splits.X_train, splits.y_train)
            val_scores = model_pipeline.predict_proba(splits.X_val)[:, 1]
            test_scores = model_pipeline.predict_proba(splits.X_test)[:, 1]
        except Exception as exc:
            if model_name == "xgboost":  # pragma: no cover - native dependency dependent
                skipped[model_name] = f"fit unavailable ({type(exc).__name__})"
                warnings.warn(
                    "XGBoost was skipped because its estimator could not fit "
                    f"({type(exc).__name__}).",
                    stacklevel=2,
                )
                continue
            raise
        fitted[model_name] = model_pipeline
        probabilities[model_name] = {"validation": val_scores, "test": test_scores}
        model_results[model_name] = score_model(
            splits.y_val,
            val_scores,
            splits.y_test,
            test_scores,
            thresholds=thresholds,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )

    for model_name, reason in skipped.items():
        model_results[model_name] = {"status": "skipped", "reason": reason}

    eligible = [
        (name, result)
        for name, result in model_results.items()
        if result.get("status") == "ok"
    ]
    if not eligible:
        raise RuntimeError("No model completed successfully")
    selected_model, selected_result = min(
        eligible,
        key=lambda item: (
            item[1]["validation"]["at_selected_cost_threshold"]["expected_cost"]["total"],
            list(model_results).index(item[0]),
        ),
    )

    first_pipeline = next(iter(fitted.values()))
    feature_names = [
        str(name)
        for name in first_pipeline.named_steps["preprocess"].get_feature_names_out()
    ]
    importances = {
        model_name: _feature_importance_rows(model_pipeline)
        for model_name, model_pipeline in fitted.items()
    }

    output_figures.mkdir(parents=True, exist_ok=True)
    _plot_roc_curves(
        splits.y_val,
        splits.y_test,
        probabilities,
        output_figures / "roc_curves.png",
    )
    _plot_pr_curves(
        splits.y_val,
        splits.y_test,
        probabilities,
        output_figures / "pr_curves.png",
    )
    _plot_cost_curve(
        selected_model,
        selected_result,
        output_figures / "selected_model_cost_curve.png",
    )
    _plot_feature_importances(
        {
            name: importances[name]
            for name in ("random_forest", selected_model)
            if name in importances
        },
        output_figures / "feature_importances.png",
    )

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "data_source": "configured raw CSV",
        "reproducibility": {
            "raw_csv_sha256": _sha256_file(raw_path),
            "config_sha256": _config_sha256(config_file, config),
            "python_version": platform.python_version(),
            "package_versions": _package_versions(),
        },
        "n_raw_samples": int(len(raw)),
        "n_samples": int(len(X)),
        "raw_feature_count": int(X.shape[1]),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "split_sizes": {
            "train": int(len(splits.X_train)),
            "validation": int(len(splits.X_val)),
            "test": int(len(splits.X_test)),
        },
        "split_proportions": {
            "train": float(len(splits.X_train) / len(X)),
            "validation": float(len(splits.X_val) / len(X)),
            "test": float(len(splits.X_test) / len(X)),
        },
        "class_balance": {
            "overall": _distribution(y),
            "train": _distribution(splits.y_train),
            "validation": _distribution(splits.y_val),
            "test": _distribution(splits.y_test),
        },
        "cost_matrix": {
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
            "units": "normalized expected-cost units",
            "assumption": "A missed default costs approximately five times a wrongly rejected good loan; illustrative and configurable.",
        },
        "threshold_grid": [float(value) for value in thresholds],
        "scale_pos_weight": scale_pos_weight,
        "models": model_results,
        "selected_model": selected_model,
        "selection": {
            "criterion": "minimum validation expected cost",
            "selected_model": selected_model,
            "rationale": "Choose the available model with the lowest validation total cost under the configured cost matrix; test scores are not used for selection.",
        },
        "feature_importances": importances,
        "figures": [],
    }
    figure_names = (
        "roc_curves.png",
        "pr_curves.png",
        "selected_model_cost_curve.png",
        "feature_importances.png",
    )
    for figure_name in figure_names:
        figure_path = (output_figures / figure_name).resolve()
        try:
            metrics["figures"].append(str(figure_path.relative_to(root)))
        except ValueError:
            metrics["figures"].append(figure_name)
    _write_json(output_metrics, metrics)

    print(
        f"Completed: n={metrics['n_samples']}, features={metrics['n_features']}, "
        f"selected={MODEL_LABELS.get(selected_model, selected_model)}, "
        f"score_threshold={selected_result['selection']['selected_threshold']:.2f}"
    )
    if skipped:
        for model_name, reason in skipped.items():
            print(f"Skipped {MODEL_LABELS.get(model_name, model_name)}: {reason}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/config.yaml",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional CSV override; defaults to the configured raw data path",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Optional metrics JSON override; defaults to the configured output",
    )
    parser.add_argument(
        "--figures",
        default=None,
        help="Optional figures directory override; defaults to the configured output",
    )
    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        data_path=args.data,
        metrics_path=args.metrics,
        figures_dir=args.figures,
    )


if __name__ == "__main__":
    main()
