"""Fixed model specifications with a graceful XGBoost fallback."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .features import FeatureBuilder, make_preprocessor


def make_model_pipeline(estimator: Any) -> Pipeline:
    """Wrap an estimator in the shared leakage-safe feature/preprocessing pipeline."""

    return Pipeline(
        steps=[
            ("feature_builder", FeatureBuilder()),
            ("preprocess", make_preprocessor()),
            ("classifier", estimator),
        ]
    )


def build_model_pipelines(
    model_config: dict[str, dict[str, Any]],
    *,
    scale_pos_weight: float,
) -> tuple[dict[str, Pipeline], dict[str, str]]:
    """Build configured models and return optional-model skip reasons separately."""

    pipelines: dict[str, Pipeline] = {}
    skipped: dict[str, str] = {}

    logistic_cfg = model_config.get("logistic_regression", {})
    if logistic_cfg.get("enabled", True):
        pipelines["logistic_regression"] = make_model_pipeline(
            LogisticRegression(
                C=float(logistic_cfg.get("C", 1.0)),
                solver=str(logistic_cfg.get("solver", "liblinear")),
                class_weight=logistic_cfg.get("class_weight", "balanced"),
                max_iter=int(logistic_cfg.get("max_iter", 1000)),
                random_state=int(logistic_cfg.get("random_state", 42)),
            )
        )
    else:
        skipped["logistic_regression"] = "disabled in configuration"

    tree_cfg = model_config.get("decision_tree", {})
    if tree_cfg.get("enabled", True):
        pipelines["decision_tree"] = make_model_pipeline(
            DecisionTreeClassifier(
                criterion=str(tree_cfg.get("criterion", "gini")),
                max_depth=int(tree_cfg.get("max_depth", 6)),
                min_samples_leaf=int(tree_cfg.get("min_samples_leaf", 50)),
                class_weight=tree_cfg.get("class_weight", "balanced"),
                random_state=int(tree_cfg.get("random_state", 42)),
            )
        )
    else:
        skipped["decision_tree"] = "disabled in configuration"

    rf_cfg = model_config.get("random_forest", {})
    if rf_cfg.get("enabled", True):
        pipelines["random_forest"] = make_model_pipeline(
            RandomForestClassifier(
                n_estimators=int(rf_cfg.get("n_estimators", 300)),
                max_depth=int(rf_cfg.get("max_depth", 10)),
                min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 20)),
                class_weight=rf_cfg.get("class_weight", "balanced"),
                random_state=int(rf_cfg.get("random_state", 42)),
                n_jobs=int(rf_cfg.get("n_jobs", -1)),
            )
        )
    else:
        skipped["random_forest"] = "disabled in configuration"

    xgb_cfg = model_config.get("xgboost", {})
    if xgb_cfg.get("enabled", True):
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - depends on local native wheels
            skipped["xgboost"] = f"dependency unavailable ({type(exc).__name__})"
        else:
            pipelines["xgboost"] = make_model_pipeline(
                XGBClassifier(
                    n_estimators=int(xgb_cfg.get("n_estimators", 500)),
                    max_depth=int(xgb_cfg.get("max_depth", 4)),
                    learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
                    subsample=float(xgb_cfg.get("subsample", 0.8)),
                    colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
                    reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=int(xgb_cfg.get("random_state", 42)),
                    n_jobs=int(xgb_cfg.get("n_jobs", -1)),
                    scale_pos_weight=float(scale_pos_weight),
                )
            )
    else:
        skipped["xgboost"] = "disabled in configuration"

    return pipelines, skipped
