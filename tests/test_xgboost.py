from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from credit_default.models import build_model_pipelines
from credit_default.pipeline import load_config


try:
    import xgboost
except Exception as exc:  # pragma: no cover - native dependency dependent
    pytest.skip(
        f"xgboost unavailable: {type(exc).__name__}",
        allow_module_level=True,
    )


def test_xgboost_imports_trains_and_matches_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/config.yaml")
    pipelines, skipped = build_model_pipelines(
        config["models"],
        scale_pos_weight=2.5,
    )
    assert xgboost.__version__
    assert "xgboost" in pipelines
    assert "xgboost" not in skipped

    estimator = pipelines["xgboost"].named_steps["classifier"]
    assert estimator.get_params()["n_estimators"] == 500
    assert estimator.get_params()["max_depth"] == 4
    assert estimator.get_params()["learning_rate"] == 0.05
    assert estimator.get_params()["subsample"] == 0.8
    assert estimator.get_params()["colsample_bytree"] == 0.8
    assert estimator.get_params()["reg_lambda"] == 1.0
    assert estimator.get_params()["scale_pos_weight"] == 2.5
    assert estimator.get_params()["random_state"] == 42

    X = np.array([[0.0], [1.0], [0.2], [0.8], [0.1], [0.9]], dtype=float)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    toy_model = xgboost.XGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=1,
        scale_pos_weight=1.0,
    )
    toy_model.fit(X, y)
    probabilities = toy_model.predict_proba(X)[:, 1]
    assert probabilities.shape == (len(y),)
    assert np.isfinite(probabilities).all()
