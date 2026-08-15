from __future__ import annotations

import copy
import json
from pathlib import Path

from credit_default.pipeline import load_config, run_pipeline


try:
    import xgboost  # noqa: F401
except Exception:  # pragma: no cover - native dependency dependent
    XGBOOST_AVAILABLE = False
else:
    XGBOOST_AVAILABLE = True


def test_end_to_end_smoke_on_synthetic_sample(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/config.yaml")
    config = copy.deepcopy(config)
    config["models"]["random_forest"]["n_jobs"] = 1
    config["models"]["xgboost"]["n_jobs"] = 1
    metrics_path = tmp_path / "metrics.json"
    figures_dir = tmp_path / "figures"

    metrics = run_pipeline(
        config=config,
        repo_root=root,
        data_path=root / "data/sample/loan_sample_synthetic.csv",
        metrics_path=metrics_path,
        figures_dir=figures_dir,
    )

    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["n_raw_samples"] == 200
    assert saved["n_samples"] == 160
    assert saved["split_sizes"] == {"train": 96, "validation": 32, "test": 32}
    assert saved["selected_model"] in saved["models"]
    assert saved["models"][saved["selected_model"]]["status"] == "ok"
    xgb_status = saved["models"]["xgboost"]["status"]
    if XGBOOST_AVAILABLE:
        # A native import can succeed while estimator.fit still fails; the
        # pipeline is required to record that fit-time fallback as skipped.
        assert xgb_status in {"ok", "skipped"}
    else:
        assert xgb_status == "skipped"
    assert (figures_dir / "roc_curves.png").exists()
    assert (figures_dir / "pr_curves.png").exists()
