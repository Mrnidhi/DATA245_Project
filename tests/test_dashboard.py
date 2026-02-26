"""Smoke tests for the Streamlit dashboard and dashboard-facing artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"


def test_metadata_is_well_formed():
    meta_path = MODEL_DIR / "metadata.json"
    if not meta_path.exists():
        pytest.skip("metadata.json not generated yet -- run run_fixed_pipeline.py")

    with open(meta_path) as f:
        meta = json.load(f)

    required = {
        "best_model_name",
        "best_model_rmse",
        "best_model_r2",
        "best_model_mae",
        "selected_features",
        "n_features_selected",
        "train_size",
        "test_size",
    }
    assert required.issubset(meta.keys())
    assert meta["best_model_r2"] > 0.5
    assert meta["best_model_rmse"] > 0
    assert meta["train_size"] > 0 and meta["test_size"] > 0


def test_model_results_table_present():
    csv_path = MODEL_DIR / "model_results.csv"
    if not csv_path.exists():
        pytest.skip("model_results.csv not generated yet")

    df = pd.read_csv(csv_path)
    assert {"Model", "RMSE", "MAE", "R2", "CV_RMSE"}.issubset(df.columns)
    assert len(df) >= 4
    assert df["R2"].max() > 0.5


def test_required_figures_exist():
    expected = [
        "01_yield_distribution.png",
        "02_correlation_heatmap.png",
        "03_feature_importance.png",
        "04_model_comparison.png",
        "05_predicted_vs_actual.png",
        "06_residuals.png",
        "07_learning_curve.png",
    ]
    missing = [name for name in expected if not (FIG_DIR / name).exists()]
    if missing:
        pytest.skip(f"Figures not generated yet: {missing}")
