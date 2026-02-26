"""Tests for src/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CATEGORICAL_COLS,
    DATE_FEATURE_COLS,
    NPK_RATIO_COLS,
    NUMERIC_COLS,
    TARGET,
    add_date_features,
    build_encoder,
    build_pipeline,
    build_preprocessor,
    build_scaler,
    split_features_target,
)


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    rows = 12
    rng = np.random.default_rng(42)

    data = {col: rng.normal(size=rows) for col in NUMERIC_COLS if col not in DATE_FEATURE_COLS}
    for col in CATEGORICAL_COLS:
        data[col] = rng.choice(["a", "b", "c"], size=rows)

    data["Planting_Date"] = pd.date_range("2024-01-01", periods=rows).astype(str)
    data["Harvest_Date"] = pd.date_range("2024-04-01", periods=rows).astype(str)
    data[TARGET] = rng.uniform(1.0, 10.0, size=rows)

    frame = pd.DataFrame(data)
    frame.loc[0, "Temperature"] = np.nan
    frame.loc[1, "Soil_Type"] = np.nan
    return frame


def test_add_date_features_creates_expected_columns(tiny_frame: pd.DataFrame):
    result = add_date_features(tiny_frame)

    for column in DATE_FEATURE_COLS + NPK_RATIO_COLS:
        assert column in result.columns
    assert "Planting_Date" not in result.columns
    assert "Harvest_Date" not in result.columns


def test_split_features_target_removes_target(tiny_frame: pd.DataFrame):
    features, target = split_features_target(tiny_frame)

    assert TARGET not in features.columns
    assert len(features) == len(target)


def test_split_features_target_rejects_missing_target(tiny_frame: pd.DataFrame):
    with pytest.raises(KeyError, match=TARGET):
        split_features_target(tiny_frame.drop(columns=[TARGET]))


def test_encoder_one_hot_shape(tiny_frame: pd.DataFrame):
    enc = build_encoder()
    out = enc.fit_transform(tiny_frame[CATEGORICAL_COLS].fillna("missing"))

    assert out.shape[0] == len(tiny_frame)
    assert out.shape[1] >= len(CATEGORICAL_COLS)


def test_scaler_zero_mean_unit_var(tiny_frame: pd.DataFrame):
    sc = build_scaler()
    numeric = add_date_features(tiny_frame)[NUMERIC_COLS]
    filled = numeric.fillna(numeric.median())
    out = sc.fit_transform(filled)
    nonconstant = filled.nunique() > 1

    assert np.allclose(out[:, nonconstant].mean(axis=0), 0.0)
    assert np.allclose(out[:, nonconstant].std(axis=0), 1.0)


def test_preprocessor_handles_missing_values(tiny_frame: pd.DataFrame):
    features, _ = split_features_target(tiny_frame)
    out = build_preprocessor().fit_transform(features)

    assert not np.isnan(out).any()


def test_pipeline_handles_missing_values(tiny_frame: pd.DataFrame):
    features, _ = split_features_target(tiny_frame)
    out = build_pipeline().fit_transform(features)

    assert not np.isnan(out).any()
