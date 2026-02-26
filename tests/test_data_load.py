"""Tests for src/data_load.py."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_load import EXPECTED_COLUMNS, EXPECTED_COLS, EXPECTED_ROWS, TARGET, load_raw_data, load_sample


def make_raw_frame() -> pd.DataFrame:
    data = {column: [float(index)] * EXPECTED_ROWS for index, column in enumerate(EXPECTED_COLUMNS)}
    data[TARGET] = [5.0] * EXPECTED_ROWS
    frame = pd.DataFrame(data)
    return frame.loc[:, EXPECTED_COLUMNS]


def test_load_raw_data_accepts_expected_shape(tmp_path: Path):
    path = tmp_path / "raw.csv"
    make_raw_frame().to_csv(path, index=False)

    loaded = load_raw_data(path)

    assert loaded.shape == (EXPECTED_ROWS, EXPECTED_COLS)
    assert TARGET in loaded.columns


def test_load_raw_data_rejects_missing_target(tmp_path: Path):
    path = tmp_path / "raw.csv"
    df = make_raw_frame().drop(columns=[TARGET])
    df.to_csv(path, index=False)

    with pytest.raises(AssertionError, match="target column"):
        load_raw_data(path)


def test_load_sample_requires_target(tmp_path: Path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"feature_1": [1.0, 2.0]}).to_csv(path, index=False)

    with pytest.raises(AssertionError):
        load_sample(path)
