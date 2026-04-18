"""Streamlit dashboard for the Agricultural Yield Prediction project."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"

st.set_page_config(page_title="Agricultural Yield Prediction", layout="wide")


@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    return meta, model


metadata, final_model = load_artifacts()
st.title("Agricultural Yield Prediction")
st.write(f"Model: {metadata['best_model_name']}, R-squared: {metadata['tuned_r2']:.3f}")
