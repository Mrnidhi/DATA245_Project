"""Streamlit dashboard for the Agricultural Yield Prediction project.

Run with:  streamlit run app_dashboard.py

Only the tuned best model from notebook 04 is loaded and used.
"""
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

st.set_page_config(
    page_title="Agricultural Yield Prediction",
    page_icon="🌾",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    return meta, model


try:
    metadata, final_model = load_artifacts()
except Exception as exc:
    st.error(f"Could not load artifacts: {exc}")
    st.info("Run the four notebooks in order (01, 02, 03, 04) to regenerate models.")
    st.stop()


@st.cache_data
def _column_template() -> list[str]:
    df = pd.read_csv(DATA_DIR / "processed" / "X_test_filtered.csv", nrows=1)
    return df.columns.tolist()


@st.cache_data
def _train_stats():
    """Mean and std for raw numeric columns, plus a median row to use as a
    base template when a manual user input only sets a few features."""
    X_train = pd.read_csv(DATA_DIR / "processed" / "X_train_filtered.csv")
    z_cols = [
        "Temperature", "Humidity", "Rainfall", "Solar_Radiation", "Wind_Speed",
        "OC", "CEC", "Ca", "Mg", "Cu", "Mo", "Zn", "Fe", "N",
        "NDVI", "LAI", "Chlorophyll",
        "Water_Holding_Capacity", "Bulk_Density", "Irrigation_Frequency",
        "Sand", "Silt", "Clay", "Slope", "Elevation",
    ]
    stats = {c: (float(X_train[c].mean()), float(X_train[c].std())) for c in z_cols if c in X_train}
    medians = X_train.median(numeric_only=True).to_dict()
    modes = X_train.select_dtypes(include="object").mode().iloc[0].to_dict()
    template = {**medians, **modes}
    return stats, template


@st.cache_data
def _build_demo_samples() -> list[dict]:
    X_test = pd.read_csv(DATA_DIR / "processed" / "X_test_filtered.csv")
    y_test = pd.read_csv(DATA_DIR / "processed" / "y_test.csv").squeeze("columns")
    df = X_test.copy()
    df["_actual"] = y_test.values

    out: list[dict] = []

    def take(label, blurb, candidates, mode):
        if candidates.empty:
            return
        if mode == "max":
            row = candidates.nlargest(1, "_actual").iloc[0]
        elif mode == "min":
            row = candidates.nsmallest(1, "_actual").iloc[0]
        else:
            med = candidates["_actual"].median()
            row = candidates.iloc[(candidates["_actual"] - med).abs().argsort()[:1]].iloc[0]
        out.append({
            "label": label,
            "blurb": blurb,
            "actual": float(row["_actual"]),
            "features": row.drop(labels=["_actual"]).to_dict(),
        })

    take("🌾 Premium Wheat",   "Happy wheat farm with strong vegetation and good soil.",
         df[df["Crop_Type"] == "Wheat"], "max")
    take("🍚 Monsoon Rice",    "Rice in a rainy, well-irrigated profile.",
         df[df["Crop_Type"] == "Rice"], "max")
    take("🌽 Stressed Maize",  "Drought-stressed maize plot, weak NDVI and water proxy.",
         df[df["Crop_Type"] == "Maize"], "min")
    take("🫘 Average Soybean", "A soybean farm at the median yield within its crop.",
         df[df["Crop_Type"] == "Soybean"], "median")
    take("🌱 Top Soybean",     "Best-performing soybean farm in the test set.",
         df[df["Crop_Type"] == "Soybean"], "max")
    return out


def _build_manual_row(stats, template, *, temperature, rainfall, ndvi, oc,
                      irrigation, crop_type, fert_type, pesticide):
    """Override a few raw inputs on the median template, then recompute the
    engineered features that depend on them so the prediction actually
    responds to the slider changes."""
    row = template.copy()
    row["Temperature"] = float(temperature)
    row["Rainfall"] = float(rainfall)
    row["NDVI"] = float(ndvi)
    row["OC"] = float(oc)
    row["Irrigation_Frequency"] = float(irrigation)
    row["Crop_Type"] = crop_type
    row["Fertilizer_Type"] = fert_type
    row["Pesticide_Usage"] = pesticide

    row["NPK_ratio"] = row["N"] / (row["P"] + row["K"] + 1.0)
    row["Soil_Health_Index"] = row["OC"] + row["CEC"] / 10 + row["Ca"] / 1000 + row["Mg"] / 200
    row["Vegetation_Index"] = row["NDVI"] * row["LAI"] * row["Chlorophyll"] / 50.0
    row["Weather_Stress"] = abs(row["Temperature"] - 25.0) + abs(row["Rainfall"] - 150.0) / 100.0
    row["Management_Proxy"] = row["NDVI"] + row["LAI"] / 3 + row["Chlorophyll"] / 40

    def z(col):
        m, s = stats[col]
        return (row[col] - m) / s if s > 0 else 0.0

    fert_score = {"Chemical": -1.0, "Organic": 1.0, "Mixed": 0.5}[row["Fertilizer_Type"]]
    pest_score = {"Low": 1.0, "Medium": 0.0, "High": -1.0}[row["Pesticide_Usage"]]

    row["Climate_Proxy"] = (
        0.35 * z("Temperature") + 0.30 * z("Rainfall") + 0.15 * z("Humidity")
        + 0.10 * z("Solar_Radiation") - 0.10 * z("Wind_Speed")
    )
    row["Soil_Proxy"] = (
        0.20 * z("OC") + 0.18 * z("CEC") + 0.16 * z("Ca") + 0.14 * z("Mg")
        + 0.20 * z("Cu") + 0.18 * z("Mo") + 0.10 * z("Zn") + 0.08 * z("Fe") + 0.10 * z("N")
    )
    row["Management_Quality_Proxy"] = (
        0.35 * z("NDVI") + 0.30 * z("LAI") + 0.25 * z("Chlorophyll")
        + 0.10 * fert_score + 0.10 * pest_score
    )
    row["Water_Proxy"] = (
        0.40 * z("Rainfall") + 0.25 * z("Water_Holding_Capacity")
        - 0.20 * z("Bulk_Density") + 0.15 * z("Irrigation_Frequency")
    )
    row["Terrain_Proxy"] = (
        0.35 * z("Sand") - 0.20 * z("Silt") - 0.25 * z("Clay")
        + 0.25 * z("Slope") + 0.15 * z("Elevation")
    )
    row["SoilHealth_Management_interaction"] = float(np.tanh(row["Soil_Proxy"] * row["Management_Quality_Proxy"]))
    row["Climate_Water_interaction"] = float(np.tanh(row["Climate_Proxy"] * row["Water_Proxy"]))
    row["Management_Water_interaction"] = float(np.tanh(row["Management_Quality_Proxy"] * row["Water_Proxy"]))
    return row


# Header --------------------------------------------------------------------
st.markdown(
    """
    <div class="farm-banner">
        ☀️ ☁️ 🚜 &nbsp; 🌾 🌽 🌿 🌱 &nbsp; 🐄 🌻
        <small>Predicting crop yield from climate, soil and management features</small>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🌾 Agricultural Yield Prediction")
st.caption(
    f"Tuned {metadata['best_model_name']} on the Agri-yield test set. "
    f"R² = {metadata['tuned_r2']:.3f}, RMSE = {metadata['tuned_rmse']:.3f} t/ha."
)

template_cols = _column_template()
stats, template = _train_stats()
samples = _build_demo_samples()

tab_predict, tab_diag = st.tabs(["🚜 Predict Yield", "📊 Model Diagnostics"])


# Predict tab ---------------------------------------------------------------
with tab_predict:
    st.subheader("🌻 Sample farms from the test set")
    st.caption("Each button loads a real row and runs it through the tuned model.")

    btn_cols = st.columns(len(samples))
    for i, sample in enumerate(samples):
        if btn_cols[i].button(sample["label"], key=f"demo_btn_{i}", use_container_width=True):
            st.session_state["demo_choice"] = i

    if "demo_choice" in st.session_state:
        choice = samples[st.session_state["demo_choice"]]
        X_row = pd.DataFrame([choice["features"]]).reindex(columns=template_cols, fill_value=0)
        pred = float(final_model.predict(X_row)[0])

        if pred >= 8.0:
            mood = "🤩 Happy harvest!"
        elif pred >= 5.5:
            mood = "🙂 Decent year"
        elif pred >= 3.0:
            mood = "😐 Below average"
        else:
            mood = "😟 Crop in trouble"

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{mood}", f"{pred:.2f} t/ha", help="Model's predicted yield")
        c2.metric("🎯 Actual (test row)", f"{choice['actual']:.2f} t/ha")
        c3.metric("📏 Absolute error", f"{abs(pred - choice['actual']):.2f} t/ha")

        st.info(f"**{choice['label']}**: {choice['blurb']}")

    st.markdown("---")
    st.subheader("🎛️ Try your own farm inputs")
    st.caption("Adjust the most influential variables. Other features are held at training-set medians.")

    with st.form("manual_form"):
        c1, c2 = st.columns(2)
        with c1:
            temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0, step=0.5)
            rainfall = st.slider("Rainfall (mm)", 0.0, 350.0, 150.0, step=5.0)
            ndvi = st.slider("NDVI (vegetation index)", 0.0, 1.0, 0.6, step=0.01)
            oc = st.slider("Soil organic carbon (%)", 0.1, 3.0, 1.0, step=0.05)
        with c2:
            irrigation = st.slider("Irrigation frequency (per season)", 0, 15, 5)
            crop_type = st.selectbox("Crop type", ["Wheat", "Rice", "Maize", "Soybean"])
            fert_type = st.selectbox("Fertilizer", ["Chemical", "Organic", "Mixed"])
            pesticide = st.selectbox("Pesticide usage", ["Low", "Medium", "High"])

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        manual_row = _build_manual_row(
            stats, template,
            temperature=temperature, rainfall=rainfall, ndvi=ndvi, oc=oc,
            irrigation=irrigation, crop_type=crop_type,
            fert_type=fert_type, pesticide=pesticide,
        )
        X_manual = pd.DataFrame([manual_row]).reindex(columns=template_cols, fill_value=0)
        pred = float(final_model.predict(X_manual)[0])

        if pred >= 8.0:
            mood = "🤩 Excellent yield"
        elif pred >= 5.5:
            mood = "🙂 Decent yield"
        elif pred >= 3.0:
            mood = "😐 Below average"
        else:
            mood = "😟 Stressed crop"

        st.metric(f"{mood}", f"{pred:.2f} t/ha")

        summary = pd.DataFrame([{
            "Crop": crop_type,
            "Temperature (°C)": temperature,
            "Rainfall (mm)": rainfall,
            "NDVI": ndvi,
            "OC (%)": oc,
            "Irrigation": irrigation,
            "Fertilizer": fert_type,
            "Pesticide": pesticide,
        }])
        st.dataframe(summary, use_container_width=True, hide_index=True)


# Diagnostics tab -----------------------------------------------------------
with tab_diag:
    st.subheader("📊 How well does the tuned model work?")

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Best model", metadata["best_model_name"])
    c2.metric("🎯 Test R²", f"{metadata['tuned_r2']:.3f}")
    c3.metric("📏 Test RMSE (t/ha)", f"{metadata['tuned_rmse']:.3f}")

    st.markdown("#### 🔧 Did fine tuning actually help?")
    cmp_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R²"],
        "Base": [
            f"{metadata['base_rmse']:.4f}",
            f"{metadata['base_mae']:.4f}",
            f"{metadata['base_r2']:.4f}",
        ],
        "Tuned": [
            f"{metadata['tuned_rmse']:.4f}",
            f"{metadata['tuned_mae']:.4f}",
            f"{metadata['tuned_r2']:.4f}",
        ],
        "Improvement %": [
            f"{metadata['rmse_improvement_pct']:+.2f}",
            f"{(metadata['base_mae'] - metadata['tuned_mae']) / metadata['base_mae'] * 100:+.2f}",
            f"{metadata['r2_improvement_pct']:+.2f}",
        ],
    })
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    fig_bvt = FIG_DIR / "08_base_vs_tuned.png"
    if fig_bvt.exists():
        st.image(str(fig_bvt), caption="Base vs tuned on the test set")

    st.markdown("#### Best hyperparameters")
    if metadata.get("best_params"):
        st.json(metadata["best_params"])

    st.markdown("#### Diagnostic plots")
    col_left, col_right = st.columns(2)
    with col_left:
        for name, caption in [("05_predicted_vs_actual.png", "Predicted vs actual"),
                              ("07_learning_curve.png", "Learning curve")]:
            fp = FIG_DIR / name
            if fp.exists():
                st.image(str(fp), caption=caption)
    with col_right:
        for name, caption in [("06_residuals.png", "Residuals"),
                              ("03_feature_importance.png", "Top 15 features")]:
            fp = FIG_DIR / name
            if fp.exists():
                st.image(str(fp), caption=caption)


st.markdown("---")
st.caption(
    f"DATA 245 project. Best model: {metadata['best_model_name']} (tuned). "
    f"Train rows: {metadata['train_size']:,}, test rows: {metadata['test_size']:,}."
)
