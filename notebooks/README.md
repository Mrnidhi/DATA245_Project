# Notebooks

Run these in order. Each one writes the artifacts the next one consumes.

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_eda.ipynb` | Inspect the raw dataset (shape, distribution, per-crop stats, correlations). |
| 02 | `02_feature_engineering.ipynb` | Build proxy and interaction features, train/test split, save preprocessor. |
| 03 | `03_model_comparison.ipynb` | Train 7 candidate regressors with 3-fold CV, pick the best one by RMSE. |
| 04 | `04_fine_tuning.ipynb` | Tune the winner with `RandomizedSearchCV`, compare base vs tuned, save `best_model.joblib`. |

## Run

```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --output 02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_model_comparison.ipynb --output 03_model_comparison.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_fine_tuning.ipynb --output 04_fine_tuning.ipynb
```

After 04 finishes, launch the dashboard:

```bash
streamlit run app_dashboard.py
```

## Reproducibility

`random_state=42` everywhere (split, models, RandomizedSearchCV). Reruns match.
