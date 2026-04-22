# Models

All trained regression models and performance metrics.

## Production Model

- svr_fine_tuned.joblib: SVR fine-tuned (kernel=rbf, C=10.0, epsilon=0.01)

Use this for production predictions.

## Candidate Models

- lasso.joblib: Lasso (alpha=0.1) - Top performer
- linear_regression.joblib: Linear regression
- ridge.joblib: Ridge (alpha=1.0)
- knn.joblib: KNN (n_neighbors=5)
- random_forest.joblib: Random Forest (100 trees)
- gradient_boosting.joblib: Gradient Boosting (100 trees)
- xgboost.joblib: XGBoost (if available)

## Performance Records

- model_results.csv: Comparison table of all models
  Columns: Model, RMSE, MAE, R2, CV_RMSE

- metadata.json: Pipeline configuration
  - n_features_selected: 15
  - selected_features: Feature names
  - best_model: Best performing model
  - best_model_rmse: Test RMSE
  - best_model_r2: Test R-squared
  - best_model_mae: Test MAE

## Top 3 Models by RMSE

1. Lasso: 2.64
2. Gradient Boosting: 2.63
3. Random Forest: 2.63

## Usage

```python
import joblib
import pandas as pd

model = joblib.load("svr_fine_tuned.joblib")
X = pd.read_csv("../data/processed/X_test_filtered.csv")
predictions = model.predict(X)
```

## Retraining

Run the pipeline to regenerate all models:
```bash
jupyter notebook ../notebooks/00_complete_pipeline.ipynb
```
