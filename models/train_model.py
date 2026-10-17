import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

PROCESSED_DATA_PATH = "data/processed/cleaned_unicorns.csv"
MODEL_SAVE_PATH = "models/valuation_pipeline.pkl"
RESULTS_DATA_PATH = "data/processed/unicorns_with_predictions.csv"

def train_and_evaluate():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Loading cleaned dataset ({len(df):,} unicorns)...")

    df["log_valuation"] = np.log1p(df["valuation_m"])

    numeric_features = ["tenure_years", "investors_count"]
    binary_features = [
        "is_ai_frontier", "has_tier1_investor", "has_sequoia", "has_a16z",
        "has_khosla", "has_thrive", "has_google", "has_softbank", "has_accel", "has_tiger"
    ]
    categorical_features = ["clean_industry", "hub_category"]

    feature_cols = numeric_features + binary_features + categorical_features
    X = df[feature_cols].copy()
    y = df["log_valuation"].values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("bin", "passthrough", binary_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    models = {
        "Ridge Regression (Interpretable)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=180, max_depth=8, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=180, learning_rate=0.06, max_depth=4, random_state=42)
    }

    best_model_name = None
    best_r2 = -float("inf")
    best_pipeline = None

    print("\n--- Model Benchmark (5-Fold Cross-Validation on Log-Valuation) ---")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
        mean_cv_r2 = cv_scores.mean()
        std_cv_r2 = cv_scores.std()
        print(f"{name:35s} | CV R²: {mean_cv_r2:.4f} (+/- {std_cv_r2:.4f})")

        if mean_cv_r2 > best_r2:
            best_r2 = mean_cv_r2
            best_model_name = name
            best_pipeline = pipe

    print(f"\nSelected Best Model: {best_model_name} (CV R²: {best_r2:.4f})")

    # Fit best pipeline on full training data
    best_pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_test_pred_log = best_pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred_log)
    
    y_test_actual_m = np.expm1(y_test)
    y_test_pred_m = np.expm1(y_test_pred_log)
    test_mae_m = mean_absolute_error(y_test_actual_m, y_test_pred_m)

    print("\n--- Test Set Evaluation ---")
    print(f"Test Set Log-Scale R²: {test_r2:.4f}")
    print(f"Test Set Mean Absolute Dollar Error: ${test_mae_m:,.2f} Million")

    # Generate predictions across entire dataset
    full_pred_log = best_pipeline.predict(X)
    df["predicted_valuation_m"] = np.expm1(full_pred_log).round(1)
    df["predicted_valuation_b"] = (df["predicted_valuation_m"] / 1000.0).round(2)
    
    # Residual = Actual - Predicted
    df["residual_m"] = (df["valuation_m"] - df["predicted_valuation_m"]).round(1)
    df["residual_pct"] = ((df["residual_m"] / df["predicted_valuation_m"]) * 100).round(1)

    def categorize_efficiency(residual_pct):
        if residual_pct >= 50.0:
            return "Decacorn Alpha (Exceeds Market Trajectory)"
        elif residual_pct <= -25.0:
            return "Valuation Discounted / Stagnant"
        else:
            return "Fair Market Valuation"

    df["efficiency_tier"] = df["residual_pct"].apply(categorize_efficiency)

    df.to_csv(RESULTS_DATA_PATH, index=False)
    print(f"\nAugmented dataset with predictions saved to {RESULTS_DATA_PATH}!")

    # Train Ridge pipeline for feature coefficient weights
    ridge_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0))])
    ridge_pipeline.fit(X, y)

    cat_encoder = ridge_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + binary_features + cat_names
    ridge_coefs = ridge_pipeline.named_steps["regressor"].coef_

    coef_df = pd.DataFrame({
        "feature": all_feature_names,
        "coefficient": ridge_coefs,
        "impact_pct": ((np.exp(ridge_coefs) - 1.0) * 100).round(2)
    }).sort_values(by="coefficient", ascending=False)

    print("\nTop 10 Valuation Drivers (Ridge Elasticity):")
    print(coef_df.head(10).to_string(index=False))

    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "cv_r2": best_r2,
        "test_r2": test_r2,
        "test_mae_m": test_mae_m,
        "coefficients": coef_df,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "binary_features": binary_features,
        "categorical_features": categorical_features
    }

    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel pipeline and metadata successfully saved to {MODEL_SAVE_PATH}!")

if __name__ == "__main__":
    train_and_evaluate()
