import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "models/valuation_pipeline.pkl"
DATA_PATH = "data/processed/unicorns_with_predictions.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run models/train_model.py first.")
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run models/train_model.py first.")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle

def predict_valuation(
    total_raised_m,
    years_to_unicorn,
    investors_count,
    deal_terms,
    industry,
    hub_category,
    has_sequoia=0,
    has_a16z=0,
    has_softbank=0,
    has_tiger=0,
    has_tier1=0
):
    bundle = load_model_bundle()
    pipe = bundle["pipeline"]

    log_total_raised = np.log1p(total_raised_m)
    
    input_df = pd.DataFrame([{
        "log_total_raised": log_total_raised,
        "years_to_unicorn": float(years_to_unicorn),
        "investors_count": float(investors_count),
        "deal_terms": float(deal_terms),
        "has_tier1_investor": int(has_tier1 or has_sequoia or has_a16z or has_softbank or has_tiger),
        "has_sequoia": int(has_sequoia),
        "has_a16z": int(has_a16z),
        "has_softbank": int(has_softbank),
        "has_tiger": int(has_tiger),
        "clean_industry": industry,
        "hub_category": hub_category
    }])

    pred_log = pipe.predict(input_df)[0]
    pred_valuation_m = float(np.expm1(pred_log))

    # Standard error estimate for ~80% confidence interval (approx 1.28 * residual standard dev)
    std_log_error = 0.38
    lower_log = pred_log - (1.28 * std_log_error)
    upper_log = pred_log + (1.28 * std_log_error)

    lower_m = float(np.expm1(lower_log))
    upper_m = float(np.expm1(upper_log))

    implied_multiple = (pred_valuation_m / (total_raised_m + 1e-5))

    return {
        "pred_valuation_m": pred_valuation_m,
        "pred_valuation_b": pred_valuation_m / 1000.0,
        "ci_lower_m": lower_m,
        "ci_upper_m": upper_m,
        "implied_multiple": implied_multiple
    }

def find_closest_comps(df, industry, total_raised_m, n=3):
    # Filter by same industry first, or fallback to full df
    subset = df[df["clean_industry"] == industry].copy()
    if len(subset) < n:
        subset = df.copy()

    # Calculate distance on log total raised
    target_log_raised = np.log1p(total_raised_m)
    subset["distance"] = (np.log1p(subset["total_raised_m_imputed"]) - target_log_raised).abs()
    comps = subset.sort_values(by="distance").head(n)
    return comps
