import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Locate the active processed CSV and model bundle dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATHS = [
    "data/processed/unicorns_with_predictions.csv",
    os.path.join(BASE_DIR, "data", "processed", "unicorns_with_predictions.csv"),
]

MODEL_PATHS = [
    "models/valuation_pipeline.pkl",
    os.path.join(BASE_DIR, "models", "valuation_pipeline.pkl"),
]

def get_valid_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

def load_data():
    path = get_valid_path(DATA_PATHS)
    if not os.path.exists(path):
        import sys
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from data.download_and_clean import clean_data
        clean_data()
        path = get_valid_path(DATA_PATHS)
    df = pd.read_csv(path)
    # Ensure bubble sizing for map emphasizes giants like Anthropic ($965B) & OpenAI ($840B)
    # while keeping smaller $1B unicorns visible
    df["map_bubble_size"] = np.log1p(df["valuation_b"]) * 5.0 + 3.0
    return df

@st.cache_resource
def load_model_bundle():
    path = get_valid_path(MODEL_PATHS)
    if not os.path.exists(path):
        import sys
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from models.train_model import train_and_evaluate
        train_and_evaluate()
        path = get_valid_path(MODEL_PATHS)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

def predict_valuation(
    tenure_years,
    investors_count,
    industry,
    hub_category,
    is_ai_frontier=0,
    has_sequoia=0,
    has_a16z=0,
    has_google=0,
    has_khosla=0,
    has_thrive=0,
    has_softbank=0,
    has_accel=0,
    has_tiger=0
):
    bundle = load_model_bundle()
    pipe = bundle["pipeline"]

    has_tier1 = int(
        has_sequoia or has_a16z or has_google or has_khosla or 
        has_thrive or has_softbank or has_accel or has_tiger
    )

    input_df = pd.DataFrame([{
        "tenure_years": float(tenure_years),
        "investors_count": float(investors_count),
        "is_ai_frontier": int(is_ai_frontier),
        "has_tier1_investor": has_tier1,
        "has_sequoia": int(has_sequoia),
        "has_a16z": int(has_a16z),
        "has_khosla": int(has_khosla),
        "has_thrive": int(has_thrive),
        "has_google": int(has_google),
        "has_softbank": int(has_softbank),
        "has_accel": int(has_accel),
        "has_tiger": int(has_tiger),
        "clean_industry": industry,
        "hub_category": hub_category
    }])

    pred_log = pipe.predict(input_df)[0]
    pred_valuation_m = float(np.expm1(pred_log))

    std_log_error = 0.42
    lower_log = pred_log - (1.28 * std_log_error)
    upper_log = pred_log + (1.28 * std_log_error)

    lower_m = float(np.expm1(lower_log))
    upper_m = float(np.expm1(upper_log))

    return {
        "pred_valuation_m": pred_valuation_m,
        "pred_valuation_b": pred_valuation_m / 1000.0,
        "ci_lower_m": lower_m,
        "ci_upper_m": upper_m,
        "ci_lower_b": lower_m / 1000.0,
        "ci_upper_b": upper_m / 1000.0
    }

def find_closest_comps(df, industry, hub_category, is_ai_frontier, n=3):
    subset = df[df["clean_industry"] == industry].copy()
    if is_ai_frontier:
        ai_subset = subset[subset["is_ai_frontier"] == 1]
        if len(ai_subset) >= n:
            subset = ai_subset

    subset["hub_match"] = (subset["hub_category"] == hub_category).astype(int)
    comps = subset.sort_values(by=["hub_match", "valuation_b"], ascending=[False, False]).head(n)
    return comps
