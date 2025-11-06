"""
models_utils.py - Universal model loader utility for Predictive Transaction Intelligence app

Functions:
    - load_model_cached(model_path: str): Streamlit-cached ML model loader
    - list_available_models(models_dir: str): List all model files in directory
    - reload_model(model_path: str): Force reload (bypass Streamlit cache)
"""

import os
import joblib
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    """Load XGBoost / LightGBM / sklearn-compatible model and cache in Streamlit."""
    try:
        if not model_path or not os.path.exists(model_path):
            st.error(f"⚠️ Model file not found: `{model_path}`")
            return None
        if not model_path.endswith((".joblib", ".pkl")):
            st.error("❌ Invalid model file type. Expected .joblib or .pkl")
            return None
        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            st.warning("⚠️ Loaded object does not have a `predict()` method.")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def list_available_models(models_dir: str = "models"):
    """Return all .joblib or .pkl model files in the directory."""
    try:
        if not os.path.exists(models_dir):
            return []
        return [
            os.path.join(models_dir, f)
            for f in os.listdir(models_dir)
            if f.endswith((".joblib", ".pkl"))
        ]
    except Exception as e:
        st.error(f"⚠️ Could not list models: {e}")
        return []

def reload_model(model_path: str):
    """Force-reload model from disk (no cache)."""
    try:
        model = joblib.load(model_path)
        st.success("🔄 Model reloaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error reloading model: {e}")
        return None
