# utils/helpers.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle

# simple model cache
_model_cache = {}

def load_model_cached(path):
    """
    Load a model from disk and cache it. Returns (model, None) on success or (None, error_msg).
    """
    path = str(path)
    if path in _model_cache:
        return _model_cache[path], None
    try:
        model = joblib.load(path)
        _model_cache[path] = model
        return model, None
    except Exception as e:
        return None, str(e)

def categorize_risk(score):
    if score >= 0.7:
        return 'High Risk'
    elif score >= 0.3:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def safe_factorize_series(ser):
    try:
        return pd.factorize(ser)[0]
    except Exception:
        return np.zeros(len(ser), dtype=int)

# Append single record to log CSV (adds timestamp)
def append_log_single(log_dir, record: dict):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = log_dir / f"module4_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df = pd.DataFrame([record])
    # ensure consistent column names
    df.to_csv(fname, index=False)
    return str(fname)

# Append batch DataFrame to log dir: saves as timestamped csv and returns path
def append_log_batch(log_dir, df):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = log_dir / f"module4_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(fname, index=False)
    return str(fname)

# Read & merge all logs into one DataFrame (adds Risk_Score and Risk_Category if missing)
def read_all_logs(log_dir):
    log_dir = Path(log_dir)
    files = sorted(log_dir.glob("module4_results_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            # try to standardize some columns
            if 'Risk_Score' not in d.columns:
                # try common names
                if 'XGBoost_Risk_Score' in d.columns:
                    d['Risk_Score'] = d['XGBoost_Risk_Score']
                elif 'LightGBM_Risk_Score' in d.columns:
                    d['Risk_Score'] = d['LightGBM_Risk_Score']
            if 'Risk_Category' not in d.columns:
                if 'XGBoost_Risk_Category' in d.columns:
                    d['Risk_Category'] = d['XGBoost_Risk_Category']
                elif 'LightGBM_Risk_Category' in d.columns:
                    d['Risk_Category'] = d['LightGBM_Risk_Category']
                else:
                    if 'Risk_Score' in d.columns:
                        d['Risk_Category'] = d['Risk_Score'].apply(lambda x: ('High Risk' if x>=0.7 else ('Medium Risk' if x>=0.3 else 'Low Risk')))
            # add timestamp column if not present
            if 'timestamp' not in d.columns:
                d['timestamp'] = datetime.utcnow().isoformat()
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True, sort=False)
    # normalize some column names
    if 'transaction_id' not in all_df.columns:
        if 'Transaction_ID' in all_df.columns:
            all_df = all_df.rename(columns={'Transaction_ID':'transaction_id'})
    # ensure Risk_Score numeric
    if 'Risk_Score' in all_df.columns:
        all_df['Risk_Score'] = pd.to_numeric(all_df['Risk_Score'], errors='coerce').fillna(0.0)
    return all_df
