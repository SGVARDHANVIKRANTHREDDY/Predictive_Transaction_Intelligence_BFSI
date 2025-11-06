# deployment/admin.py
"""
Admin utilities & admin UI pieces for FraudDetection BFSI.

Features:
- get_model_status() -> detect model files, load lightweight metadata
- compute_model_metrics(logs_df) -> compute simple metrics (counts, mean risk, accuracy/precision/recall if labels exist)
- reload_models() -> force reload of XGBoost / LightGBM models from disk (admin only)
- admin_panel(st, ...) -> renders the admin UI inside Streamlit (cards, charts, log filters)
"""

from pathlib import Path
from datetime import datetime, timedelta
import json
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Paths expected to be the same as app_streamlit.py
BASE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = BASE_DIR / "Module4_Deployment"
DATA_DIR = APP_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = APP_DIR / "models"

XGB_MODEL_FILE = MODELS_DIR / "XGBoost_tuned.pkl"
LGB_MODEL_FILE = MODELS_DIR / "LightGBM_tuned.pkl"

# simple global cache for currently loaded models (kept in memory)
MODEL_CACHE = {
    "xgb": None,
    "lgb": None,
    "loaded_at": None
}


def _safe_load_model(path: Path):
    if not path.exists():
        return None, f"Missing file: {path}"
    try:
        m = joblib.load(str(path))
        return m, None
    except Exception as e:
        return None, str(e)


def get_model_status():
    """
    Return dict with presence, last-modified timestamp, and load status.
    """
    status = {}
    for name, path in [("XGBoost", XGB_MODEL_FILE), ("LightGBM", LGB_MODEL_FILE)]:
        status[name] = {
            "exists": path.exists(),
            "path": str(path),
            "last_modified": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
        }
    status["cache_loaded_at"] = MODEL_CACHE.get("loaded_at")
    return status


def reload_models():
    """
    Force reload models from disk into MODEL_CACHE.
    Returns tuple (ok:boolean, details:dict)
    """
    details = {}
    ok = True
    xgb, xerr = _safe_load_model(XGB_MODEL_FILE)
    lgb, lerr = _safe_load_model(LGB_MODEL_FILE)
    if xerr:
        details["xgb_error"] = xerr
        ok = False
    if lerr:
        details["lgb_error"] = lerr
        ok = False
    MODEL_CACHE["xgb"] = xgb
    MODEL_CACHE["lgb"] = lgb
    MODEL_CACHE["loaded_at"] = datetime.utcnow().isoformat()
    details["loaded_at"] = MODEL_CACHE["loaded_at"]
    return ok, details


def compute_model_metrics(logs_df: pd.DataFrame, window_days: int = 7):
    """
    Compute simple monitoring metrics from logs DataFrame.
    - total_predictions
    - mean_risk_overall
    - mean_risk_recent (last window_days)
    - high_risk_count
    - drift_proxy = mean_recent - mean_overall
    - if 'is_fraud' exists -> compute accuracy/precision/recall at threshold 0.5
    Returns metrics dict.
    """
    if logs_df is None or logs_df.empty:
        return {"note": "no logs available"}

    df = logs_df.copy()
    # ensure numeric Risk_Score
    if 'Risk_Score' in df.columns:
        df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce').fillna(0.0)
    else:
        df['Risk_Score'] = 0.0

    metrics = {}
    metrics['total_predictions'] = len(df)
    metrics['mean_risk_overall'] = float(df['Risk_Score'].mean())
    # recent window
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=window_days)
        recent = df[df['timestamp'] >= cutoff]
    else:
        recent = df.tail(int(len(df) * 0.1) or 1)

    metrics['mean_risk_recent'] = float(recent['Risk_Score'].mean()) if not recent.empty else None
    metrics['high_risk_count'] = int((df['Risk_Score'] >= 0.8).sum())
    # drift proxy
    if metrics['mean_risk_recent'] is not None:
        metrics['drift_proxy'] = metrics['mean_risk_recent'] - metrics['mean_risk_overall']
    else:
        metrics['drift_proxy'] = None

    # If labels exist, compute simple classification metrics at threshold 0.5
    if 'is_fraud' in df.columns:
        # coerce to binary
        df['is_fraud'] = df['is_fraud'].astype(int)
        preds = (df['Risk_Score'] >= 0.5).astype(int)
        tp = int(((preds == 1) & (df['is_fraud'] == 1)).sum())
        tn = int(((preds == 0) & (df['is_fraud'] == 0)).sum())
        fp = int(((preds == 1) & (df['is_fraud'] == 0)).sum())
        fn = int(((preds == 0) & (df['is_fraud'] == 1)).sum())
        metrics['label_metrics'] = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        metrics['accuracy'] = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else None
        metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else None
        metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    else:
        metrics['note'] = "no ground-truth labels ('is_fraud') found; label metrics unavailable"

    return metrics


def _plot_metric_trend(df: pd.DataFrame, metric_col='Risk_Score', resample='D'):
    """
    Returns matplotlib figure with trend of metric_col aggregated by resample period.
    """
    if df is None or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha='center')
        return fig
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        series = df.set_index('timestamp')[metric_col].resample(resample).mean().fillna(0)
    else:
        series = pd.Series(df[metric_col].values)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(series.index, series.values, marker='o')
    ax.set_title(f"{metric_col} trend ({resample})")
    ax.set_xlabel("Time")
    ax.set_ylabel(metric_col)
    fig.tight_layout()
    return fig


def admin_panel(st, current_user: str, logs_df: pd.DataFrame, append_action_fn):
    """
    Renders the Admin Panel inside Streamlit.
    - current_user: name/email for action logging
    - logs_df: DataFrame of logs (read_all_logs)
    - append_action_fn: function to append audit actions (from audits.py)
    """
    st.header("🔧 Admin Panel — Model Monitoring & Ops")

    # Model status cards
    status = get_model_status()
    col1, col2, col3 = st.columns(3)
    col1.metric("XGBoost file", "Present" if status['XGBoost']['exists'] else "Missing",
                f"Last modified: {status['XGBoost']['last_modified'] or 'N/A'}")
    col2.metric("LightGBM file", "Present" if status['LightGBM']['exists'] else "Missing",
                f"Last modified: {status['LightGBM']['last_modified'] or 'N/A'}")
    col3.metric("Models loaded (cache)", "Yes" if MODEL_CACHE.get('xgb') is not None else "No",
                f"Loaded at: {MODEL_CACHE.get('loaded_at') or 'N/A'}")

    st.divider()

    # Reload models button (admin only)
    if st.button("🔁 Reload Models from disk (admin only)"):
        ok, details = reload_models()
        append_action_fn(current_user or 'admin', "reload_models", json.dumps(details))
        if ok:
            st.success("Models reloaded into memory.")
        else:
            st.error(f"Reload errors: {details}")
    st.markdown("**Note:** This does not retrain models — it only reloads pickled model files from disk.")

    st.divider()

    # Monitoring metrics from logs
    st.subheader("Model monitoring metrics (derived from prediction logs)")
    metrics = compute_model_metrics(logs_df)
    st.json(metrics)

    st.markdown("### Risk Score Trend (last 30 days)")
    fig = _plot_metric_trend(logs_df, metric_col='Risk_Score', resample='D')
    st.pyplot(fig)

    st.divider()
    # Error & Audit viewer
    st.subheader("Error & Audit Viewer")
    st.markdown("Filter audit log events below (from actions_log.csv).")
    # actions are stored separately; admins should use audits.read_actions() in app to fetch them.
    # Here we just show logs_df errors as well
    if logs_df is None or logs_df.empty:
        st.info("No prediction logs to show.")
    else:
        # show last 200 prediction logs + simple filters
        st.markdown("Recent prediction logs (sample)")
        st.dataframe(logs_df.sort_values('timestamp', ascending=False).head(200), use_container_width=True)

    # NOTE: download and more advanced log management happens in the main app (logs page)
    st.markdown("### Quick Diagnostics")
    st.markdown("- Check model files presence.  \n- Reload models after updating pickle files.  \n- For retraining, use a dedicated training pipeline (not available in-app).")

    return True
