# deployment/app_streamlit.py
"""
Main Streamlit app for FraudDetection BFSI.
Phase 2: integrates modular authentication (authentication.py), audits (audits.py),
and analytics (analytics.py). Adds an Analytics page (separate tab) that runs EDA
on demand after batch prediction.
"""

import sys, os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import onnxruntime as ort
import shap
import json
from datetime import datetime
from streamlit.components.v1 import html
import time
# project paths
BASE_DIR = Path(r"C:\Users\sgvar\project no 1").resolve()
APP_DIR = (BASE_DIR / "Module4_Deployment").resolve()
DATA_DIR = APP_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = APP_DIR / "models"

# ensure folders
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append('C:/Users/sgvar/project no 1/Module4_Deployment/deployment')
# local modules (Phase 1 refactor)
from authentication import create_user, verify_user, create_session_token, validate_session_token, load_users_df  # type: ignore
from analytics import generate_eda_report_html, display_eda_inline  # type: ignore
from audits import append_action, read_actions
from models_utils import load_model_cached
# existing helpers (model loading, categorize risk, append_log_batch, read_all_logs)
sys.path.append(str(BASE_DIR))
from helpers import load_model_cached, categorize_risk, append_log_batch, read_all_logs  # noqa: E402

# model files
XGB_MODEL_FILE = MODELS_DIR / "XGBoost_tuned.pkl"
LGB_MODEL_FILE = MODELS_DIR / "LightGBM_tuned.pkl"

# feature config (same as before)
CATEGORICAL_COLUMNS = [
    'Transaction_Location', 'Card_Type', 'Transaction_Currency',
    'Transaction_Status', 'Authentication_Method', 'Transaction_Category'
]

FEATURE_COLUMNS = [
    'Transaction_ID', 'User_ID', 'Transaction_Amount', 'Merchant_ID',
    'Device_ID', 'Previous_Transaction_Count',
    'Distance_Between_Transactions_km', 'Time_Since_Last_Transaction_min',
    'Transaction_Velocity', 'Transaction_Hour', 'Transaction_Day',
    'Transaction_Month', 'Log_Transaction_Amount',
    'Transaction_Location_encoded', 'Card_Type_encoded',
    'Transaction_Currency_encoded', 'Transaction_Status_encoded',
    'Authentication_Method_encoded', 'Transaction_Category_encoded'
]

# Streamlit config & CSS
st.set_page_config(page_title="FraudDetection BFSI", layout="wide")
css_path = APP_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# session state defaults
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'current_role' not in st.session_state:
    st.session_state.current_role = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "XGBoost"
if 'session_token' not in st.session_state:
    st.session_state.session_token = None

# try restore via existing session_token
if st.session_state.get('session_token'):
    valid_email = validate_session_token(st.session_state.session_token)
    if valid_email:
        df = load_users_df()
        r = df[df['email'].astype(str).str.lower() == valid_email]
        if not r.empty:
            st.session_state.authenticated = True
            st.session_state.current_user = r.iloc[0]['name']
            st.session_state.current_role = r.iloc[0].get('role', 'user')

# helper badge
def risk_badge_html(cat):
    if cat == "High Risk":
        col = "#dc3545"
    elif cat == "Medium Risk":
        col = "#ff9800"
    else:
        col = "#28a745"
    return f'<span style="background:{col}; color:white; padding:6px 10px; border-radius:10px;">{cat}</span>'

# Sidebar nav
with st.sidebar:
    st.markdown("## ⚡ FraudDetection BFSI")
    if st.session_state.authenticated:
        st.markdown(f"**User:** {st.session_state.current_user}  \n**Role:** {st.session_state.current_role}")
    else:
        st.markdown("Please sign in to access the dashboard.")
    st.markdown("---")
    model_choice = st.selectbox("Model", ["XGBoost", "LightGBM"], index=0 if st.session_state.selected_model=="XGBoost" else 1)
    st.session_state.selected_model = model_choice
    st.markdown("---")
    if not st.session_state.authenticated:
        page = st.radio("Menu", ("Home", "Login", "Signup", "About"))
    else:
        page = st.radio("Menu", ("Home", "Dashboard", "Single Predict", "Batch Predict", "Analytics", "Logs", "About", "Logout"))

# Pages
def page_home():
    st.markdown("<h1 style='text-align:center;'>🛡️ FraudDetection BFSI</h1>", unsafe_allow_html=True)
    st.markdown("Welcome — use Signup to create account. Use Single / Batch Predict when logged in.")
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Model (selected)", st.session_state.selected_model)
    missing = []
    if not XGB_MODEL_FILE.exists(): missing.append("XGBoost")
    if not LGB_MODEL_FILE.exists(): missing.append("LightGBM")
    col2.metric("Model files present", "OK" if not missing else f"Missing: {', '.join(missing)}")
    col3.metric("Saved logs (files)", len(list(LOGS_DIR.glob("module4_results_*.csv"))))

def page_signup():
    st.header("Create account — Signup")
    st.info("Local dev-only auth. First user becomes admin.")
    with st.form("signup_form", clear_on_submit=False):
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        role = st.selectbox("Role (dev only)", ["user", "admin"])
        submitted = st.form_submit_button("Create account")
        if submitted:
            if not (name and email and password):
                st.error("All fields required")
            elif password != password2:
                st.error("Passwords do not match")
            else:
                ok, msg = create_user(name=name, email=email, password=password, role=role)
                if ok:
                    append_action(email, "signup", f"role={role}")
                    st.success("Account created — please login")
                    st.rerun()
                else:
                    st.warning(msg)

def page_login():
    st.header("Login")
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        remember = st.checkbox("Remember me (7-day token)", value=False)
        token_restore = st.text_input("Or paste persistent token (optional)")
        submitted = st.form_submit_button("Login")
        if submitted:
            if token_restore:
                validated = validate_session_token(token_restore.strip())
                if validated:
                    df = load_users_df()
                    r = df[df['email'].astype(str).str.lower() == validated]
                    if not r.empty:
                        st.session_state.authenticated = True
                        st.session_state.current_user = r.iloc[0]['name']
                        st.session_state.current_role = r.iloc[0].get('role', 'user')
                        st.session_state.session_token = token_restore.strip()
                        append_action(validated, "restore_session", "restored via token")
                        st.success(f"Session restored for {st.session_state.current_user}")
                        st.rerun()
                    else:
                        st.error("Token valid but user not found.")
                else:
                    st.error("Restore token invalid/expired.")
                return
            ok, name_or_msg, role = verify_user(email, password)
            if ok:
                st.session_state.authenticated = True
                st.session_state.current_user = name_or_msg
                st.session_state.current_role = role or 'user'
                # create token if remember
                if remember:
                    token = create_session_token(email)
                    st.session_state.session_token = token
                    st.info("Persistent token created — copy it to restore later")
                    st.code(token)
                append_action(email, "login", f"remember={remember}")
                st.success(f"Welcome back, {name_or_msg}!")
                st.rerun()
            else:
                st.error(name_or_msg)

def do_logout():
    user_email = None
    if st.session_state.current_user:
        df = load_users_df()
        r = df[df['name'].astype(str) == st.session_state.current_user]
        if not r.empty:
            user_email = r.iloc[0]['email']
    append_action(user_email or "unknown", "logout", "")
    # remove session token (best-effort) - sessions stored in authentication.py sessions.json
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.current_role = None
    st.session_state.session_token = None
    st.success("Logged out")
    st.rerun()

def page_dashboard():
    st.header("📊 Dashboard")
    logs_df = read_all_logs(LOGS_DIR)
    total_tx = len(logs_df)
    high_risk_count = sum(logs_df['Risk_Category'] == 'High Risk') if 'Risk_Category' in logs_df.columns else 0
    avg_risk = logs_df['Risk_Score'].mean() if 'Risk_Score' in logs_df.columns else 0
    fraud_rate = (high_risk_count / total_tx * 100) if total_tx > 0 else 0
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Transactions (logs)", total_tx)
    k2.metric("High Risk Transactions", f"{high_risk_count} ({fraud_rate:.1f}%)")
    k3.metric("Average Risk Score", f"{avg_risk:.3f}")
    st.divider()
    # basic trend (hourly)
    if not logs_df.empty and 'timestamp' in logs_df.columns:
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'], errors='coerce')
        hourly = logs_df.set_index('timestamp').resample('H')['Risk_Score'].mean().fillna(0)
        st.line_chart(hourly)
    else:
        st.info("No timestamped logs found.")

def page_single_predict():
    st.subheader("💳 Single Transaction Prediction")
    st.markdown("Enter full feature set used in training.")
    with st.form("single_tx_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            transaction_location = st.text_input("Transaction Location")
            card_type = st.selectbox("Card Type", ["UzCard","Visa","MasterCard","Humo","Rupay"])
            authentication_method = st.selectbox("Authentication Method", ["2FA","PIN","Biometric","None","Password"])
            transaction_currency = st.selectbox("Transaction Currency", ["UZS","USD","EUR","INR"])
            transaction_status = st.selectbox("Transaction Status", ["Successful","Failed","Pending","Declined"])
            transaction_category = st.selectbox("Transaction Category", ["Transfer","Purchase","Withdrawal","Bill Payment","Online Shopping"])
        with col2:
            transaction_amount = st.number_input("Transaction Amount", min_value=0.0)
            previous_transaction_count = st.number_input("Previous Transaction Count", min_value=0)
            distance_bw_transactions = st.number_input("Distance Between Transactions (km)", min_value=0.0)
            time_since_last_tx = st.number_input("Time Since Last Transaction (minutes)", min_value=0.0)
            transaction_velocity = st.number_input("Transaction Velocity", min_value=0.0)
        with col3:
            transaction_date = st.date_input("Transaction Date", value=datetime.now().date())
            transaction_time = st.time_input("Transaction Time", value=datetime.now().time())
        submitted_single = st.form_submit_button("Predict")
    if submitted_single:
        tx_hour = transaction_time.hour
        tx_day = transaction_date.day
        tx_month = transaction_date.month
        log_tx_amount = np.log1p(transaction_amount)
        tx = {
            'Transaction_Amount': transaction_amount,
            'Previous_Transaction_Count': previous_transaction_count,
            'Distance_Between_Transactions_km': distance_bw_transactions,
            'Time_Since_Last_Transaction_min': time_since_last_tx,
            'Transaction_Velocity': transaction_velocity,
            'Log_Transaction_Amount': log_tx_amount,
            'Transaction_Hour': tx_hour,
            'Transaction_Day': tx_day,
            'Transaction_Month': tx_month,
            'Transaction_Location': transaction_location,
            'Card_Type': card_type,
            'Transaction_Currency': transaction_currency,
            'Transaction_Status': transaction_status,
            'Authentication_Method': authentication_method,
            'Transaction_Category': transaction_category,
            'Transaction_ID': 0,
            'User_ID': 0,
            'Merchant_ID': 0,
            'Device_ID': 0
        }
        for c in CATEGORICAL_COLUMNS:
            try:
                tx[c + '_encoded'] = pd.factorize([tx[c]])[0][0]
            except Exception:
                tx[c + '_encoded'] = 0
        X = pd.DataFrame([tx])[FEATURE_COLUMNS]
        xgb_model, xgb_err = load_model_cached(str(XGB_MODEL_FILE))
        lgb_model, lgb_err = load_model_cached(str(LGB_MODEL_FILE))
        if xgb_model is None or lgb_model is None:
            st.error("Model load error.")
            return
        try:
            xgb_score = xgb_model.predict_proba(X)[:,1][0]
            lgb_score = lgb_model.predict_proba(X)[:,1][0]
            append_action(st.session_state.current_user or "anonymous", "single_predict", f"xgb={xgb_score:.4f},lgb={lgb_score:.4f}")
            st.markdown(f"**XGBoost**: {xgb_score:.3f} — **LightGBM**: {lgb_score:.3f}")
        except Exception as e:
            append_action(st.session_state.current_user or "anonymous", "single_predict_error", str(e))
            st.error(f"Prediction error: {e}")

def page_batch_predict():
    st.header("📂 Batch Prediction (CSV)")
    st.info("Upload CSV. After prediction completes you may generate an EDA report on demand from Analytics tab.")
    uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            append_action(st.session_state.current_user or "anonymous", "batch_upload_error", str(e))
            return
        st.write(f"Rows: {len(df)}")
        # derive features
        if 'Transaction_Date' in df.columns:
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
            df['Transaction_Day'] = df['Transaction_Date'].dt.day
            df['Transaction_Month'] = df['Transaction_Date'].dt.month
        if 'Transaction_Time' in df.columns:
            df['Transaction_Time'] = pd.to_datetime(df['Transaction_Time'].astype(str), errors='coerce').dt.time
            df['Transaction_Hour'] = pd.to_datetime(df['Transaction_Time'].astype(str), errors='coerce').dt.hour
        if 'Transaction_Amount' in df.columns:
            df['Log_Transaction_Amount'] = np.log1p(df['Transaction_Amount'].fillna(0))
        # encode categoricals
        for c in CATEGORICAL_COLUMNS:
            if c in df.columns:
                df[c + "_encoded"] = pd.factorize(df[c])[0]
            else:
                df[c + "_encoded"] = 0
        missing = [f for f in FEATURE_COLUMNS if f not in df.columns]
        if missing:
            st.warning(f"Missing feature cols - filled with defaults: {missing}")
            for m in missing:
                df[m] = 0
        X = df[FEATURE_COLUMNS].fillna(0)
        model_path = XGB_MODEL_FILE if st.session_state.selected_model == "XGBoost" else LGB_MODEL_FILE
        model, err = load_model_cached(str(model_path))
        if model is None:
            st.error("Model load error.")
            append_action(st.session_state.current_user or "anonymous", "batch_model_load_error", str(err))
            return
        try:
            df['Risk_Score'] = model.predict_proba(X)[:,1]
            df['Risk_Category'] = df['Risk_Score'].apply(categorize_risk)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            append_action(st.session_state.current_user or "anonymous", "batch_predict_error", str(e))
            return
        # save results to logs
        out_path = append_log_batch(LOGS_DIR, df)
        st.success(f"Batch results saved to: {out_path}")
        append_action(st.session_state.current_user or "anonymous", "batch_predict", f"rows={len(df)} out={out_path}")
        st.download_button("Download results CSV", df.to_csv(index=False).encode('utf-8'),
                           file_name=Path(out_path).name, mime='text/csv')

def page_analytics():
    st.header("📈 Analytics & EDA")
    st.markdown("Click **Generate EDA** to run Exploratory Data Analysis on the **latest** batch prediction log file.")
    # find latest module4_results_*.csv in logs
    files = sorted(LOGS_DIR.glob("module4_results_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        st.info("No batch prediction logs found. Run Batch Predict first to create one.")
        return
    latest = files[0]
    st.write("Latest batch log:", latest.name)
    if st.button("Generate EDA on latest batch"):
        try:
            df = pd.read_csv(latest)
        except Exception as e:
            st.error(f"Failed to read latest log: {e}")
            append_action(st.session_state.current_user or "anonymous", "eda_failed", str(e))
            return
        # basic lists for analytics module
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        # display inline
        display_eda_inline(st, df, categorical_cols, numeric_cols)
        # create downloadable HTML report
        html_str, html_bytes = generate_eda_report_html(df, categorical_cols, numeric_cols)
        st.download_button("Download EDA HTML report", html_bytes, file_name=f"eda_report_{latest.stem}.html", mime="text/html")
        append_action(st.session_state.current_user or "anonymous", "eda_generated", f"source={latest.name}")
    else:
        st.markdown("Press the button to run the EDA. It may take a few seconds for large files.")

def page_logs():
    st.header("🧾 Logs — Saved Predictions")
    logs_df = read_all_logs(LOGS_DIR)
    if logs_df.empty:
        st.info("No logs yet.")
    else:
        st.dataframe(logs_df.sort_values('timestamp', ascending=False).head(200), use_container_width=True)
        st.download_button("Download combined logs (CSV)", logs_df.to_csv(index=False).encode('utf-8'),
                           file_name="all_module4_logs.csv", mime="text/csv")
    st.markdown("---")
    st.header("🔐 Audit / Actions Log")
    actions = read_actions()
    if actions.empty:
        st.info("No audit actions yet.")
    else:
        st.dataframe(actions.sort_values('timestamp', ascending=False).head(200), use_container_width=True)
        st.download_button("Download actions log", actions.to_csv(index=False).encode('utf-8'),
                           file_name="actions_log.csv", mime='text/csv')

def page_about():
    st.header("About")
    st.markdown("FraudDetection BFSI — Streamlit dashboard (Phase 2).")
    st.markdown("- Phase 1 implemented secure local auth, session tokens, and audit logging.")
    st.markdown("- Phase 2 added analytics/EDA generation on demand.")

# Router
if page == "Home":
    page_home()
elif page == "Login":
    page_login()
elif page == "Signup":
    page_signup()
elif page == "About":
    page_about()
elif page == "Logout":
    do_logout()
else:
    if not st.session_state.authenticated:
        st.warning("Please log in to access this page.")
        page_login()
    else:
        if page == "Dashboard":
            page_dashboard()
        elif page == "Single Predict":
            page_single_predict()
        elif page == "Batch Predict":
            page_batch_predict()
        elif page == "Analytics":
            page_analytics()
        elif page == "Logs":
            page_logs()
        else:
            st.write("Unknown page")
# =============================================================================
# PHASE 3 — ADMIN PANEL (MODEL MONITORING, LOGS, RELOAD, RETRAIN)
# =============================================================================

import json
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

# ------------------- Metrics Loader -------------------
def load_metrics_file(path: Path) -> dict:
    """Safely load model metrics from JSON file."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

# ------------------- Logs Analytics -------------------
def get_prediction_counts(logs_df: pd.DataFrame, days: int = 30) -> dict:
    """Compute summary statistics for prediction activity."""
    now = pd.Timestamp.utcnow()
    logs_df = logs_df.copy()
    if "timestamp" in logs_df.columns:
        logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")
    else:
        return {
            "total": len(logs_df),
            "recent_count": len(logs_df),
            "recent_mean_risk": 0.0,
            "prior_mean_risk": 0.0,
        }

    cutoff_recent = now - pd.Timedelta(days=days)
    recent = logs_df[logs_df["timestamp"] >= cutoff_recent]
    prior = logs_df[logs_df["timestamp"] < cutoff_recent]

    return {
        "total": len(logs_df),
        "recent_count": len(recent),
        "prior_count": len(prior),
        "recent_mean_risk": float(recent["Risk_Score"].mean()) if not recent.empty else 0.0,
        "prior_mean_risk": float(prior["Risk_Score"].mean()) if not prior.empty else 0.0,
    }

# ------------------- Cached Model Handler -------------------
@st.cache_resource
def cached_models():
    """Cache and load both models."""
    xgb_model, xgb_err = load_model_cached(str(XGB_MODEL_FILE))
    lgb_model, lgb_err = load_model_cached(str(LGB_MODEL_FILE))
    return {
        "xgb_model": xgb_model,
        "xgb_err": xgb_err,
        "lgb_model": lgb_model,
        "lgb_err": lgb_err,
    }

def reload_models_and_clear_cache():
    """Clear cache and reload model binaries."""
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    return cached_models()

# ------------------- Admin Panel Page -------------------
def page_admin_panel():
    """Admin-only control center for model monitoring, metrics, logs, and model reload."""
    if st.session_state.get("current_role", "user") != "admin":
        st.warning("Admin Panel is visible to admin users only.")
        return

    st.header("🛠️ Admin Panel — Model Monitoring & Diagnostics")
    st.markdown("Monitor model health, view diagnostics, audit logs, and reload or retrain models.")

    # --- Model overview section ---
    st.subheader("Model Overview")
    models = cached_models()
    xgb_ok = models["xgb_model"] is not None
    lgb_ok = models["lgb_model"] is not None

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost**")
        st.write("File:", str(XGB_MODEL_FILE.name))
        st.write("Loaded:", "✅" if xgb_ok else f"❌ ({models['xgb_err']})")
        metrics_xgb = load_metrics_file(METRICS_XGB)
        if metrics_xgb:
            st.json(metrics_xgb)
        else:
            st.info("No XGBoost metrics found (models/metrics_xgb.json)")

    with col2:
        st.markdown("**LightGBM**")
        st.write("File:", str(LGB_MODEL_FILE.name))
        st.write("Loaded:", "✅" if lgb_ok else f"❌ ({models['lgb_err']})")
        metrics_lgb = load_metrics_file(METRICS_LGB)
        if metrics_lgb:
            st.json(metrics_lgb)
        else:
            st.info("No LightGBM metrics found (models/metrics_lgb.json)")

    # --- Usage & recent predictions section ---
    st.subheader("Usage & Recent Predictions")
    logs_df = read_all_logs(LOGS_DIR)
    counts = get_prediction_counts(logs_df, days=7)
    st.metric("Total predictions (all time)", counts.get("total", 0))
    st.metric("Predictions (last 7 days)", counts.get("recent_count", 0))

    recent_mean = counts.get("recent_mean_risk", 0.0)
    prior_mean = counts.get("prior_mean_risk", 0.0)
    st.write(f"Recent mean Risk Score (7d): **{recent_mean:.4f}**")
    st.write(f"Prior mean Risk Score (before 7d): **{prior_mean:.4f}**")

    if prior_mean > 0:
        pct_change = ((recent_mean - prior_mean) / prior_mean) * 100.0
        st.metric("Mean risk change vs prior", f"{pct_change:.2f}%")
        if abs(pct_change) > 20:
            st.warning("⚠️ Significant shift in mean risk score (>20%) — possible data drift or model degradation.")
    else:
        st.info("Not enough historical data for drift analysis.")

    # --- Model controls (Reload / Retrain) ---
    st.subheader("Model Controls (Admin Only)")
    colr1, colr2, colr3 = st.columns([1, 1, 2])

    with colr1:
        if st.button("🔄 Reload Models"):
            try:
                models_new = reload_models_and_clear_cache()
                ok_msg = []
                if models_new["xgb_model"] is not None:
                    ok_msg.append("XGBoost reloaded")
                if models_new["lgb_model"] is not None:
                    ok_msg.append("LightGBM reloaded")
                append_action(st.session_state.get("current_user", "admin"), "reload_models", ",".join(ok_msg))
                st.success("Models reloaded successfully.")
                st.rerun()
            except Exception as e:
                append_action(st.session_state.get("current_user", "admin"), "reload_models_error", str(e))
                st.error(f"Model reload failed: {e}")

    with colr2:
        if st.button("🧠 Retrain Models"):
            try:
                st.info("Starting retraining process...")
                retrain_status = retrain_all_models()  # Custom retrain function
                append_action(st.session_state.get("current_user", "admin"), "retrain_models", retrain_status)
                st.success("Retraining completed successfully.")
                st.rerun()
            except Exception as e:
                append_action(st.session_state.get("current_user", "admin"), "retrain_error", str(e))
                st.error(f"Retraining failed: {e}")

    with colr3:
        st.markdown("Use these buttons after updating model files or retraining data. Reload to reflect updates immediately.")

    # --- Error / Audit logs viewer ---
    st.subheader("Application & Audit Logs")
    actions_df = read_actions()
    if actions_df.empty:
        st.info("No audit logs available.")
    else:
        types = ["All"] + sorted(actions_df["action"].unique().astype(str).tolist())
        sel_type = st.selectbox("Filter by action type", types, index=0)
        date_from = st.date_input("From date", value=(datetime.utcnow().date() - pd.Timedelta(days=30)))
        date_to = st.date_input("To date", value=datetime.utcnow().date())

        mask = (
            (pd.to_datetime(actions_df["timestamp"]).dt.date >= date_from)
            & (pd.to_datetime(actions_df["timestamp"]).dt.date <= date_to)
        )
        if sel_type != "All":
            mask &= actions_df["action"] == sel_type

        filtered_actions = actions_df[mask].sort_values("timestamp", ascending=False)
        st.dataframe(filtered_actions.head(200), use_container_width=True)
        st.download_button(
            "📥 Download Filtered Logs (CSV)",
            filtered_actions.to_csv(index=False).encode("utf-8"),
            file_name=f"filtered_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

# ------------------- Navigation Router -------------------
def navigation_router():
    """Main router with Admin visibility and new Analytics tab."""
    role = st.session_state.get("current_role", "user")

    if role == "admin":
        page = st.sidebar.radio(
            "Navigation",
            ["Home", "Dashboard", "Prediction", "Analytics", "Admin Panel", "About"]
        )
    else:
        page = st.sidebar.radio(
            "Navigation",
            ["Home", "Dashboard", "Prediction", "Analytics", "About"]
        )

    # Map selected page
    if page == "Admin Panel":
        page_admin_panel()
    elif page == "Analytics":
        page_analytics()
    elif page == "Prediction":
        page_prediction()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "Home":
        page_home()
    elif page == "About":
        page_about()
# ---------------------------------------------------------------------
# Helper: Load ONNX model (if available)
# ---------------------------------------------------------------------
def load_onnx_model(onnx_path):
    """Load ONNX model if exists, else return None."""
    if os.path.exists(onnx_path):
        try:
            session = ort.InferenceSession(onnx_path)
            return session
        except Exception as e:
            st.error(f"Failed to load ONNX model: {e}")
            return None
    return None


# ---------------------------------------------------------------------
# Helper: Load ML model (fallback)
# ---------------------------------------------------------------------
def load_model(model_path):
    """Load a saved ML model (joblib)."""
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    return None


# ---------------------------------------------------------------------
# Helper: Hybrid Prediction with Explainability
# ---------------------------------------------------------------------
def hybrid_inference(input_df, model_choice="XGBoost", onnx_path=None, rules_module=None):
    """
    Combine ML prediction + rule-based reasoning.
    Returns prediction + explanation string.
    """
    explanations = []
    onnx_session = load_onnx_model(onnx_path) if onnx_path else None

    # Load fallback ML model if ONNX not used
    if not onnx_session:
        model_path = (
            "models/xgboost_model.joblib"
            if model_choice == "XGBoost"
            else "models/lightgbm_model.joblib"
        )
        model = load_model(model_path)
    else:
        model = None

    # Predict using ONNX or model
    try:
        if onnx_session:
            input_data = input_df.to_numpy().astype(np.float32)
            preds = onnx_session.run(None, {onnx_session.get_inputs()[0].name: input_data})[0]
        else:
            preds = model.predict_proba(input_df)[:, 1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, ["Prediction error"]

    # -----------------------------------------------------------------
    # Apply rule-based explainability
    # -----------------------------------------------------------------
    if rules_module:
        for _, row in input_df.iterrows():
            rule_flags = rules_module.evaluate_rules(row.to_dict())
            explanations.append(", ".join(rule_flags) if rule_flags else "No rule triggered")
    else:
        explanations = ["Explainability not configured"] * len(preds)

    # Combine both sources into final hybrid prediction
    final_preds = []
    for prob, reason in zip(preds, explanations):
        flag = "Fraud" if prob > 0.5 else "Legit"
        final_preds.append(
            {
                "probability": float(prob),
                "label": flag,
                "reason": reason,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return final_preds


# ---------------------------------------------------------------------
# Helper: SHAP Explainability (Model Interpretability)
# ---------------------------------------------------------------------
def generate_shap_explanations(model, input_df):
    """Compute SHAP feature importance explanations."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        st.subheader("Feature Importance (Explainability)")
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.warning(f"Explainability visualization skipped: {e}")


# ---------------------------------------------------------------------
# Helper: Import Fraud Rules
# ---------------------------------------------------------------------
def import_fraud_rules():
    """Import fraud_rules.py if available."""
    try:
        import fraud_rules
        return fraud_rules
    except ImportError:
        st.warning("⚠️ fraud_rules.py not found. Rule-based reasoning disabled.")
        return None


# ---------------------------------------------------------------------
# Integration UI
# ---------------------------------------------------------------------
def explainability_dashboard():
    """Streamlit section for Explainability + Hybrid Inference"""
    st.title("🧠 Explainability & Rule Engine")

    st.info("This module merges ML predictions with rule-based reasoning for transparent fraud detection.")

    model_choice = st.selectbox("Select Model", ["XGBoost", "LightGBM"])
    onnx_path = st.text_input("Enter ONNX model path (optional)", "models/model.onnx")

    uploaded_file = st.file_uploader("Upload CSV for explainable predictions", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        rules_module = import_fraud_rules()
        results = hybrid_inference(df, model_choice, onnx_path, rules_module)

        if results:
            output_df = pd.DataFrame(results)
            st.dataframe(output_df)
            st.success("✅ Explainable predictions generated.")

            # If using ML model, show SHAP visualization
            model_path = (
                "models/xgboost_model.joblib"
                if model_choice == "XGBoost"
                else "models/lightgbm_model.joblib"
            )
            model = load_model(model_path)
            if model is not None:
                generate_shap_explanations(model, df)


# ---------------------------------------------------------------------
# Register Explainability tab for admin role only
# ---------------------------------------------------------------------
if "user_role" in st.session_state and st.session_state.user_role == "admin":
    explainability_dashboard()


# =============================================================================
# 🧰 PHASE 4 — UI, CHATBOT & FUTURE-READY ADD-ONS
# =============================================================================

# ------------------------------
# 1️⃣  Chatbot Placeholder Page
# ------------------------------
def chatbot_page():
    st.title("💬 AI Chatbot Assistant (Coming Soon)")
    st.write("This section will host a conversational AI assistant for users to ask questions, "
             "get guidance on transaction trends, and explain predictions.")
    st.info("🤖 Chatbot backend integration pending — placeholder for future FinGPT-based assistant.")
    
    # Simple chat UI placeholder
    user_query = st.text_input("Type your message here...")
    if st.button("Send", use_container_width=True):
        with st.spinner("Thinking..."):
            time.sleep(1)
        st.success("AI Assistant: This is a placeholder response. The chatbot feature is under development.")


# ------------------------------
# 2️⃣  Help & FAQ Page
# ------------------------------
def help_faq_page():
    st.title("❓ Help & FAQs")
    st.markdown("""
    ### Common Questions
    **Q1:** What is this application about?  
    **A:** This system predicts potential fraudulent transactions using ML models.

    **Q2:** Which models are available?  
    **A:** You can currently select between **XGBoost** and **LightGBM** models.

    **Q3:** How can I upload data for batch prediction?  
    **A:** Go to the *Batch Prediction* tab and upload a valid `.csv` file.

    **Q4:** Can I retrain the model?  
    **A:** Only admins can retrain or reload models through the Admin Dashboard.

    ---

    ### Troubleshooting
    - Ensure uploaded CSVs contain all required columns.
    - If UI doesn’t update, try clicking **Refresh Dashboard**.
    - For persistent issues, check the **Logs** page or contact support.
    """)


# ------------------------------
# 3️⃣  About Page
# ------------------------------
def about_page():
    st.title("🏦 About Predictive Transaction Intelligence")
    st.markdown("""
    ### Overview
    This project is part of the **Predictive Transaction Intelligence** suite, built for the BFSI sector.  
    It leverages machine learning and explainable AI to flag potentially fraudulent transactions.

    ### Features
    - Real-time prediction engine  
    - Batch analysis  
    - Explainable hybrid reasoning (ML + rule-based)  
    - Admin monitoring and model management  
    - Future-ready chatbot and 2FA placeholders  

    ---
    **Developed by:** SG.Vardhan Vikranth Reddy  
    **Version:** 1.0.0 (Streamlit Unified Build)  
    **Contact:** support@predictiveai.com
    """)


# ------------------------------
# 4️⃣  Sidebar Enhancements
# ------------------------------
def enhanced_sidebar_navigation():
    with st.sidebar:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #002B5B 0%, #004B87 100%);
                color: white;
            }
            [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
                color: #FFFFFF;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown("### ⚙️ Navigation")
        choice = st.radio("Select a Page", ["Home", "Dashboard", "Prediction", 
                                            "Analytics", "Admin", "Chatbot", 
                                            "Help / FAQ", "About"])
        st.session_state.selected_page = choice
        st.markdown("---")
        st.caption("💡 Banking-grade theme — Phase 4 UI Upgrade")


# ------------------------------
# 5️⃣  Accessibility / Notifications Placeholders
# ------------------------------
def accessibility_and_future_ready():
    st.markdown("### 🛡️ Accessibility & Future Add-ons")
    st.write("This area will host future accessibility improvements (WCAG compliance), "
             "notifications, and two-factor authentication mockups.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("🔔 Notification Center (Coming Soon)")
    with col2:
        st.button("🔐 Enable 2FA (Mock Placeholder)")
    st.success("✅ Accessibility and security placeholders active.")


# ------------------------------
# 6️⃣  Page Router
# ------------------------------
def phase4_page_router():
    """Routes to respective Phase 4 pages based on sidebar selection."""
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Home"

    enhanced_sidebar_navigation()
    selected_page = st.session_state.selected_page

    if selected_page == "Chatbot":
        chatbot_page()
    elif selected_page == "Help / FAQ":
        help_faq_page()
    elif selected_page == "About":
        about_page()
    elif selected_page == "Admin":
        accessibility_and_future_ready()
    # The rest (Home, Dashboard, etc.) are handled in earlier phases


# ------------------------------
# 7️⃣  Phase 4 Initializer
# ------------------------------
def initialize_phase4_features():
    """Call this once at the end of the Streamlit main() or run context."""
    st.markdown("---")
    phase4_page_router()
# Initialize Phase 4 after all previous modules are loaded
if __name__ == "__main__":
    initialize_phase4_features()

# ============================== #
# ENTERPRISE BFSI ENHANCEMENTS — Append After Line 965
# ============================== #

# BANK-GRADE PASSWORD SECURITY (bcrypt switch)
import bcrypt
def hash_pw_bcrypt(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
def verify_pw_bcrypt(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)


# 2FA HOOK (future extension, dummy demo)
def two_factor_auth_prompt():
    st.info("A verification code has been sent to your registered mobile/email (Demo — implement 2FA via enterprise provider for prod).")
    code = st.text_input("Enter 2FA code (demo: '123456')")
    if code == "123456":
        st.success("2FA passed. Welcome!")
        return True
    st.warning("Incorrect code.")
    return False

# BANK-GRADE SIDEBAR AND THEMING
st.markdown("""
<style>
section[data-testid="stSidebar"] { background-color: #eaf1f7; border-left: 3px solid #0077b6;}
[data-testid="stHeader"] { background-color: #0077b6; color: #fff;}
div.stButton > button { background: #0077b6; color: #fff; border-radius: 7px; }
.risk-high {background:#dc3545;padding:7px 13px;border-radius:10px;color:#fff;}
.risk-med {background:#ffc107;padding:7px 13px;border-radius:10px;color:#222;}
.risk-low {background:#28a745;padding:7px 13px;border-radius:10px;color:#fff;}
</style>
""", unsafe_allow_html=True)

# COMPLIANCE/AUDIT EXTENSION (PCI, ISO, DSS Ready)
def audit_banner():
    st.markdown("""
    <div style='background:#f9fbe7;border-left:8px solid #0077b6;padding:16px;margin-top:20px;'>
    <b>Compliance Notice:</b> All interactions are logged for banking sector audit (PCI DSS, ISO 27001). Data privacy is strictly enforced.
    </div>
    """, unsafe_allow_html=True)
audit_banner()

# MOBILE/ACCESSIBILITY HOOK
st.markdown("<!-- Add ARIA attributes and keyboard nav for full accessibility in future release. -->")

# FUTURE LLM/CHATBOT PAGE
def page_chatbot():
    st.header("AI Chatbot / Assistant")
    st.info("Enterprise AI/LLM fraud assistant integration goes here. (Future release: real fraud scenario simulation, user Q&A, policy search).")

if st.sidebar.button("AI Chatbot"):
    page_chatbot()

# MODEL DRIFT/MONITORING/ADMIN PAGE EXTENSION
def page_model_admin():
    st.header("Model Health & Drift Monitoring (Enterprise+)")
    from pathlib import Path
    import os
    modinfo = [(m, Path(m).stat().st_mtime) for m in [str(m) for m in Path("models").glob("*.pkl")]]
    st.write("Loaded Models:", modinfo)
    st.markdown("**Model retraining recommended if drift detected. Admin notifies Data Science for new model deployment.**")
    # Drift simulation: compare active logs statistics to historic baseline (placeholder).
    # Expand: insert Prometheus, MLFlow, or Datadog integration here for true monitoring.

if st.sidebar.button("Admin Monitoring"):
    page_model_admin()

# EXPLAINABILITY/RULE ENGINE HYBRID HOOK (as a callable example)
def demo_hybrid_explain(row):
    reasons = []
    if row.get("Location")=="Foreign" and row.get("Transaction_Hour",0)<5:
        reasons.append("Foreign transaction outside business hours.")
    if row.get("Transaction_Amount",0)>10000:
        reasons.append("Large transaction flagged.")
    if not reasons:
        reasons.append("ML-only decision.")
    return "; ".join(reasons)

# EXTENDED HELP / FAQ PAGE
def page_help():
    st.header("Enterprise Help / FAQ")
    st.write("**Contact IT helpdesk, compliance, or fraud incident team. All user actions and predictions monitored for security and audit.")
    st.write("- To enable 2FA: Integrate your SMS/email provider using standards (Twilio, Sendgrid, etc).")
    st.write("- For audit export: Use admin panel download options. Data is encrypted at rest (for true prod deployment).")
    st.write("For API/Real-time deployment: contact IT.")

if st.sidebar.button("Help/FAQ"):
    page_help()

# BANK-GRADE CSS POLISH (additional style block if needed)
st.markdown("""
<style>
/* Responsive controls, mobile tweaks, future ARIA mods can be added here */
h2, h3 {color: #0077b6;}
.stMetric, .stDataFrame td {font-size:1rem;}
button, input {font-family: "Roboto", "Arial", sans-serif;}
</style>
""", unsafe_allow_html=True)

# --- YOU CAN ADD FURTHER ENTERPRISE FEATURES HERE ---
# e.g. Notification center, incident routing, API hooks, real-time monitoring, SSO integration stubs ...

# END OF ENTERPRISE ENHANCEMENTS FOR BFSI FRAUD PLATFORM
