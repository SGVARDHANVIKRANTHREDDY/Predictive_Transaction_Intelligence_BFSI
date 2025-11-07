"""
audits.py - Audit logging utilities for FraudDetection BFSI

- append_action: append a row to actions_log.csv
- read_actions: return DataFrame of actions
"""

from pathlib import Path
import pandas as pd
from datetime import datetime

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "data" / "logs"
ACTIONS_LOG = LOGS_DIR / "actions_log.csv"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def append_action(user_email: str, action: str, details: str = ""):
    """Append action row to actions_log.csv."""
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'user': user_email or 'unknown',
        'action': action,
        'details': details
    }
    df_row = pd.DataFrame([row])
    if ACTIONS_LOG.exists():
        df_old = pd.read_csv(ACTIONS_LOG)
        df_out = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_out = df_row
    df_out.to_csv(ACTIONS_LOG, index=False)

def read_actions() -> pd.DataFrame:
    if not ACTIONS_LOG.exists():
        return pd.DataFrame(columns=['timestamp', 'user', 'action', 'details'])
    return pd.read_csv(ACTIONS_LOG)
