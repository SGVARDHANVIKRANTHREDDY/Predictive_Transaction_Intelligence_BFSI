"""
authentication.py - Authentication utilities for FraudDetection BFSI

- PBKDF2 password hashing
- User creation & verification (CSV)
- Session token creation & validation (sessions.json)
"""

import json
import secrets
import hashlib
import html
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional

import pandas as pd

# Directory configuration (auto-detects project root)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
USERS_CSV = DATA_DIR / "users.csv"
SESSIONS_JSON = DATA_DIR / "sessions.json"

PBKDF2_ITERATIONS = 200_000
SALT_BYTES = 16
SESSION_EXPIRY_DAYS = 7

# Ensure required dirs/files
DATA_DIR.mkdir(parents=True, exist_ok=True)
if not USERS_CSV.exists():
    pd.DataFrame(columns=['name', 'email', 'salt', 'pw_hash', 'role', 'created_at']).to_csv(USERS_CSV, index=False)
if not SESSIONS_JSON.exists():
    SESSIONS_JSON.write_text(json.dumps({}), encoding='utf-8')

def _sanitize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return html.escape(str(s).strip())

def _hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    if salt is None:
        salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, PBKDF2_ITERATIONS)
    return salt.hex(), dk.hex()

def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, PBKDF2_ITERATIONS)
    return dk.hex() == hash_hex

def load_users_df() -> pd.DataFrame:
    return pd.read_csv(USERS_CSV)

def save_users_df(df: pd.DataFrame):
    df.to_csv(USERS_CSV, index=False)

def create_user(name: str, email: str, password: str, role: str = "user") -> Tuple[bool, str]:
    df = load_users_df()
    email_clean = _sanitize_text(email).lower()
    name_clean = _sanitize_text(name)
    if email_clean in df['email'].astype(str).str.lower().values:
        return False, "Email already registered"
    salt, pw_hash = _hash_password(password)
    created_at = datetime.utcnow().isoformat()
    if df.empty and role == "user":
        role = "admin"
    new = {
        'name': name_clean,
        'email': email_clean,
        'salt': salt,
        'pw_hash': pw_hash,
        'role': role,
        'created_at': created_at
    }
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    save_users_df(df)
    return True, "User created"

def verify_user(email: str, password: str) -> Tuple[bool, str, Optional[str]]:
    df = load_users_df()
    email_clean = _sanitize_text(email).lower()
    row = df[df['email'].astype(str).str.lower() == email_clean]
    if row.empty:
        return False, "Unknown email", None
    row = row.iloc[0]
    if _verify_password(password, row['salt'], row['pw_hash']):
        return True, row['name'], row.get('role', 'user')
    else:
        return False, "Invalid password", None

def _load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_JSON.read_text(encoding='utf-8'))
    except Exception:
        return {}

def _save_sessions(sessions: dict):
    SESSIONS_JSON.write_text(json.dumps(sessions, indent=2), encoding='utf-8')

def create_session_token(email: str, days: int = SESSION_EXPIRY_DAYS) -> str:
    sessions = _load_sessions()
    token = secrets.token_urlsafe(32)
    expiry_ts = (datetime.utcnow() + timedelta(days=days)).timestamp()
    sessions[token] = {
        'email': email.strip().lower(),
        'expiry': expiry_ts,
        'created_at': datetime.utcnow().isoformat()
    }
    _save_sessions(sessions)
    return token

def validate_session_token(token: str) -> Optional[str]:
    if not token:
        return None
    sessions = _load_sessions()
    info = sessions.get(token)
    if not info:
        return None
    if datetime.utcnow().timestamp() > float(info.get('expiry', 0)):
        sessions.pop(token, None)
        _save_sessions(sessions)
        return None
    return info.get('email')
