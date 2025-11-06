"""
analytics.py - EDA utilities for FraudDetection BFSI

- generate_eda_report_html: produce inline plots and return an HTML report (as bytes)
- display_eda_inline: show EDA in Streamlit (matplotlib + st.pyplot)
"""

import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    return b64

def generate_eda_report_html(df: pd.DataFrame, categorical_cols: list, numeric_cols: list) -> Tuple[str, bytes]:
    """Generate plots and embed them into an HTML EDA report."""
    parts = [
        "<h1>FraudDetection BFSI — EDA Report</h1>",
        f"<p>Generated: {pd.Timestamp.utcnow().isoformat()} UTC</p>",
        "<h2>Basic summary</h2>",
        f"<pre>{df.describe(include='all').to_html()}</pre>"
    ]
    if 'Risk_Category' in df.columns:
        counts = df['Risk_Category'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        counts.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        ax.set_title('Risk Category Distribution')
        parts.append("<h2>Risk Category Distribution</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    for c in categorical_cols:
        if c in df.columns:
            vc = df[c].value_counts().head(20)
            fig, ax = plt.subplots(figsize=(6, 3))
            vc.plot.bar(ax=ax)
            ax.set_title(f'{c} (top categories)')
            ax.set_ylabel('count')
            parts.append(f"<h3>{c}</h3>")
            parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    parts.append("<h2>Numerical distributions</h2>")
    for n in numeric_cols:
        if n in df.columns:
            fig, ax = plt.subplots(figsize=(6, 3))
            df[n].dropna().astype(float).hist(bins=40, ax=ax)
            ax.set_title(n)
            parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if not num_df.empty and num_df.shape[1] > 1:
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        fig.colorbar(cax)
        ax.set_title('Correlation matrix')
        parts.append("<h2>Correlation matrix</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    html = "<html><body>" + "\n".join(parts) + "</body></html>"
    return html, html.encode('utf-8')

def display_eda_inline(st, df: pd.DataFrame, categorical_cols: list, numeric_cols: list):
    """Display EDA plots inline in Streamlit using matplotlib & st.pyplot."""
    st.markdown("## EDA — Overview")
    if 'Risk_Category' in df.columns:
        st.markdown("### Risk Category Distribution")
        counts = df['Risk_Category'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        counts.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
    st.markdown("### Categorical distributions (sample)")
    shown = 0
    for c in categorical_cols:
        if c in df.columns and shown < 3:
            st.markdown(f"**{c}**")
            fig, ax = plt.subplots(figsize=(6, 3))
            df[c].value_counts().head(20).plot.bar(ax=ax)
            st.pyplot(fig)
            shown += 1
    st.markdown("### Numerical distributions (sample)")
    shown = 0
    for n in numeric_cols:
        if n in df.columns and shown < 3:
            st.markdown(f"**{n}**")
            fig, ax = plt.subplots(figsize=(6, 3))
            df[n].dropna().astype(float).hist(bins=40, ax=ax)
            st.pyplot(fig)
            shown += 1
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if not num_df.empty and num_df.shape[1] > 1:
        st.markdown("### Correlation heatmap")
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        fig.colorbar(cax)
        st.pyplot(fig)
