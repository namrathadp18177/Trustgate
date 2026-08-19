"""
TrustGate Dashboard
----------------------
Streamlit app for monitoring the trust/validation pipeline end to end:
decision breakdown, trust score distribution, per-signal averages, and a
drill-down table with full explanations for every evaluated document.

Run:
    streamlit run dashboard/app.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from db.connection import get_cursor

st.set_page_config(page_title="TrustGate", layout="wide")
st.title("TrustGate — Data Trust & Validation Monitor")


@st.cache_data(ttl=10)
def load_results():
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT v.doc_id, d.source, v.structural_score, v.duplication_score,
                   v.pii_score, v.toxicity_score, v.freshness_score,
                   v.hallucination_score, v.trust_score, v.decision,
                   v.explanation, v.evaluated_at
            FROM validation_results v
            JOIN documents d ON d.doc_id = v.doc_id
            ORDER BY v.evaluated_at DESC
            """
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows)


if st.button("Refresh"):
    st.cache_data.clear()

df = load_results()

if df.empty:
    st.info("No evaluated documents yet. Run ingestion + the pipeline first.")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total documents", len(df))
    col2.metric("PASS", int((df["decision"] == "PASS").sum()))
    col3.metric("REVIEW", int((df["decision"] == "REVIEW").sum()))
    col4.metric("REJECT", int((df["decision"] == "REJECT").sum()))

    st.subheader("Decision breakdown")
    st.bar_chart(df["decision"].value_counts())

    st.subheader("Trust score distribution")
    st.bar_chart(df["trust_score"])

    st.subheader("Average signal scores")
    signal_cols = ["structural_score", "duplication_score", "pii_score",
                   "toxicity_score", "freshness_score", "hallucination_score"]
    avg_signals = df[signal_cols].mean().rename("average_score")
    st.dataframe(avg_signals)

    st.subheader("Filter by decision")
    decision_filter = st.multiselect(
        "Decision", options=["PASS", "REVIEW", "REJECT"], default=["PASS", "REVIEW", "REJECT"]
    )
    filtered = df[df["decision"].isin(decision_filter)]

    st.subheader("Documents")
    st.dataframe(
        filtered[["doc_id", "source", "trust_score", "decision", "evaluated_at"]],
        use_container_width=True,
    )

    st.subheader("Explanation lookup")
    selected_doc = st.selectbox("Select a doc_id to see its full explanation", filtered["doc_id"])
    if selected_doc:
        row = filtered[filtered["doc_id"] == selected_doc].iloc[0]
        st.code(row["explanation"], language=None)
