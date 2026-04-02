from __future__ import annotations

"""Dashboard opcional para monitoreo de señales y drawdown."""

from pathlib import Path

import pandas as pd

# === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===

def run_dashboard() -> None:
    try:
        import streamlit as st
    except Exception:
        print("Streamlit no disponible. Instala dependencias opcionales.")
        return

    st.set_page_config(page_title="BETTERME Dashboard", layout="wide")
    st.title("BETTERME / PHOENIX-BESTIA V4 Dashboard")

    csv_path = Path("audit_log.csv")
    if not csv_path.exists():
        st.warning("No se encontró audit_log.csv todavía.")
        return

    df = pd.read_csv(csv_path)
    st.metric("Spins registrados", len(df))

    if "confidence" in df:
        st.line_chart(df["confidence"].tail(300), height=180)
    if "edge" in df:
        st.line_chart(df["edge"].tail(300), height=180)

    # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
    if "bet" in df:
        equity = 100.0 + df["bet"].fillna(0).cumsum() * 0.0
        peak = equity.cummax()
        drawdown = ((peak - equity) / peak.replace(0, 1)).fillna(0)
        st.subheader("Drawdown (estimado)")
        st.line_chart(drawdown.tail(300), height=180)

    st.subheader("Últimos registros")
    st.dataframe(df.tail(40), use_container_width=True)


if __name__ == "__main__":
    run_dashboard()
