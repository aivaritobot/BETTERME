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

    # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
    c1, c2, c3 = st.columns(3)
    if "confidence" in df:
        c1.metric("Conf promedio", f"{df['confidence'].fillna(0).mean():.3f}")
    if "edge" in df:
        c2.metric("Edge promedio", f"{df['edge'].fillna(0).mean():.3f}")
    if "should_bet" in df:
        hit_rate = float((df["should_bet"].fillna(False).astype(bool)).mean())
        c3.metric("Hit-rate señal", f"{hit_rate*100:.1f}%")

    if "confidence" in df:
        st.line_chart(df["confidence"].tail(300), height=180)
    if "edge" in df:
        st.line_chart(df["edge"].tail(300), height=180)

    # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
    if "drawdown" in df:
        st.subheader("Drawdown (runtime)")
        st.line_chart(df["drawdown"].fillna(0).tail(300), height=180)

    # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
    st.subheader("Bias estimado (top numbers en histórico)")
    if "top_numbers" in df:
        parsed = df["top_numbers"].astype(str).str.extract(r"(\d+)")[0].dropna()
        if not parsed.empty:
            bias = parsed.value_counts().head(12)
            st.bar_chart(bias)
        else:
            st.info("No hay datos de top_numbers parseables todavía.")

    st.subheader("Últimos registros")
    st.dataframe(df.tail(40), use_container_width=True)


if __name__ == "__main__":
    run_dashboard()
