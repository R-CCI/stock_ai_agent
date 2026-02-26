# -*- coding: utf-8 -*-
"""Earnings Analysis Tab — Historical EPS beat/miss dot-plot + Markov Chain predictor."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.analysis import compute_markov_earnings
from src.technical_analysis import create_earnings_chart


def render_earnings_tab(ticker: str, earnings_data: dict):
    """Render the Earnings Analysis tab."""
    st.markdown(f"### 📊 Análisis de Ganancias — {ticker}")

    history = earnings_data.get("history", pd.DataFrame())
    next_date = earnings_data.get("next_date")
    next_estimate = earnings_data.get("next_estimate")

    if history.empty:
        st.warning("⚠️ No hay historial de ganancias disponible para este ticker.")
        return

    # ── Summary Metrics ──
    total = len(history)
    beats = history["beat"].sum()
    misses = total - beats
    beat_rate = round(100 * beats / total, 1) if total > 0 else 0

    last_row = history.iloc[-1]
    last_surprise = last_row.get("surprise_pct", 0) or 0
    last_color = "#00C853" if last_surprise > 0 else "#FF5252"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trimestres Analizados", f"{total}")
    c2.metric("Beats ✅", f"{int(beats)}", delta=f"{beat_rate}% tasa histórica")
    c3.metric("Misses ❌", f"{int(misses)}")
    c4.metric(
        "Última Sorpresa",
        f"{last_surprise:+.2f}%",
        delta_color="normal" if last_surprise > 0 else "inverse",
    )
    if next_estimate is not None:
        c5.metric("Estimado Próx.", f"${float(next_estimate):.2f}",
                  delta=str(next_date)[:10] if next_date else "")
    else:
        c5.metric("Próx. Ganancias", str(next_date)[:10] if next_date else "N/A")

    st.divider()

    # ── Dot Plot Chart ──
    fig = create_earnings_chart(earnings_data, ticker)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Markov Chain Prediction ──
    st.markdown("#### 🔮 Predicción de Beat — Cadenas de Markov")
    st.caption(
        "El modelo de Cadenas de Markov aprende el patrón histórico de beat/miss y predice "
        "la probabilidad para el próximo trimestre en base a la transición de estados."
    )

    markov = compute_markov_earnings(history)

    if markov:
        beat_p = markov["beat_probability"]
        miss_p = markov["miss_probability"]
        near_p = markov["near_probability"]
        verdict = markov["verdict"]
        b_streak = markov["beat_streak"]
        m_streak = markov["miss_streak"]
        hist_rate = markov["historical_beat_rate"]
        current_state_label = {"B": "Beat", "M": "Miss", "N": "Aproximado"}.get(
            markov["current_state"], markov["current_state"]
        )

        # Probability visual
        mc1, mc2, mc3 = st.columns(3)
        beat_color = "#00C853" if beat_p > miss_p else "#888"
        miss_color = "#FF5252" if miss_p > beat_p else "#888"

        with mc1:
            st.markdown(
                f"""<div style="text-align:center;padding:20px;background:rgba(0,200,83,0.1);
                    border:2px solid {beat_color};border-radius:12px;">
                    <div style="font-size:0.8rem;color:#888;">Prob. de BEAT</div>
                    <div style="font-size:2.5rem;font-weight:700;color:{beat_color};">{beat_p}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with mc2:
            st.markdown(
                f"""<div style="text-align:center;padding:20px;background:rgba(255,82,82,0.1);
                    border:2px solid {miss_color};border-radius:12px;">
                    <div style="font-size:0.8rem;color:#888;">Prob. de MISS</div>
                    <div style="font-size:2.5rem;font-weight:700;color:{miss_color};">{miss_p}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with mc3:
            st.markdown(
                f"""<div style="text-align:center;padding:20px;background:rgba(255,193,7,0.08);
                    border:2px solid #888;border-radius:12px;">
                    <div style="font-size:0.8rem;color:#888;">Prob. de Meet (~)</div>
                    <div style="font-size:2.5rem;font-weight:700;color:#FFC107;">{near_p}%</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown(f"**Veredicto:** &nbsp; {verdict}", unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Último Resultado", current_state_label)
        col_b.metric("Racha Actual", f"{'✅ × ' + str(b_streak) if b_streak else ''}"
                     f"{'❌ × ' + str(m_streak) if m_streak else ''}" or "—")
        col_c.metric("Tasa Histórica de Beat", f"{hist_rate}%")

        # Transition matrix
        with st.expander("🔬 Ver Matriz de Transición de Markov"):
            st.caption("Filas = estado actual | Columnas = próximo estado | Valores = probabilidad de transición")
            tm = markov["transition_matrix"].copy()
            tm.index = ["Beat (B)", "Miss (M)", "Aprox.(N)"]
            tm.columns = ["Beat (B)", "Miss (M)", "Aprox.(N)"]
            st.dataframe(tm.style.format("{:.1%}").background_gradient(cmap="RdYlGn"), use_container_width=True)
    else:
        st.info("Se necesitan al menos 2 trimestres de historial para aplicar el modelo de Markov.")

    st.divider()

    # ── History Table ──
    st.markdown("#### 📋 Historial de Ganancias")

    display_df = history[["date", "estimate", "reported", "surprise_pct", "qoq_pct", "yoy_pct"]].copy()
    display_df.columns = ["Fecha", "Estimado ($)", "Reportado ($)", "Sorpresa (%)", "QoQ (%)", "YoY (%)"]
    display_df = display_df.sort_values("Fecha", ascending=False)

    def highlight_row(row):
        surprise = row["Sorpresa (%)"]
        if pd.isna(surprise):
            return [""] * len(row)
        color = "rgba(0,200,83,0.15)" if surprise > 0 else "rgba(255,82,82,0.15)"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        display_df.style
            .apply(highlight_row, axis=1)
            .format({
                "Estimado ($)": "${:.2f}",
                "Reportado ($)": "${:.2f}",
                "Sorpresa (%)": "{:+.2f}%",
                "QoQ (%)": "{:+.1f}%",
                "YoY (%)": "{:+.1f}%",
            }, na_rep="N/A"),
        use_container_width=True,
        hide_index=True,
    )
