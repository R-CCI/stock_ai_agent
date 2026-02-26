# -*- coding: utf-8 -*-
"""Options tab — interactive options chain analysis with user-selectable expiration."""

import streamlit as st
import plotly.graph_objects as go

from src.data_fetcher import get_options_for_expiration
from src.analysis import compute_options_analysis
from src.technical_analysis import create_options_chart


def render_options_tab(
    ticker: str,
    options_analysis: dict,
    options_chart: go.Figure | None,
    options_ai_analysis: str,
):
    """Render the Options tab with interactive expiration selection."""

    if not options_analysis:
        st.warning("⚠️ No hay datos de opciones disponibles para este ticker.")
        return

    # ── Expiration Selector ──
    all_expirations = options_analysis.get("all_expirations", [])
    current_exp = options_analysis.get("expiration", "")

    if all_expirations:
        st.markdown("#### 📅 Seleccionar Fecha de Expiración")
        selected_exp = st.selectbox(
            "Expiración",
            options=all_expirations,
            index=all_expirations.index(current_exp) if current_exp in all_expirations else 0,
            key="options_expiration_selector",
            label_visibility="collapsed",
        )

        # Recalculate if expiration changed
        if selected_exp != current_exp:
            with st.spinner(f"Cargando cadena de opciones para {selected_exp}..."):
                new_data = get_options_for_expiration(ticker, selected_exp)
                if new_data:
                    from src.config import get_risk_free_rate
                    rfr = get_risk_free_rate() / 100
                    options_analysis = compute_options_analysis(new_data, rfr=rfr)
                    options_analysis["all_expirations"] = all_expirations
                    options_chart = create_options_chart(options_analysis)

    st.markdown("---")

    # ── Key Metrics ──
    st.markdown("#### 🎯 Resumen de Opciones")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Precio Spot", f"${options_analysis.get('spot_price', 0):,.2f}")
    with col2:
        st.metric("Expiración", options_analysis.get("expiration", "N/A"))
    with col3:
        st.metric("Días al Vencimiento", f"{options_analysis.get('days_to_expiry', 0)}")
    with col4:
        iv = options_analysis.get("iv_format", 0)
        st.metric("IV ATM", f"{iv}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        em = options_analysis.get("expected_move_return", 0)
        st.markdown(
            f"""<div style="text-align:center;padding:15px;background:linear-gradient(135deg,rgba(0,212,170,0.1),rgba(108,92,231,0.1));
                border-radius:10px;border:1px solid #6C5CE7;">
                <div style="font-size:0.75rem;color:#888;">Movimiento Esperado</div>
                <div style="font-size:1.5rem;font-weight:700;color:#6C5CE7;">±{em}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        lb = options_analysis.get("range_lower_bound", 0)
        ub = options_analysis.get("range_upper_bound", 0)
        st.markdown(
            f"""<div style="text-align:center;padding:15px;background:rgba(0,0,0,0.2);border-radius:10px;">
                <div style="font-size:0.75rem;color:#888;">Rango de Precio</div>
                <div style="font-size:1.2rem;font-weight:600;color:#FFD93D;">${lb} – ${ub}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        mp = options_analysis.get("max_pain_strike", 0)
        mp_dist = options_analysis.get("max_pain_distance", 0)
        color = "#00D4AA" if mp_dist >= 0 else "#FF4757"
        st.markdown(
            f"""<div style="text-align:center;padding:15px;background:rgba(0,0,0,0.2);border-radius:10px;border:1px solid {color};">
                <div style="font-size:0.75rem;color:#888;">Max Pain</div>
                <div style="font-size:1.5rem;font-weight:700;color:{color};">${mp} ({mp_dist:+.1f}%)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        pcr = options_analysis.get("put_call_oi_ratio", 0)
        sentiment = "Bajista" if pcr > 100 else "Alcista" if pcr < 70 else "Neutral"
        s_color = "#FF4757" if sentiment == "Bajista" else "#00D4AA" if sentiment == "Alcista" else "#FFA502"
        st.markdown(
            f"""<div style="text-align:center;padding:15px;background:rgba(0,0,0,0.2);border-radius:10px;border:1px solid {s_color};">
                <div style="font-size:0.75rem;color:#888;">P/C OI Ratio</div>
                <div style="font-size:1.5rem;font-weight:700;color:{s_color};">{pcr}%</div>
                <div style="font-size:0.7rem;color:{s_color};">{sentiment}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Charts ──
    if options_chart:
        st.plotly_chart(options_chart, use_container_width=True)

    st.divider()

    # ── AI Analysis ──
    if options_ai_analysis:
        st.markdown("#### 🤖 Análisis de Opciones IA")
        st.markdown(options_ai_analysis)
