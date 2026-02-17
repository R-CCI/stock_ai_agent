# -*- coding: utf-8 -*-
"""Risk tab — risk metrics, drawdown, Monte Carlo simulations."""

import streamlit as st
import plotly.graph_objects as go


def render_risk_tab(
    risk_metrics: dict,
    risk_analysis: str,
    mc_data: dict | None,
    mc_chart: go.Figure | None,
    mc_analysis: str,
    dd_chart: go.Figure | None,
):
    """Render the Risk & Simulation tab."""

    # ── Risk Metrics ──
    st.markdown("#### ⚠️ Métricas de Riesgo")

    col1, col2, col3 = st.columns(3)
    with col1:
        ret = risk_metrics.get("annualized_return", 0)
        st.metric("Retorno Anualizado", f"{ret:.2f}%", delta=f"{'▲' if ret > 0 else '▼'} {abs(ret):.1f}%")
    with col2:
        st.metric("Volatilidad", f"{risk_metrics.get('volatility', 0):.2f}%")
    with col3:
        beta = risk_metrics.get("beta", 0)
        st.metric("Beta", f"{beta:.3f}", delta="Riesgo alto" if beta > 1.3 else "Moderado" if beta > 0.8 else "Riesgo bajo")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sharpe = risk_metrics.get("sharpe_ratio", 0)
        color = "#00D4AA" if sharpe > 1 else "#FFA502" if sharpe > 0 else "#FF4757"
        st.markdown(
            f"""<div style="text-align:center;padding:15px;background:rgba(0,0,0,0.2);border-radius:10px;border:1px solid {color};">
                <div style="font-size:0.75rem;color:#888;">Sharpe Ratio</div>
                <div style="font-size:1.8rem;font-weight:700;color:{color};">{sharpe:.3f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        sortino = risk_metrics.get("sortino_ratio", 0)
        color = "#00D4AA" if sortino > 1.5 else "#FFA502" if sortino > 0 else "#FF4757"
        st.markdown(
            f"""<div style="text-align:center;padding:15px;background:rgba(0,0,0,0.2);border-radius:10px;border:1px solid {color};">
                <div style="font-size:0.75rem;color:#888;">Sortino Ratio</div>
                <div style="font-size:1.8rem;font-weight:700;color:{color};">{sortino:.3f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        treynor = risk_metrics.get("treynor_ratio", 0)
        st.markdown(
            f"""<div style="text-align:center;padding:15px;background:rgba(0,0,0,0.2);border-radius:10px;">
                <div style="font-size:0.75rem;color:#888;">Treynor Ratio</div>
                <div style="font-size:1.8rem;font-weight:700;color:#74B9FF;">{treynor:.3f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        dd_val = risk_metrics.get("max_drawdown", 0)
        st.markdown(
            f"""<div style="text-align:center;padding:15px;background:rgba(0,0,0,0.2);border-radius:10px;border:1px solid #FF4757;">
                <div style="font-size:0.75rem;color:#888;">Máximo Drawdown</div>
                <div style="font-size:1.8rem;font-weight:700;color:#FF4757;">{dd_val:.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Drawdown Chart ──
    if dd_chart:
        st.plotly_chart(dd_chart, use_container_width=True)

    # ── AI Risk Analysis ──
    if risk_analysis:
        with st.expander("🤖 Análisis de Riesgo IA", expanded=True):
            st.markdown(risk_analysis)

    st.divider()

    # ── Monte Carlo ──
    st.markdown("#### 🎲 Simulación Monte Carlo")

    if mc_chart:
        st.plotly_chart(mc_chart, use_container_width=True)

    if mc_data:
        ps = mc_data.get("percentile_summary", {})
        st.markdown("**Niveles de Precio Proyectados (Fin del Periodo)**")
        cols = st.columns(7)
        labels = [("P1", "🔴"), ("P5", "🔴"), ("P25", "🟡"), ("P50", "🟢"), ("P75", "🟡"), ("P95", "🔵"), ("P99", "🔵")]
        for col, (key, icon) in zip(cols, labels):
            with col:
                val = ps.get(key, 0)
                st.metric(f"{icon} {key}", f"${val:,.2f}" if val else "N/A")

    if mc_analysis:
        with st.expander("🤖 Análisis Monte Carlo IA", expanded=True):
            st.markdown(mc_analysis)
