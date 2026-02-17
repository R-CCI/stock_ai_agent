# -*- coding: utf-8 -*-
"""Pestaña de Analisis Técnico — gráficos interactivos, indicadores, patrones y análisis con IA."""

import streamlit as st
import plotly.graph_objects as go


def render_technicals_tab(ta_data: dict, chart: go.Figure | None, technical_analysis: str):
    """Renderiza la pestaña de Análisis Técnico."""

    # ── Gráfico Interactivo ──
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("Generando gráfico...")

    # ── Panel de Indicadores ──
    st.markdown("#### 📊 Panel de Indicadores")
    indicators = ta_data.get("indicators", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        rsi = indicators.get("RSI", 50)
        rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
        st.metric(f"{rsi_color} RSI(14)", f"{rsi:.1f}", delta=indicators.get("RSI_Signal", ""))

    with col2:
        macd_dir = indicators.get("MACD_Direction", "Neutral")
        # Translate signal
        signal = "Alcista" if "Bullish" in macd_dir else "Bajista" if "Bearish" in macd_dir else "Neutral"
        icon = "🟢" if "Bullish" in macd_dir else "🔴" if "Bearish" in macd_dir else "🟡"
        st.metric(f"{icon} MACD", f"{indicators.get('MACD', 0):.3f}", delta=signal)

    with col3:
        adx = indicators.get("ADX", 0)
        adx_signal = indicators.get("ADX_Signal", "")
        # Basic translation for ADX signal
        trans_adx = "Fuerte" if "Strong" in adx_signal else "Débil" if "Weak" in adx_signal else adx_signal
        st.metric("📈 ADX", f"{adx:.1f}", delta=trans_adx)

    with col4:
        st.metric("📏 ATR(14)", f"${indicators.get('ATR', 0):.2f}")

    with col5:
        bb_pos = indicators.get("BB_Position", 50)
        st.metric("📊 Posición BB", f"{bb_pos:.0f}%")

    # Second row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("CCI", f"{indicators.get('CCI', 0):.1f}")
    with col2:
        st.metric("Williams %R", f"{indicators.get('Williams_R', 0):.1f}")
    with col3:
        st.metric("Stoch %K", f"{indicators.get('Stoch_K', 0):.1f}")
    with col4:
        st.metric("Stoch %D", f"{indicators.get('Stoch_D', 0):.1f}")
    with col5:
        st.metric("Ancho BB", f"{indicators.get('BB_Width', 0):.4f}")

    st.divider()

    # ── Tendencia ──
    trend = ta_data.get("trend", {})
    if trend:
        col1, col2, col3 = st.columns(3)
        with col1:
            direction = trend.get("direction", "N/A")
            emoji = "🟢" if "Up" in direction else "🔴" if "Down" in direction else "🟡"
            trans_dir = "Alcista" if "Up" in direction else "Bajista" if "Down" in direction else "Lateral"
            st.metric(f"{emoji} Tendencia", trans_dir)
        with col2:
            st.metric("Pendiente", f"{trend.get('slope_pct', 0):.2f}%")
        with col3:
            st.metric("R²", f"{trend.get('r_squared', 0):.3f}")

    st.divider()

    # ── Patrones Detectados ──
    patterns = ta_data.get("patterns", [])
    if patterns:
        st.markdown("#### ⚡ Patrones Detectados")
        for p in patterns:
            ptype = p.get("type", "Desconocido")
            # Basic translation for common patterns
            trans_type = ptype.replace("Bullish", "Alcista").replace("Bearish", "Bajista")
            is_bull = "Bullish" in ptype or "Alcista" in trans_type
            color = "#00D4AA" if is_bull else "#FF4757"
            icon = "🟢" if is_bull else "🔴"

            # Reconstruct the HTML safely to avoid leakage
            date_val = p.get("date", p.get("head_date", p.get("second_peak", p.get("second_trough", ""))))
            reliability = p.get("reliability", "N/A")
            description = p.get("description", "")
            
            # Translate reliability
            trans_rel = reliability.replace("High", "Alta").replace("Moderate", "Moderada").replace("Low", "Baja")

            html_content = f"""
            <div style="background:rgba({'0,212,170' if is_bull else '255,71,87'},0.1);
                        border-left:3px solid {color};padding:10px 15px;margin:5px 0;border-radius:6px;">
                <strong style="color:white;">{icon} {trans_type}</strong><br>
                <span style="color:#aaa;font-size:0.85rem;">
                    {"Fecha: " + date_val + " | " if date_val else ""}
                    {"Fiabilidad: " + trans_rel + " | " if reliability else ""}
                    {description}
                </span>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.info("No se detectaron patrones gráficos significativos en la ventana actual.")

    # ── Soporte / Resistencia ──
    sr = ta_data.get("support_resistance", {})
    if sr:
        st.markdown("#### 🎯 Niveles de Soporte y Resistencia")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Soporte** 🟢")
            for s in sr.get("support", [])[:5]:
                # Force float conversion for formatting
                try:
                    val = float(s)
                    st.markdown(f"- ${val:,.2f}")
                except (ValueError, TypeError):
                    st.markdown(f"- ${s}")
        with col2:
            st.markdown("**Resistencia** 🔴")
            for r in sr.get("resistance", [])[:5]:
                try:
                    val = float(r)
                    st.markdown(f"- ${val:,.2f}")
                except (ValueError, TypeError):
                    st.markdown(f"- ${r}")

    st.divider()

    # ── Análisis de IA ──
    if technical_analysis:
        st.markdown("#### 🤖 Análisis Técnico con IA")
        st.markdown(technical_analysis)
