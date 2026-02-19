# -*- coding: utf-8 -*-
"""Overview tab — company info, news, market sentiment."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.config import INSTRUMENT_ETF, INSTRUMENT_REIT


def render_overview_tab(overview: dict, news_analysis: str, market_analysis: str,
                        macro_analysis: str, etf_holdings: dict | None = None):
    """Render the Overview tab content."""

    instrument = overview.get("instrument_type", "Stock")

    # ── Header Card ──
    col_logo, col_info, col_price = st.columns([1, 3, 2])

    with col_logo:
        logo_url = overview.get("logo_url", "")
        if logo_url:
            st.image(logo_url, width=80)
        else:
            st.markdown(
                f"""<div style="width:80px;height:80px;border-radius:12px;
                    background:linear-gradient(135deg,#00D4AA,#6C5CE7);
                    display:flex;align-items:center;justify-content:center;
                    font-size:2rem;font-weight:bold;color:white;">
                    {overview.get('ticker', '?')[:2]}
                </div>""",
                unsafe_allow_html=True,
            )

    with col_info:
        name = overview.get("name") or overview.get("ticker", "")
        st.markdown(f"### {name}")
        trans_inst = {"Stock": "Acción", "ETF": "ETF", "REIT": "REIT"}.get(instrument, instrument)
        badge_color = {"Stock": "#00D4AA", "ETF": "#6C5CE7", "REIT": "#FFA502"}.get(instrument, "#888")
        st.markdown(
            f"""<span style="background:{badge_color};color:white;padding:3px 10px;
                border-radius:12px;font-size:0.75rem;font-weight:600;">
                {trans_inst}</span>""",
            unsafe_allow_html=True,
        )
        if instrument == INSTRUMENT_ETF:
            st.caption(f"Categoría: {overview.get('category', 'N/A')}")
        else:
            st.caption(f"{overview.get('sector', 'N/A')} → {overview.get('industry', 'N/A')}")

    with col_price:
        price = overview.get("price", 0)
        st.metric("Precio Actual", f"${price:,.2f}" if isinstance(price, (int, float)) else f"${price}")

    st.divider()

    # ── Description ──
    desc = overview.get("description", "")
    if desc:
        with st.expander("📝 Descripción de la Empresa", expanded=False):
            st.write(desc[:2000])

    # ── ETF Holdings ──
    if instrument == INSTRUMENT_ETF and etf_holdings:
        st.markdown("#### 🏦 Principales Posiciones")
        holdings = etf_holdings.get("top_holdings", [])
        if holdings:
            holdings_df = pd.DataFrame(holdings)
            if not holdings_df.empty:
                st.dataframe(holdings_df, use_container_width=True, hide_index=True)

        sector_weights = etf_holdings.get("sector_weights", {})
        if sector_weights:
            st.markdown("#### 📊 Asignación por Sector")
            # Flatten if it's a list of dicts
            if isinstance(sector_weights, list):
                flat = {}
                for sw in sector_weights:
                    if isinstance(sw, dict):
                        flat.update(sw)
                sector_weights = flat

            if sector_weights:
                labels = list(sector_weights.keys())
                values = [float(v) * 100 if float(v) <= 1 else float(v) for v in sector_weights.values()]
                fig = go.Figure(data=[go.Pie(
                    labels=labels, values=values,
                    hole=0.4,
                    marker=dict(colors=[
                        "#00D4AA", "#6C5CE7", "#FFD93D", "#FF4757", "#74B9FF",
                        "#FFA502", "#55EFC4", "#636E72", "#E17055", "#00CEC9",
                        "#FDCB6E", "#A29BFE",
                    ]),
                    textinfo="label+percent",
                    textfont_size=10,
                )])
                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        # ETF metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ta = overview.get("total_assets", 0)
            st.metric("Activos Totales", f"${ta / 1e9:.2f}B" if ta and ta > 1e6 else "N/A")
        with col2:
            er = overview.get("expense_ratio")
            st.metric("Ratio de Gastos", f"{er * 100:.2f}%" if er else "N/A")
        with col3:
            yld = overview.get("yield")
            st.metric("Rendimiento (Yield)", f"{yld * 100:.2f}%" if yld else "N/A")
        with col4:
            beta = overview.get("beta_3y")
            st.metric("Beta (3A)", f"{beta:.2f}" if beta else "N/A")

    # ── REIT Metrics ──
    if instrument == INSTRUMENT_REIT:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            dy = overview.get("dividend_yield")
            st.metric("Rendimiento de Dividendos", f"{dy * 100:.2f}%" if dy else "N/A")
        with col2:
            pr = overview.get("payout_ratio")
            st.metric("Ratio de Pago", f"{pr * 100:.1f}%" if pr else "N/A")
        with col3:
            st.metric("FFO/Acción (proxy)", f"${overview.get('ffo_per_share', 'N/A')}")
        with col4:
            st.metric("Market Cap", overview.get("market_cap", "N/A"))

    st.divider()

    # ── News Analysis ──
    if news_analysis:
        st.markdown("#### 📰 Análisis de Noticias y Sentimiento")
        st.markdown(news_analysis)

    # ── Market Overview ──
    if market_analysis:
        with st.expander("🌍 Resumen del Mercado", expanded=False):
            st.markdown(market_analysis)

    # ── Macro ──
    if macro_analysis:
        with st.expander("🏛️ Informe Macroeconómico", expanded=False):
            st.markdown(macro_analysis)
