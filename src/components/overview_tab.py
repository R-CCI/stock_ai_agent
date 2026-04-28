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
        # ── ETF Performance Metrics ──
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            ta = overview.get("total_assets", 0)
            st.metric("Activos Totales", f"${ta/1e9:.2f}B" if ta and ta > 1e6 else "N/A")
        with col2:
            er = overview.get("expense_ratio")
            er_val = er * 100 if er else None
            er_str = f"{er_val:.2f}%" if er_val is not None else "N/A"
            er_note = "Bajo" if er_val and er_val < 0.20 else ("Alto" if er_val and er_val > 0.75 else "Moderado")
            st.metric("Expense Ratio", er_str, delta=er_note, delta_color="inverse" if er_val and er_val > 0.5 else "off")
        with col3:
            yld = overview.get("yield")
            st.metric("Rendimiento (Yield)", f"{yld*100:.2f}%" if yld else "N/A")
        with col4:
            ytd = overview.get("ytd_return")
            st.metric("Retorno YTD", f"{ytd*100:.2f}%" if ytd else "N/A",
                      delta_color="normal" if ytd and ytd > 0 else "inverse")
        with col5:
            r3y = overview.get("three_year_return")
            st.metric("Retorno 3 Anos", f"{r3y*100:.2f}%" if r3y else "N/A")
        with col6:
            beta = overview.get("beta_3y")
            st.metric("Beta (3A)", f"{beta:.2f}" if beta else "N/A")

        st.divider()

        # ── Holdings + Sector chart side by side ──
        col_left, col_right = st.columns([2, 3])

        with col_left:
            st.markdown("##### Principales Posiciones")
            holdings = etf_holdings.get("top_holdings", [])
            if holdings:
                holdings_df = pd.DataFrame(holdings)
                # Normalize column names
                if not holdings_df.empty:
                    col_rename = {}
                    for c in holdings_df.columns:
                        cl = c.lower()
                        if "symbol" in cl or "ticker" in cl:
                            col_rename[c] = "Ticker"
                        elif "holding" in cl or "name" in cl:
                            col_rename[c] = "Nombre"
                        elif "weight" in cl or "percent" in cl or "%" in cl:
                            col_rename[c] = "Peso %"
                    holdings_df = holdings_df.rename(columns=col_rename)
                    if "Peso %" in holdings_df.columns:
                        holdings_df["Peso %"] = holdings_df["Peso %"].apply(
                            lambda v: f"{float(v)*100:.2f}%" if float(v) <= 1 else f"{float(v):.2f}%"
                        )
                    st.dataframe(holdings_df.head(10), use_container_width=True, hide_index=True)
            else:
                st.caption("Datos de holdings no disponibles.")

        with col_right:
            sector_weights = etf_holdings.get("sector_weights", {})
            if sector_weights:
                st.markdown("##### Distribucion por Sector")
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

                    PALETTE = [
                        "#00D4AA", "#6C5CE7", "#FFD93D", "#FF4757", "#74B9FF",
                        "#FFA502", "#55EFC4", "#636E72", "#E17055", "#00CEC9",
                        "#FDCB6E", "#A29BFE", "#B2BEC3", "#DFE6E9",
                    ]
                    fig = go.Figure(data=[go.Pie(
                        labels=labels, values=values, hole=0.42,
                        marker=dict(colors=PALETTE[:len(labels)]),
                        textinfo="label+percent", textfont_size=9,
                        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
                    )])
                    fig.update_layout(
                        template="plotly_dark", height=380,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=True,
                        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=9)),
                        margin=dict(l=0, r=120, t=20, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Distribucion sectorial no disponible.")

        # ── Return comparison bar chart ──
        returns_data = {
            "YTD":    overview.get("ytd_return"),
            "3 Anos": overview.get("three_year_return"),
            "5 Anos": overview.get("five_year_return"),
        }
        returns_data = {k: v * 100 for k, v in returns_data.items() if v is not None}
        if returns_data:
            st.markdown("##### Rendimiento Historico del Fondo")
            fig_ret = go.Figure(go.Bar(
                x=list(returns_data.keys()),
                y=list(returns_data.values()),
                marker_color=["#00D4AA" if v >= 0 else "#FF4757" for v in returns_data.values()],
                text=[f"{v:.1f}%" for v in returns_data.values()],
                textposition="outside",
            ))
            fig_ret.update_layout(
                template="plotly_dark", height=220,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Retorno (%)", showlegend=False,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_ret, use_container_width=True)

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
