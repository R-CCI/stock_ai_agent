# -*- coding: utf-8 -*-
"""Pestaña de Análisis DCF — Valoración intrínseca interactiva."""

import streamlit as st
import plotly.graph_objects as go
from src.analysis import compute_dcf
from src.technical_analysis import create_dcf_charts
from src.data_fetcher import parse_financial_val


def render_dcf_tab(ticker: str, overview: dict, fundamentals: dict):
    """Renderiza la pestaña de Análisis DCF."""
    st.markdown(f"### 💎 Valoración Intrínseca (DCF) — {ticker}")
    st.info("Este modelo estima el valor de la empresa descontando sus flujos de caja futuros al presente.")

    # ── Extracción de Datos Base ──
    # Intentamos obtener FCF y EBITDA de los fundamentales
    fcf_raw = fundamentals.get("Free Cash Flow", "N/A")
    if fcf_raw == "N/A":
        # Fallback a Market Cap / P/FCF si está disponible
        mkt_cap = parse_financial_val(fundamentals.get("Market Cap", 0))
        p_fcf = parse_financial_val(fundamentals.get("P/FCF", 0))
        fcf_base = mkt_cap / p_fcf if p_fcf > 0 else 0
    else:
        fcf_base = parse_financial_val(fcf_raw)

    ebitda_base = parse_financial_val(fundamentals.get("EBITDA", 0))
    shares = parse_financial_val(fundamentals.get("Shs Outstand", 1))
    
    # Net Debt = Total Debt - Cash
    total_debt = parse_financial_val(fundamentals.get("Debt/Eq", 0)) * parse_financial_val(fundamentals.get("Book/sh", 0)) * shares # Simplified
    # Better: use direct metrics if available
    debt_raw = fundamentals.get("Debt/Eq", "0")
    cash_raw = fundamentals.get("Cash/sh", "0")
    
    # Finviz doesn't give Total Debt directly easily, but we can approximate or use defaults
    # For now, let's look for common keys
    net_debt = 0 # Default if not found
    
    # ── Sidebar/Panel de Control Lateral (específico de esta pestaña) ──
    with st.expander("⚙️ Ajustar Parámetros del Modelo", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            growth = st.slider("Crecimiento FCF (Años 1-5) %", 0, 50, 10) / 100
            wacc = st.slider("Tasa de Descuento (WACC) %", 15, 25, 15) / 100
        with col2:
            method = st.selectbox("Método de Valor Terminal", ["Crecimiento Perpetuo", "Múltiplo de Salida (Exit Multiple)"])
            t_growth = st.slider("Crecimiento Perpetuo %", 0.0, 5.0, 2.0, step=0.1) / 100
        with col3:
            exit_mult = st.number_input("Múltiplo EV/EBITDA", value=12.0, step=0.5)
            manual_fcf = st.number_input("FCF Base Manual ($)", value=float(fcf_base) if fcf_base != 0 else 1e6)

    # ── Cálculo ──
    dcf_results = compute_dcf(
        fcf_base=manual_fcf,
        growth_rate=growth,
        wacc=wacc,
        terminal_growth=t_growth,
        ebitda_base=ebitda_base,
        exit_multiple=exit_mult,
        net_debt=net_debt,
        shares_outstanding=shares
    )

    selected_method = "gordon" if method == "Crecimiento Perpetuo" else "exit"
    res = dcf_results[selected_method]
    
    # ── Visualización ──
    col_chart1, col_chart2 = st.columns([2, 1])
    charts = create_dcf_charts(dcf_results, method=selected_method)
    
    with col_chart1:
        if charts:
            st.plotly_chart(charts[0], use_container_width=True)
    with col_chart2:
        if len(charts) > 1:
            st.plotly_chart(charts[1], use_container_width=True)

    # ── Resumen de Valoración ──
    curr_price = overview.get("price", 0)
    implied_price = res["implied_price"]
    upside = (implied_price - curr_price) / curr_price if curr_price > 0 else 0
    
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Precio Actual", f"${curr_price:,.2f}")
    with c2:
        st.metric("Precio Intrínseco", f"${implied_price:,.2f}", 
                  delta=f"{upside*100:.1f}%", delta_color="normal")
    with c3:
        margin = (1 - curr_price / implied_price) * 100 if implied_price > curr_price else 0
        st.metric("Margen de Seguridad", f"{margin:.1f}%")
    with c4:
        status = "SOBREVALORADA" if upside < -0.1 else "INFRAVALORADA" if upside > 0.1 else "VALOR JUSTO"
        color = "red" if upside < -0.1 else "green" if upside > 0.1 else "orange"
        st.markdown(f"**Estado:** <span style='color:{color}; font-size:1.2rem;'>{status}</span>", unsafe_allow_html=True)

    # ── Detalle de Flujos ──
    with st.expander("📄 Ver Detalle de Proyecciones"):
        df_proj = pd.DataFrame(dcf_results["projections"])
        df_proj.columns = ["Año", "FCF Proyectado ($)", "PV del FCF ($)"]
        st.table(df_proj.style.format({"FCF Proyectado ($)": "${:,.2f}", "PV del FCF ($)": "${:,.2f}"}))
        
        st.write(f"**Suma PV Flujos (5 años):** ${dcf_results['sum_pv_cf']:,.2f}")
        st.write(f"**PV Valor Terminal:** ${res['pv_terminal_value']:,.2f} ({res['pct_from_tv']}% del total)")
        st.write(f"**Valor de Empresa (EV):** ${res['enterprise_value']:,.2f}")
        st.write(f"**Valor de Capital (Equity):** ${res['equity_value']:,.2f}")
