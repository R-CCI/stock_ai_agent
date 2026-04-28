# -*- coding: utf-8 -*-
"""DCF Tab — Multi-stage valuation with scenario analysis and sensitivity heatmap."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.analysis import (
    compute_multistage_dcf, compute_dcf_scenarios,
    compute_capm_cost_of_equity, compute_wacc,
)
from src.data_fetcher import parse_financial_val
from src.config import get_risk_free_rate


def _render_etf_valuation(ticker: str, overview: dict, fundamentals: dict):
    """ETF-specific valuation: NAV, performance, fee-drag, risk-adjusted metrics."""
    import yfinance as yf
    import numpy as np
    from src.data_fetcher import _get_yf_info

    st.markdown(f"### 📊 Análisis de Valor — ETF {ticker}")
    st.caption("Para ETFs la valoración se basa en NAV, métricas de rendimiento, costo del fondo y análisis de tracking.")

    info = _get_yf_info(ticker)
    price = float(overview.get("price") or info.get("regularMarketPrice") or info.get("navPrice") or 0)
    nav   = float(info.get("navPrice") or price or 0)
    total_assets = float(info.get("totalAssets") or 0)
    shares_out   = float(info.get("impliedSharesOutstanding") or info.get("sharesOutstanding") or 1)
    expense_ratio = float(info.get("annualReportExpenseRatio") or info.get("expenseRatio") or 0)
    three_yr_return = float(info.get("threeYearAverageReturn") or 0)
    five_yr_return  = float(info.get("fiveYearAverageReturn") or 0)
    ytd_return  = float(info.get("ytdReturn") or 0)
    beta        = float(info.get("beta3Year") or info.get("beta") or 1.0)
    category    = info.get("category") or info.get("fundFamily") or "N/A"

    # ── KPI Row ────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Precio", f"${price:,.2f}")
    if nav and nav != price:
        prem = round((price - nav) / nav * 100, 2)
        k2.metric("NAV", f"${nav:,.2f}", delta=f"{prem:+.2f}% vs precio")
    else:
        k2.metric("NAV estimado", f"${nav:,.2f}" if nav else "N/A")
    k3.metric("Activos Totales", f"${total_assets/1e9:.2f}B" if total_assets >= 1e9 else f"${total_assets/1e6:.0f}M" if total_assets else "N/A")
    k4.metric("Ratio de Gastos", f"{expense_ratio*100:.2f}%" if expense_ratio else "N/A")
    k5.metric("Beta (3Y)", f"{beta:.2f}")

    st.divider()

    # ── Performance ────────────────────────────────────────────────────────
    st.markdown("#### 📈 Rendimiento del Fondo")
    perf_cols = st.columns(3)
    with perf_cols[0]:
        color = "#00D4AA" if ytd_return >= 0 else "#FF4757"
        st.markdown(f'<div style="text-align:center;padding:18px;background:rgba(0,0,0,0.2);border-radius:10px;border:1px solid {color};">'
                    f'<div style="color:#888;font-size:.75rem;">Rendimiento YTD</div>'
                    f'<div style="color:{color};font-size:2rem;font-weight:700;">{ytd_return*100:+.2f}%</div></div>',
                    unsafe_allow_html=True)
    with perf_cols[1]:
        color = "#00D4AA" if three_yr_return >= 0 else "#FF4757"
        st.markdown(f'<div style="text-align:center;padding:18px;background:rgba(0,0,0,0.2);border-radius:10px;border:1px solid {color};">'
                    f'<div style="color:#888;font-size:.75rem;">Retorno Anual 3Y</div>'
                    f'<div style="color:{color};font-size:2rem;font-weight:700;">{three_yr_return*100:+.2f}%</div></div>',
                    unsafe_allow_html=True)
    with perf_cols[2]:
        color = "#00D4AA" if five_yr_return >= 0 else "#FF4757"
        st.markdown(f'<div style="text-align:center;padding:18px;background:rgba(0,0,0,0.2);border-radius:10px;border:1px solid {color};">'
                    f'<div style="color:#888;font-size:.75rem;">Retorno Anual 5Y</div>'
                    f'<div style="color:{color};font-size:2rem;font-weight:700;">{five_yr_return*100:+.2f}%</div></div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Fee-Drag Calculator ────────────────────────────────────────────────
    st.markdown("#### 💸 Calculadora de Arrastre por Comisiones")
    st.caption("El ratio de gastos reduce silenciosamente el rendimiento compuesto a largo plazo.")
    col_fd1, col_fd2 = st.columns([1, 2])
    with col_fd1:
        inv_amount = st.number_input("Inversión inicial ($)", value=10000, min_value=1000, step=1000, key="etf_inv")
        gross_return = st.number_input("Retorno bruto anual (%)", value=round(three_yr_return*100 or 7.0, 1), step=0.5, key="etf_ret")
        years_proj   = st.slider("Horizonte (años)", 5, 30, 10, key="etf_yrs")
        exp_r_input  = st.number_input("Ratio de Gastos (%)", value=round(expense_ratio*100, 2) if expense_ratio else 0.95, step=0.01, key="etf_exp")

    net_return = (gross_return - exp_r_input) / 100
    gross_r    = gross_return / 100
    gross_fv = inv_amount * ((1 + gross_r)  ** years_proj)
    net_fv   = inv_amount * ((1 + net_return) ** years_proj)
    fee_cost = gross_fv - net_fv

    with col_fd2:
        fig_fd = go.Figure()
        years_range = list(range(years_proj + 1))
        fig_fd.add_trace(go.Scatter(x=years_range, y=[inv_amount*(1+gross_r)**y for y in years_range],
                                    name="Sin comisiones", line=dict(color="#00D4AA", width=2)))
        fig_fd.add_trace(go.Scatter(x=years_range, y=[inv_amount*(1+net_return)**y for y in years_range],
                                    name="Con comisiones", line=dict(color="#FF4757", width=2), fill="tonexty",
                                    fillcolor="rgba(255,71,87,0.1)"))
        fig_fd.update_layout(height=250, margin=dict(t=10,b=30,l=0,r=0),
                             xaxis_title="Años", yaxis_title="Valor ($)",
                             legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_fd, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Bruto", f"${gross_fv:,.0f}")
        c2.metric("Valor Neto", f"${net_fv:,.0f}")
        c3.metric("Costo Total Comisiones", f"${fee_cost:,.0f}", delta=f"-{fee_cost/gross_fv*100:.1f}%", delta_color="inverse")

    st.divider()

    # ── Risk-Return ────────────────────────────────────────────────────────
    st.markdown("#### ⚖️ Métricas de Riesgo-Retorno (estimadas 3Y)")
    try:
        hist = yf.download(ticker, period="3y", interval="1mo", progress=False, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)
        rets = hist["Close"].pct_change().dropna()
        ann_ret   = rets.mean() * 12
        ann_vol   = rets.std() * (12 ** 0.5)
        rfr_m     = 0.045
        sharpe    = (ann_ret - rfr_m) / ann_vol if ann_vol else 0
        downside  = rets[rets < 0].std() * (12 ** 0.5)
        sortino   = (ann_ret - rfr_m) / downside if downside else 0
        max_dd_val = ((hist["Close"] / hist["Close"].cummax()) - 1).min()

        rr1, rr2, rr3, rr4 = st.columns(4)
        rr1.metric("Retorno Anualizado", f"{ann_ret*100:+.2f}%")
        rr2.metric("Volatilidad Anualizada", f"{ann_vol*100:.2f}%")
        rr3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        rr4.metric("Sortino Ratio", f"{sortino:.2f}")
        st.metric("Max Drawdown", f"{max_dd_val*100:.2f}%", delta_color="inverse")
    except Exception:
        st.info("Métricas de riesgo no disponibles. Se requiere historial de precios.")


def render_dcf_tab(ticker: str, overview: dict, fundamentals: dict, instrument_type: str = "stock"):
    """Render the multi-stage DCF valuation tab, or ETF analysis if instrument is an ETF."""
    from src.config import INSTRUMENT_ETF
    if instrument_type == INSTRUMENT_ETF:
        return _render_etf_valuation(ticker, overview, fundamentals)

    st.markdown(f"### 💸 Valoración Intrínseca (DCF Multi-Etapa) — {ticker}")
    st.caption(
        "Modelo de tres etapas: Crecimiento Alto → Transición → Terminal. "
        "Incluye análisis de escenarios Bull/Base/Bear y tabla de sensibilidad WACC × TG."
    )

    # ── Extract base financials ─────────────────────────────────────────
    fcf_raw    = fundamentals.get("Free Cash Flow", "N/A")
    mkt_cap    = parse_financial_val(fundamentals.get("Market Cap", 0))
    p_fcf      = parse_financial_val(fundamentals.get("P/FCF", 0))
    fcf_base   = parse_financial_val(fcf_raw) if fcf_raw != "N/A" else (mkt_cap / p_fcf if p_fcf > 0 else 0)
    ebitda_base = parse_financial_val(fundamentals.get("EBITDA", 0))
    shares     = parse_financial_val(fundamentals.get("Shs Outstand", 1)) or 1
    beta_val   = float(parse_financial_val(fundamentals.get("Beta", 1.2)) or 1.2)
    curr_price = float(overview.get("price") or 0)

    # CAPM-based cost of equity as default WACC
    rfr = get_risk_free_rate() / 100
    default_coe = compute_capm_cost_of_equity(beta=beta_val, risk_free_rate=rfr)
    default_wacc = round(max(min(default_coe * 100, 25.0), 8.0), 1)

    # ── WACC Builder ───────────────────────────────────────────────────
    with st.expander("🔧 Constructor de WACC (CAPM)", expanded=False):
        st.caption("Usa el CAPM para estimar el costo de capital propio y el WACC final.")
        c1, c2, c3 = st.columns(3)
        with c1:
            rfr_input = st.number_input("Tasa Libre de Riesgo (%)", value=round(rfr * 100, 2), min_value=0.0, max_value=15.0, step=0.1, key="dcf_rfr")
            beta_input = st.number_input("Beta", value=beta_val, min_value=0.1, max_value=5.0, step=0.05, key="dcf_beta")
        with c2:
            erp = st.number_input("Prima de Riesgo de Mercado (ERP %)", value=5.5, min_value=1.0, max_value=15.0, step=0.1, key="dcf_erp")
            debt_weight = st.slider("Peso de Deuda (D/V %)", 0, 80, 20, key="dcf_dw") / 100
        with c3:
            cost_of_debt = st.number_input("Costo de Deuda Pre-Tax (%)", value=5.0, min_value=1.0, max_value=20.0, step=0.1, key="dcf_cod")
            tax_rate = st.slider("Tasa Impositiva (%)", 10, 40, 21, key="dcf_tax") / 100

        eq_weight = 1 - debt_weight
        capm_coe  = rfr_input / 100 + beta_input * (erp / 100)
        wacc_computed = compute_wacc(eq_weight, debt_weight, capm_coe, cost_of_debt / 100, tax_rate)
        st.info(
            f"**CAPM Costo de Capital Propio:** {capm_coe*100:.2f}%  |  "
            f"**WACC Computado:** {wacc_computed*100:.2f}%"
        )
        default_wacc = round(max(5.0, min(35.0, wacc_computed * 100)), 2)

    # ── Model Parameters ────────────────────────────────────────────────
    with st.expander("⚙️ Parámetros del Modelo DCF", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stage1_growth = st.slider("Crec. Etapa 1 (%) — Años 1-5", 0, 40, 12, key="dcf_s1g") / 100
            stage2_growth = st.slider("Crec. Etapa 2 (%) — Años 6-10", 0, 20, 7, key="dcf_s2g") / 100
        with c2:
            terminal_growth = st.slider("Crec. Terminal (%)", 0.0, 5.0, 2.5, step=0.25, key="dcf_tg") / 100
            wacc = st.number_input("WACC (%)", value=default_wacc, min_value=5.0, max_value=35.0, step=0.25, key="dcf_wacc") / 100
        with c3:
            exit_mult = st.number_input("Múltiplo EV/EBITDA Salida", value=12.0, step=0.5, key="dcf_em")
            stage1_yrs = st.number_input("Años Etapa 1", value=5, min_value=3, max_value=10, step=1, key="dcf_s1y")
        with c4:
            stage2_yrs = st.number_input("Años Etapa 2", value=5, min_value=2, max_value=10, step=1, key="dcf_s2y")
            manual_fcf = st.number_input("FCF Base Manual ($M)", value=float(fcf_base / 1e6) if fcf_base else 100.0, step=10.0, key="dcf_fcf")

    fcf_input = manual_fcf * 1e6

    # ── Compute DCF ─────────────────────────────────────────────────────
    dcf_result = compute_multistage_dcf(
        fcf_base=fcf_input,
        stage1_growth=stage1_growth,
        stage2_growth=stage2_growth,
        terminal_growth=terminal_growth,
        wacc=wacc,
        stage1_years=int(stage1_yrs),
        stage2_years=int(stage2_yrs),
        net_debt=0,
        shares_outstanding=shares,
        ebitda_base=ebitda_base,
        exit_multiple=exit_mult,
    )

    g_price = dcf_result["gordon"]["implied_price"]
    e_price = dcf_result["exit"]["implied_price"]
    avg_price = (g_price + e_price) / 2 if e_price > 0 else g_price
    upside = (avg_price - curr_price) / curr_price if curr_price > 0 else 0
    margin_of_safety = max(0, (1 - curr_price / g_price)) * 100 if g_price > 0 else 0

    # ── KPI Metrics ─────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Precio Actual", f"${curr_price:,.2f}" if curr_price else "N/D")
    with c2:
        st.metric("Intrínseco (Gordon)", f"${g_price:,.2f}",
                  delta=f"{(g_price/curr_price-1)*100:+.1f}%" if curr_price > 0 else None)
    with c3:
        st.metric("Intrínseco (Exit)", f"${e_price:,.2f}" if e_price > 0 else "N/D",
                  delta=f"{(e_price/curr_price-1)*100:+.1f}%" if curr_price > 0 and e_price > 0 else None)
    with c4:
        st.metric("Margen de Seguridad", f"{margin_of_safety:.1f}%")
    with c5:
        label = "INFRAVALORADA" if upside > 0.10 else "SOBREVALORADA" if upside < -0.10 else "VALOR JUSTO"
        color = "green" if upside > 0.10 else "red" if upside < -0.10 else "orange"
        st.markdown(
            f"<div style='padding:8px; text-align:center; border-radius:8px; border:1px solid {color};'>"
            f"<span style='color:{color}; font-weight:700; font-size:1.0rem;'>{label}</span></div>",
            unsafe_allow_html=True,
        )

    # ── Charts ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        # FCF Waterfall chart
        projs = dcf_result["projections"]
        years_list  = [f"Año {p['year']}" for p in projs]
        fcf_values  = [p["fcf"] / 1e6 for p in projs]
        pv_values   = [p["pv_fcf"] / 1e6 for p in projs]
        stage_colors = [
            "#00D4AA" if p["stage"] == 1 else "#6C5CE7" for p in projs
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=years_list, y=fcf_values,
            name="FCF Proyectado ($M)", marker_color=stage_colors, opacity=0.8,
        ))
        fig.add_trace(go.Scatter(
            x=years_list, y=pv_values,
            name="PV del FCF ($M)", mode="lines+markers",
            line=dict(color="#FFD93D", width=2), marker=dict(size=6),
        ))
        fig.update_layout(
            title=f"Proyección de FCF — {int(stage1_yrs + stage2_yrs)} Años",
            template="plotly_dark", height=340,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
            xaxis_title="Año", yaxis_title="USD Millones",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Value bridge (waterfall)
        sum_pv = dcf_result["sum_pv_cf"] / 1e6
        pv_tv  = dcf_result["gordon"]["pv_terminal_value"] / 1e6
        ev     = dcf_result["gordon"]["enterprise_value"] / 1e6

        fig2 = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["PV Flujos (FCF)", "PV Valor Terminal", "Valor Empresa (EV)"],
            y=[sum_pv, pv_tv, 0],
            text=[f"${sum_pv:,.0f}M", f"${pv_tv:,.0f}M", f"${ev:,.0f}M"],
            textposition="outside",
            connector=dict(line=dict(color="rgba(255,255,255,0.3)")),
            increasing=dict(marker=dict(color="#00D4AA")),
            totals=dict(marker=dict(color="#6C5CE7")),
        ))
        fig2.update_layout(
            title="Descomposición del Valor",
            template="plotly_dark", height=340,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Scenario Analysis ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎭 Análisis de Escenarios — Bull / Base / Bear")

    scenarios = compute_dcf_scenarios(
        fcf_base=fcf_input,
        wacc_base=wacc,
        terminal_growth_base=terminal_growth,
        shares_outstanding=shares,
        ebitda_base=ebitda_base,
        exit_multiple=exit_mult,
    )

    sc = scenarios["scenarios"]
    scen_labels  = [sc[k]["label"] for k in sc]
    scen_gordon  = [sc[k]["gordon_price"] for k in sc]
    scen_exit    = [sc[k]["exit_price"] for k in sc]
    scen_colors  = [sc[k]["color"] for k in sc]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="Gordon Growth", x=scen_labels, y=scen_gordon,
        marker_color=scen_colors, opacity=0.9,
        text=[f"${v:,.2f}" for v in scen_gordon], textposition="outside",
    ))
    fig3.add_trace(go.Bar(
        name="Exit Multiple", x=scen_labels, y=scen_exit,
        marker_color=scen_colors, opacity=0.55,
        text=[f"${v:,.2f}" for v in scen_exit], textposition="outside",
    ))
    if curr_price:
        fig3.add_hline(
            y=curr_price, line_dash="dash", line_color="#FFA502",
            annotation_text=f"  Precio Actual: ${curr_price:,.2f}",
            annotation_position="top left",
        )
    fig3.update_layout(
        barmode="group",
        title="Precio Intrínseco por Escenario",
        template="plotly_dark", height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Precio Estimado ($)",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Scenario comparison table
    scen_df = pd.DataFrame([
        {
            "Escenario": sc[k]["label"],
            "Crec. S1": f"{sc[k]['stage1_growth']*100:.0f}%",
            "Crec. S2": f"{sc[k]['stage2_growth']*100:.0f}%",
            "WACC":     f"{sc[k]['wacc']*100:.1f}%",
            "TG":       f"{sc[k]['terminal_growth']*100:.1f}%",
            "Precio Gordon": f"${sc[k]['gordon_price']:,.2f}",
            "Precio Exit":   f"${sc[k]['exit_price']:,.2f}",
            "Upside":        f"{(sc[k]['gordon_price']/curr_price-1)*100:+.1f}%" if curr_price else "N/D",
        }
        for k in sc
    ])
    st.dataframe(scen_df, use_container_width=True, hide_index=True)

    # ── Sensitivity Heatmap ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🌡️ Tabla de Sensibilidad — WACC × Crecimiento Terminal")

    sens = scenarios.get("sensitivity_wacc_tg", {})
    if sens:
        sens_df = pd.DataFrame(sens).T
        sens_df.index.name = "WACC%"
        # Highlight current price
        fig4 = px.imshow(
            sens_df.astype(float),
            labels=dict(x="Crec. Terminal (%)", y="WACC (%)", color="Precio ($)"),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            text_auto=".0f",
        )
        if curr_price:
            fig4.update_coloraxes(cmid=curr_price)
        fig4.update_layout(
            title="Precio Intrínseco (Gordon Growth) — Sensibilidad WACC vs TG",
            template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(
            "🟢 Verde = precio intrínseco mayor al precio actual (potencial alcista) | "
            "🔴 Rojo = precio intrínseco menor (potencial bajista). "
            "Celdas en blanco = WACC ≤ TG (modelo no válido)."
        )

    # ── Detailed Projections ────────────────────────────────────────────
    with st.expander("📋 Ver Proyecciones Detalladas"):
        df_proj = pd.DataFrame(dcf_result["projections"])
        df_proj["Etapa"] = df_proj["stage"].map({1: "Alta Creación (S1)", 2: "Transición (S2)"})
        df_proj = df_proj.rename(columns={
            "year": "Año", "growth_rate": "Crec. %",
            "fcf": "FCF ($)", "discount_factor": "Factor Descuento",
            "pv_fcf": "PV FCF ($)",
        })
        df_proj = df_proj[["Año", "Etapa", "Crec. %", "FCF ($)", "Factor Descuento", "PV FCF ($)"]]
        st.dataframe(
            df_proj.style.format({
                "FCF ($)": "${:,.0f}", "PV FCF ($)": "${:,.0f}",
                "Crec. %": "{:.1f}%", "Factor Descuento": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        g = dcf_result["gordon"]
        ex = dcf_result["exit"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Gordon Growth (Crecimiento Perpetuo)**")
            st.write(f"- Suma PV FCFs: ${dcf_result['sum_pv_cf']:,.0f}")
            st.write(f"- PV Valor Terminal: ${g['pv_terminal_value']:,.0f} ({g['pct_from_tv']}% del EV)")
            st.write(f"- Valor de Empresa: ${g['enterprise_value']:,.0f}")
            st.write(f"- **Precio Intrínseco: ${g['implied_price']:,.2f}**")
        with col2:
            st.markdown("**Exit Multiple (EV/EBITDA)**")
            st.write(f"- Suma PV FCFs: ${dcf_result['sum_pv_cf']:,.0f}")
            st.write(f"- PV Valor Terminal: ${ex['pv_terminal_value']:,.0f} ({ex['pct_from_tv']}% del EV)")
            st.write(f"- Valor de Empresa: ${ex['enterprise_value']:,.0f}")
            st.write(f"- **Precio Intrínseco: ${ex['implied_price']:,.2f}**")
