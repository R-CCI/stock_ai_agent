# -*- coding: utf-8 -*-
"""Earnings Analysis Tab — Historical EPS beat/miss dot-plot + Market Reaction + Markov predictor."""

import streamlit as st
import pandas as pd

from src.analysis import compute_markov_earnings
from src.technical_analysis import create_earnings_chart
from src.config import INSTRUMENT_ETF, INSTRUMENT_REIT


def _render_etf_composition_tab(ticker: str, earnings_data: dict):
    """Show ETF holdings, sector breakdown, and composition metrics instead of earnings."""
    import plotly.graph_objects as go
    import plotly.express as px
    import yfinance as yf
    import numpy as np

    st.markdown(f"### 🏢 Composición del Fondo — {ticker}")

    holdings   = earnings_data.get("holdings", pd.DataFrame())
    top_sectors= earnings_data.get("top_sectors", {})
    alloc      = earnings_data.get("asset_allocation", {})
    metrics    = earnings_data.get("fund_metrics", {})

    # ── Pull extra info directly ─────────────────────────────────────────
    from src.data_fetcher import _get_yf_info
    info = _get_yf_info(ticker)

    aum   = float(info.get("totalAssets") or 0)
    er    = float(info.get("annualReportExpenseRatio") or info.get("expenseRatio") or 0)
    yld   = float(info.get("yield") or 0)
    ytd   = float(info.get("ytdReturn") or 0)
    cat   = info.get("category") or info.get("fundFamily") or "N/A"
    desc  = info.get("longBusinessSummary") or info.get("description") or ""

    # ── KPI row ────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Activos Totales", f"${aum/1e9:.2f}B" if aum>=1e9 else f"${aum/1e6:.0f}M" if aum else "N/A")
    k2.metric("Ratio de Gastos", f"{er*100:.2f}%" if er else "N/A")
    k3.metric("Rendimiento YTD", f"{ytd*100:+.2f}%" if ytd else "N/A")
    k4.metric("Dividendo Yield", f"{yld*100:.2f}%" if yld else "N/A")

    if cat != "N/A":
        st.caption(f"📂 Categoría: **{cat}**")

    st.divider()

    # ── Sector / Holdings grid ────────────────────────────────────────────
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown("#### 📊 Distribución Sectorial")
        if top_sectors:
            labels = list(top_sectors.keys())
            vals   = list(top_sectors.values())
            palette= ['#00D4AA','#6C5CE7','#4ECDC4','#FFD93D','#45B7D1',
                      '#FFA07A','#98D8C8','#F7DC6F','#BB8FCE','#85C1E2']
            fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.45,
                                   textinfo="label+percent",
                                   marker=dict(colors=palette[:len(labels)])))
            fig.update_layout(height=340, showlegend=False,
                              margin=dict(t=10,b=10,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            # No sector data — show rolling performance chart as visual substitute
            st.caption("Distribución sectorial no disponible · mostrando rendimiento histórico")
            try:
                hist = yf.download(ticker, period="5y", interval="1mo",
                                   auto_adjust=True, progress=False)
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel(1)
                close = hist["Close"].dropna()
                rets  = close.pct_change().dropna()
                cum   = (1 + rets).cumprod() - 1
                fig2  = go.Figure(go.Scatter(
                    x=cum.index, y=cum.values * 100,
                    mode="lines", fill="tozeroy",
                    line=dict(color="#00D4AA", width=2),
                    fillcolor="rgba(0,212,170,0.1)"
                ))
                fig2.update_layout(height=320, margin=dict(t=10,b=30,l=0,r=0),
                                   xaxis_title="", yaxis_title="Retorno acumulado %",
                                   title=f"{ticker} — Retorno acumulado 5 años")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.info("No hay datos de distribución sectorial disponibles para este instrumento.")

    with col_right:
        st.markdown("#### 💼 Principales Posiciones")
        if not holdings.empty:
            display_cols = [c for c in ['symbol','name','weight','sector'] if c in holdings.columns]
            top_10 = holdings[display_cols].head(10).copy()
            if 'weight' in top_10.columns:
                top_10['weight'] = pd.to_numeric(top_10['weight'], errors='coerce')
                top_10['weight'] = top_10['weight'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            top_10 = top_10.rename(columns={'symbol':'Ticker','name':'Empresa','weight':'Pond.','sector':'Sector'})
            st.dataframe(top_10, use_container_width=True, hide_index=True)
        elif top_sectors:
            rows = [{"Segmento": k, "Ponderación": f"{v:.1f}%"} for k, v in top_sectors.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            # Commodity / futures ETFs: compute rolling monthly returns table as proxy
            try:
                hist = yf.download(ticker, period="2y", interval="1mo",
                                   auto_adjust=True, progress=False)
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel(1)
                close = hist["Close"].dropna()
                rets  = close.pct_change().dropna() * 100
                rets_df = pd.DataFrame({
                    "Mes": rets.index.strftime("%b %Y"),
                    "Retorno (%)": rets.values.round(2),
                    "Señal": rets.apply(lambda x: "✅" if x >= 0 else "🔴")
                })
                st.caption("Posiciones individuales no disponibles · retornos mensuales:")
                st.dataframe(rets_df.tail(24), use_container_width=True, hide_index=True)
            except Exception:
                st.caption(
                    f"Las posiciones individuales de **{ticker}** no están disponibles en Yahoo Finance.\n\n"
                    "Este es un fondo de futuros/commodities — consulta el prospecto oficial del fondo."
                )

    st.divider()

    # ── Asset allocation bar chart ────────────────────────────────────────
    if alloc:
        st.markdown("#### 🎯 Asignación de Activos")
        alloc_cols = st.columns(len(alloc))
        for col_ui, (asset, pct) in zip(alloc_cols, alloc.items()):
            col_ui.metric(asset, f"{pct:.1f}%")
        st.divider()

    # ── Fund description ─────────────────────────────────────────────────
    if desc:
        with st.expander("📝 Descripción del Fondo", expanded=False):
            st.write(desc[:2000])


def render_earnings_tab(ticker: str, earnings_data: dict, instrument_type: str = "stock"):
    """Render the Earnings Analysis tab (stocks) or ETF Composition (for ETFs)."""

    # For ETFs, show composition instead of earnings
    if instrument_type == INSTRUMENT_ETF:
        return _render_etf_composition_tab(ticker, earnings_data)

    st.markdown(f"### 📊 Análisis de Ganancias — {ticker}")

    history = earnings_data.get("history", pd.DataFrame())
    next_date = earnings_data.get("next_date")
    next_estimate = earnings_data.get("next_estimate")

    if history.empty:
        if next_date or next_estimate is not None:
            st.info(
                "**No se pudo construir el historial trimestral de sorpresas EPS**, "
                "pero sí se detectó información del próximo reporte.\n\n"
                f"- Próxima fecha estimada: `{str(next_date)[:10] if next_date else 'N/A'}`\n"
                f"- EPS estimado: `{f'${float(next_estimate):.2f}' if next_estimate is not None else 'N/A'}`\n\n"
                "Esto normalmente indica un problema parcial de datos históricos en Yahoo Finance, "
                "no necesariamente que la empresa no reporte ganancias."
            )
        else:
            st.info(
                "**No hay historial de ganancias disponible para este ticker.**\n\n"
                "Posibles razones:\n"
                "- El instrumento no reporta EPS trimestral\n"
                "- La cobertura histórica del proveedor es incompleta\n"
                "- El ticker es reciente o cambió de símbolo"
            )
        return

    # ── Summary Metrics ──
    total = len(history)
    beats = history["beat"].sum()
    misses = total - beats
    beat_rate = round(100 * beats / total, 1) if total > 0 else 0.0

    last_row = history.iloc[-1]
    last_surprise = last_row.get("surprise_pct", 0) or 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trimestres Analizados", f"{total}")
    c2.metric("Beats ✅", f"{int(beats)}", delta=f"{beat_rate}% histórico")
    c3.metric("Misses ❌", f"{int(misses)}")
    c4.metric("Última Sorpresa EPS", f"{last_surprise:+.2f}%",
              delta_color="normal" if last_surprise > 0 else "inverse")
    if next_estimate is not None:
        c5.metric("Estimado Próx.", f"${float(next_estimate):.2f}",
                  delta=str(next_date)[:10] if next_date else "")
    else:
        c5.metric("Próx. Ganancias", str(next_date)[:10] if next_date else "N/A")

    st.divider()

    # ── Dual Chart: EPS Dot Plot + Market Reaction ──
    fig = create_earnings_chart(earnings_data, ticker)
    st.plotly_chart(fig, use_container_width=True)

    # Legend note for yellow borders
    if "stock_reaction_pct" in history.columns:
        st.caption("🟡 Borde amarillo en barras = Paradoja de mercado (Beat pero cayó o Miss pero subió)")

    st.divider()

    # ── Market Reaction Summary ──
    if "market_reaction" in history.columns and "stock_reaction_pct" in history.columns:
        st.markdown("#### 🎭 Reacción del Mercado Post-Ganancias")
        st.caption(
            "Análisis de cómo reacciona el precio de la acción al día siguiente de cada reporte. "
            "Un 'Beat + Cae' indica que el mercado ya tenía las expectativas puestas más alto."
        )

        has_react = history["stock_reaction_pct"].notna()
        react_df = history[has_react].copy()

        if not react_df.empty:
            avg_reaction = react_df["stock_reaction_pct"].mean()
            avg_beat_reaction = react_df[react_df["beat"] == True]["stock_reaction_pct"].mean()
            avg_miss_reaction = react_df[react_df["beat"] == False]["stock_reaction_pct"].mean()

            # Paradox counts
            beat_falls = len(react_df[(react_df["surprise_pct"] > 2) & (react_df["stock_reaction_pct"] < -1)])
            miss_rises = len(react_df[(react_df["surprise_pct"] < -2) & (react_df["stock_reaction_pct"] > 1)])
            total_paradox = beat_falls + miss_rises

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Reacción Media (todos)",
                f"{avg_reaction:+.2f}%" if not pd.isna(avg_reaction) else "N/A",
                delta_color="normal" if avg_reaction > 0 else "inverse",
            )
            m2.metric(
                "Reacción tras Beat ✅",
                f"{avg_beat_reaction:+.2f}%" if not pd.isna(avg_beat_reaction) else "N/A",
                delta_color="normal" if not pd.isna(avg_beat_reaction) and avg_beat_reaction > 0 else "inverse",
            )
            m3.metric(
                "Reacción tras Miss ❌",
                f"{avg_miss_reaction:+.2f}%" if not pd.isna(avg_miss_reaction) else "N/A",
                delta_color="inverse" if not pd.isna(avg_miss_reaction) and avg_miss_reaction < 0 else "normal",
            )
            m4.metric(
                "Paradojas 🟡",
                f"{total_paradox}",
                delta=f"{beat_falls} beat-caída / {miss_rises} miss-subida",
            )

            # Reaction distribution by category
            if "market_reaction" in history.columns:
                category_counts = react_df["market_reaction"].value_counts()
                cols_react = st.columns(len(category_counts))
                cat_colors = {
                    "Beat + Sube ✅📈": ("#00C853", "rgba(0,200,83,0.1)"),
                    "Beat + Cae ✅📉": ("#FFD93D", "rgba(255,217,61,0.1)"),
                    "Miss + Cae ❌📉": ("#FF5252", "rgba(255,82,82,0.1)"),
                    "Miss + Sube ❌📈": ("#FFA502", "rgba(255,165,2,0.1)"),
                    "Neutral ⚪": ("#888888", "rgba(136,136,136,0.1)"),
                }
                for col_ui, (cat, count) in zip(cols_react, category_counts.items()):
                    color, bg = cat_colors.get(cat, ("#888", "rgba(0,0,0,0.1)"))
                    pct = round(100 * count / len(react_df), 0)
                    col_ui.markdown(
                        f"""<div style="text-align:center;padding:12px;background:{bg};
                            border:1px solid {color};border-radius:10px;">
                            <div style="font-size:0.75rem;color:#aaa;">{cat}</div>
                            <div style="font-size:1.6rem;font-weight:700;color:{color};">{int(count)}</div>
                            <div style="font-size:0.7rem;color:#888;">{int(pct)}% de veces</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── Markov Chain Prediction ──
    st.markdown("#### 🔮 Predicción — Cadenas de Markov")

    markov = compute_markov_earnings(history)

    if not markov:
        st.info("Se necesitan al menos 2 trimestres para calcular las probabilidades con Markov.")
    else:
        beat_p  = markov["beat_probability"]
        miss_p  = markov["miss_probability"]
        near_p  = markov["near_probability"]
        verdict = markov["verdict"]
        b_streak = markov["beat_streak"]
        m_streak = markov["miss_streak"]
        hist_rate = markov["historical_beat_rate"]
        cur_label = {"B": "Beat ✅", "M": "Miss ❌", "N": "Aprox ⚪"}.get(
            markov["current_state"], markov["current_state"]
        )
        mr = markov.get("markov_reaction", {})

        # ── Two columns: Model 1 | Model 2 ──
        col_m1, col_m2 = st.columns(2, gap="large")

        # ──────────── Model 1: EPS only ────────────
        with col_m1:
            st.markdown("##### Modelo 1 · Solo EPS")
            st.caption("Probabilidad basada exclusivamente en si la empresa batió o no las expectativas.")

            beat_color = "#00C853" if beat_p > miss_p else "#888"
            miss_color = "#FF5252" if miss_p > beat_p else "#888"

            m1a, m1b, m1c = st.columns(3)
            with m1a:
                st.markdown(
                    f"""<div style="text-align:center;padding:14px;background:rgba(0,200,83,0.1);
                        border:2px solid {beat_color};border-radius:10px;">
                        <div style="font-size:0.72rem;color:#888;">BEAT</div>
                        <div style="font-size:2rem;font-weight:700;color:{beat_color};">{beat_p}%</div>
                    </div>""", unsafe_allow_html=True,
                )
            with m1b:
                st.markdown(
                    f"""<div style="text-align:center;padding:14px;background:rgba(255,82,82,0.1);
                        border:2px solid {miss_color};border-radius:10px;">
                        <div style="font-size:0.72rem;color:#888;">MISS</div>
                        <div style="font-size:2rem;font-weight:700;color:{miss_color};">{miss_p}%</div>
                    </div>""", unsafe_allow_html=True,
                )
            with m1c:
                st.markdown(
                    f"""<div style="text-align:center;padding:14px;background:rgba(255,193,7,0.08);
                        border:2px solid #555;border-radius:10px;">
                        <div style="font-size:0.72rem;color:#888;">MEET</div>
                        <div style="font-size:2rem;font-weight:700;color:#FFC107;">{near_p}%</div>
                    </div>""", unsafe_allow_html=True,
                )

            st.markdown(f"**{verdict}**")
            sa, sb, sc = st.columns(3)
            sa.metric("Último", cur_label)
            sb.metric("Racha", f"✅×{b_streak}" if b_streak else (f"❌×{m_streak}" if m_streak else "—"))
            sc.metric("Tasa Hist.", f"{hist_rate}%")

            with st.expander("📊 Matriz de Transición EPS"):
                st.caption("Filas = estado actual · Columnas = próximo estado")
                tm1 = markov["transition_matrix"].copy()
                tm1.index   = ["Beat", "Miss", "Aprox"]
                tm1.columns = ["Beat", "Miss", "Aprox"]
                st.dataframe(tm1.style.format("{:.1%}").background_gradient(cmap="RdYlGn"),
                             use_container_width=True)

        # ──────────── Model 2: EPS + Market Reaction ────────────
        with col_m2:
            st.markdown("##### Modelo 2 · EPS + Reacción de Mercado")
            st.caption("Probabilidad del resultado combinado: si la empresa bate/falla Y cómo reacciona el precio.")

            if mr:
                probs = mr["probabilities"]
                best = mr["best_outcome"]
                best_prob = mr["best_probability"]
                paradox  = mr["paradox_rate"]
                cmb_verdict = mr["combined_verdict"]
                cur_cmb = mr["current_state"]

                color_map2 = {
                    "Beat+Sube ✅📈": ("#00C853", "rgba(0,200,83,0.1)"),
                    "Beat+Cae ✅📉":  ("#FFD93D", "rgba(255,217,61,0.1)"),
                    "Miss+Sube ❌📈": ("#FFA502", "rgba(255,165,2,0.1)"),
                    "Miss+Cae ❌📉":  ("#FF5252", "rgba(255,82,82,0.1)"),
                    "Neutral ⚪":     ("#888888", "rgba(136,136,136,0.08)"),
                }

                # Cards for every active outcome
                for outcome, prob in sorted(probs.items(), key=lambda x: -x[1]):
                    color, bg = color_map2.get(outcome, ("#888", "rgba(0,0,0,0.1)"))
                    border = f"2px solid {color}" if outcome == best else f"1px solid {color}66"
                    st.markdown(
                        f"""<div style="display:flex;align-items:center;justify-content:space-between;
                            padding:8px 14px;margin-bottom:6px;background:{bg};
                            border:{border};border-radius:8px;">
                            <span style="font-size:0.85rem;color:#ccc;">{outcome}</span>
                            <span style="font-size:1.3rem;font-weight:700;color:{color};">{prob}%</span>
                        </div>""", unsafe_allow_html=True,
                    )

                st.markdown(cmb_verdict)

                pm_a, pm_b = st.columns(2)
                pm_a.metric("Último resultado", cur_cmb)
                pm_b.metric("Paradojas históricas", f"{paradox}%",
                            help="% de veces: Beat pero cayó, o Miss pero subió")

                with st.expander("📊 Matriz de Transición EPS + Mercado"):
                    st.caption("Fila = estado actual · Columna = próximo estado")
                    st.dataframe(
                        mr["transition_matrix"].style.format("{:.1%}").background_gradient(cmap="RdYlGn"),
                        use_container_width=True,
                    )
            else:
                st.info("No hay suficientes datos de reacción de precio para construir el Modelo 2. "
                        "Se necesita historial de precios diario alineado con las fechas de ganancias.")

    st.divider()

    # ── History Table ──
    st.markdown("#### 📋 Historial Completo de Ganancias")

    base_cols = ["date", "estimate", "reported", "surprise_pct"]
    extra_cols = []
    if "stock_reaction_pct" in history.columns:
        extra_cols += ["stock_reaction_pct", "stock_gap_pct"]
    if "market_reaction" in history.columns:
        extra_cols += ["market_reaction"]
    extra_cols += ["qoq_pct", "yoy_pct"]

    display_cols = [c for c in base_cols + extra_cols if c in history.columns]
    display_df = history[display_cols].copy().sort_values("date", ascending=False)

    rename = {
        "date": "Fecha", "estimate": "Estimado ($)", "reported": "Reportado ($)",
        "surprise_pct": "Sorpresa (%)", "stock_reaction_pct": "Reacción Stock (%)",
        "stock_gap_pct": "Gap Apertura (%)", "market_reaction": "Reacción Mercado",
        "qoq_pct": "QoQ (%)", "yoy_pct": "YoY (%)",
    }
    display_df = display_df.rename(columns=rename)

    fmt = {"Estimado ($)": "${:.2f}", "Reportado ($)": "${:.2f}",
           "Sorpresa (%)": "{:+.2f}%", "Reacción Stock (%)": "{:+.2f}%",
           "Gap Apertura (%)": "{:+.2f}%", "QoQ (%)": "{:+.1f}%", "YoY (%)": "{:+.1f}%"}
    fmt = {k: v for k, v in fmt.items() if k in display_df.columns}

    def highlight_row(row):
        surprise = row.get("Sorpresa (%)", None)
        if surprise is None or pd.isna(surprise):
            return [""] * len(row)
        return [f"background-color: {'rgba(0,200,83,0.12)' if surprise > 0 else 'rgba(255,82,82,0.12)'}"] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_row, axis=1).format(fmt, na_rep="N/A"),
        use_container_width=True,
        hide_index=True,
    )
