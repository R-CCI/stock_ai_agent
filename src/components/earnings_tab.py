# -*- coding: utf-8 -*-
"""Earnings Analysis Tab — Historical EPS beat/miss dot-plot + Market Reaction + Markov predictor."""

import streamlit as st
import pandas as pd

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
    st.markdown("#### 🔮 Predicción de Beat — Cadenas de Markov")
    st.caption(
        "El modelo aprende el patrón histórico de beat/miss y predice la probabilidad "
        "para el próximo trimestre con base en la transición de estados."
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

        beat_color = "#00C853" if beat_p > miss_p else "#888"
        miss_color = "#FF5252" if miss_p > beat_p else "#888"

        mc1, mc2, mc3 = st.columns(3)
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
                    <div style="font-size:0.8rem;color:#888;">Prob. Meet (~)</div>
                    <div style="font-size:2.5rem;font-weight:700;color:#FFC107;">{near_p}%</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown(f"**Veredicto:** &nbsp; {verdict}", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Último Resultado", current_state_label)
        col_b.metric("Racha Actual",
                     f"✅ × {b_streak}" if b_streak else (f"❌ × {m_streak}" if m_streak else "—"))
        col_c.metric("Tasa Histórica de Beat", f"{hist_rate}%")

        with st.expander("🔬 Ver Matriz de Transición de Markov"):
            st.caption("Filas = estado actual | Columnas = próximo estado | Valores = probabilidad")
            tm = markov["transition_matrix"].copy()
            tm.index = ["Beat (B)", "Miss (M)", "Aprox.(N)"]
            tm.columns = ["Beat (B)", "Miss (M)", "Aprox.(N)"]
            st.dataframe(
                tm.style.format("{:.1%}").background_gradient(cmap="RdYlGn"),
                use_container_width=True,
            )
    else:
        st.info("Se necesitan al menos 2 trimestres para calcular la probabilidad con Markov.")

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
