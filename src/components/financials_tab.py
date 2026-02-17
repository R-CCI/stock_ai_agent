# -*- coding: utf-8 -*-
"""Financials tab — statements, valuation, and fundamental analysis."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_financials_tab(
    instrument_type: str,
    valuation_analysis: str,
    fundamentals_analysis: str,
    statements: dict,
    statement_analyses: dict,
    etf_analysis: str | None = None,
):
    """Render the Financials tab."""

    # ── Valuation ──
    if valuation_analysis:
        st.markdown("#### 💎 Análisis de Valoración")
        st.markdown(valuation_analysis)
        st.divider()

    # ── ETF-specific ──
    if instrument_type == "ETF" and etf_analysis:
        st.markdown("#### 📦 Análisis de Fondo ETF")
        st.markdown(etf_analysis)
        st.divider()

    # ── Fundamentals ──
    if fundamentals_analysis:
        st.markdown("#### 🔬 Análisis Fundamental")
        st.markdown(fundamentals_analysis)
        st.divider()

    # ── Financial Statements ──
    if instrument_type in ("Stock", "REIT") and statements:
        st.markdown("#### 📊 Estados Financieros")

        tab_names = []
        tab_data = []
        for label, key in [("Estado de Resultados", "income"), ("Balance General", "balance"), ("Flujo de Caja", "cashflow")]:
            if key in statements and not statements[key].empty:
                tab_names.append(label)
                tab_data.append((key, statements[key]))

        if tab_names:
            tabs = st.tabs(tab_names)
            for tab, (key, df) in zip(tabs, tab_data):
                with tab:
                    st.dataframe(df, use_container_width=True, hide_index=True, height=400)
                    if key in statement_analyses and statement_analyses[key]:
                        with st.expander("📝 Análisis IA", expanded=True):
                            st.markdown(statement_analyses[key])
