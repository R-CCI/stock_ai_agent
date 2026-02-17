# -*- coding: utf-8 -*-
"""Sidebar component for the Stock AI Agent Streamlit app."""

import streamlit as st
from src.config import DEFAULT_MODEL, DEFAULT_BENCHMARK, DEFAULT_SIMULATION_DAYS, DEFAULT_N_SIMULATIONS


def render_sidebar() -> dict:
    """Render the sidebar and return user configuration."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 10px 0 20px 0;">
                <h1 style="margin:0; font-size:1.8rem; 
                    background: linear-gradient(135deg, #00D4AA, #6C5CE7);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    📊 Stock AI Agent
                </h1>
                <p style="color: #888; font-size: 0.8rem; margin-top:4px;">
                    Análisis de Grado Institucional
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Ticker Input
        ticker = st.text_input(
            "📈 Símbolo (Ticker)",
            value="AAPL",
            placeholder="ej. AAPL, SPY, VNQ",
            key="ticker_input",
        ).upper().strip()

        # Model Selection
        model = st.selectbox(
            "🤖 Modelo LLM",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            key="model_select",
        )

        st.divider()

        # Analysis Parameters
        st.markdown("##### ⚙️ Parámetros")

        benchmark = st.text_input(
            "Benchmark",
            value=DEFAULT_BENCHMARK,
            key="benchmark_input",
        ).upper().strip()

        col1, col2 = st.columns(2)
        with col1:
            sim_days = st.number_input(
                "Días MC",
                min_value=10, max_value=365,
                value=DEFAULT_SIMULATION_DAYS,
                step=10,
                key="sim_days",
            )
        with col2:
            n_sims = st.number_input(
                "Rutas MC",
                min_value=10, max_value=200,
                value=DEFAULT_N_SIMULATIONS,
                step=10,
                key="n_sims",
            )

        st.divider()

        # Run Analysis Button
        run_analysis = st.button(
            "🚀 Ejecutar Análisis Completo",
            use_container_width=True,
            type="primary",
            key="run_analysis_btn",
        )

        # Status
        if "analysis_complete" in st.session_state and st.session_state.analysis_complete:
            st.success("✅ ¡Análisis completado!")
        elif "analysis_running" in st.session_state and st.session_state.analysis_running:
            st.info("⏳ Análisis en progreso...")

        st.divider()
        st.caption("Built with Streamlit + OpenAI")

    return {
        "ticker": ticker,
        "model": model,
        "benchmark": benchmark,
        "sim_days": sim_days,
        "n_sims": n_sims,
        "run_analysis": run_analysis,
    }
