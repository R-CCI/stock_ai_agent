# -*- coding: utf-8 -*-
"""Application configuration and settings."""

import os
import streamlit as st
import yfinance as yf


def get_api_key() -> str:
    """Get OpenAI API key from Streamlit secrets or environment variable."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")


def get_risk_free_rate() -> float:
    """Fetch the current 10Y US Treasury yield from Yahoo Finance."""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return 4.25  # fallback


# ── Model defaults ──────────────────────────────────────────────────────────
DEFAULT_MODEL = "gpt-4o"
ANALYSIS_MODEL = "gpt-4o"  # for chart / vision analysis
TEMPERATURE = 0.3

# ── Instrument type constants ───────────────────────────────────────────────
INSTRUMENT_STOCK = "Stock"
INSTRUMENT_ETF = "ETF"
INSTRUMENT_REIT = "REIT"

# ── Style & branding ───────────────────────────────────────────────────────
ACCENT_COLOR = "#00D4AA"
WARNING_COLOR = "#FF6B6B"
BULLISH_COLOR = "#00D4AA"
BEARISH_COLOR = "#FF4757"
NEUTRAL_COLOR = "#FFA502"

# ── Simulation defaults ────────────────────────────────────────────────────
DEFAULT_SIMULATION_DAYS = 60
DEFAULT_N_SIMULATIONS = 1000
DEFAULT_LOOKBACK_DAYS = 500
DEFAULT_LOOKBACK_YEARS = 5
DEFAULT_BENCHMARK = "SPY"
