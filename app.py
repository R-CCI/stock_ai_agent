# -*- coding: utf-8 -*-
"""Stock AI Agent — Streamlit Application (Main Entry Point).

Run with:  streamlit run app.py
"""

import streamlit as st
import json
import traceback
from openai import OpenAI
import pandas as pd

# ── Local modules ────────────────────────────────────────────────────────
from src.config import (
    get_api_key, get_risk_free_rate, DEFAULT_MODEL,
    INSTRUMENT_STOCK, INSTRUMENT_ETF, INSTRUMENT_REIT,
    DEFAULT_LOOKBACK_YEARS,
)
from src.data_fetcher import (
    detect_instrument_type, get_overview, get_price_history,
    get_fundamentals, get_financial_statements, get_valuation_data,
    get_ticker_news, get_market_news, get_macro_blogs,
    get_options_data, get_company_logo_url, get_current_price,
    get_earnings_history,
)
from src.analysis import (
    compute_risk_metrics, run_gmm_montecarlo, compute_options_analysis,
)
from src.technical_analysis import (
    generate_ta_summary, create_technical_chart,
    create_options_charts, create_montecarlo_chart, create_drawdown_chart,
)
from src.prompts import (
    SYSTEM_PROMPT, VALUATION_PROMPT, FUNDAMENTALS_PROMPT,
    TECHNICAL_PROMPT, RISK_METRICS_PROMPT, MONTECARLO_PROMPT,
    OPTIONS_PROMPT, MACRO_PROMPT, MARKET_NEWS_PROMPT,
    STOCK_NEWS_PROMPT, CONCLUSION_PROMPT,
    FINANCIAL_STATEMENT_PROMPT, ETF_ANALYSIS_PROMPT,
)
from src.pdf_report import generate_report

# ── UI Components ────────────────────────────────────────────────────────
from src.components.sidebar import render_sidebar
from src.components.overview_tab import render_overview_tab
from src.components.financials_tab import render_financials_tab
from src.components.technicals_tab import render_technicals_tab
from src.components.risk_tab import render_risk_tab
from src.components.options_tab import render_options_tab
from src.components.dcf_tab import render_dcf_tab
from src.components.earnings_tab import render_earnings_tab
from src.components.report_tab import render_report_tab


# ═══════════════════════════════════════════════════════════════════════════
#  Page Config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stock AI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .stApp { background-color: #0E1117; }
    
    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 1.1rem; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem; }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(14,17,23,0.8);
        padding: 4px 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader { font-weight: 600; }
    
    /* Divider */
    hr { border-color: rgba(255,255,255,0.1) !important; }
    
    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #00D4AA, #6C5CE7); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Helper
# ═══════════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.4) -> str:
    """Call OpenAI LLM with the system prompt and return the response."""
    api_key = get_api_key()
    if not api_key:
        return "⚠️ OpenAI API key not configured. Set OPENAI_API_KEY env var or .streamlit/secrets.toml."
    try:
        client = OpenAI(api_key=api_key)
        
        # Models like o1, o3, or "gpt-5" (o1-based) may not support temperature other than 1.0
        temp = temperature
        if any(m in model.lower() for m in ["o1", "o3", "gpt-5"]):
            temp = 1.0

        response = client.chat.completions.create(
            model=model,
            temperature=temp,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
#  Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_full_analysis(ticker: str, config: dict):
    """Execute the full analysis pipeline and store results in session state."""
    model = config.get("model", DEFAULT_MODEL)
    benchmark = config.get("benchmark", "SPY")
    sim_days = config.get("sim_days", 60)
    n_sims = config.get("n_sims", 50)
    rfr = get_risk_free_rate()

    progress = st.progress(0, text="Iniciando análisis...")
    results = {}
    df_results = {}
    statements = {}
    charts = {}

    try:
        # ── Step 1: Instrument Detection & Overview ──
        progress.progress(5, text="🔍 Detectando tipo de instrumento...")
        instrument_type = detect_instrument_type(ticker)
        overview = get_overview(ticker)
        current_price_data = get_current_price(ticker)
        overview["instrument_type"] = instrument_type
        overview["ticker"] = ticker

        logo_url = get_company_logo_url(ticker)
        overview["logo_url"] = logo_url

        # ── Step 2: Price Data ──
        progress.progress(10, text="📈 Obteniendo historial de precios...")
        price_df = get_price_history(ticker, period=f"{DEFAULT_LOOKBACK_YEARS}y")
        if overview["price"] == 0:
            overview["price"] = price_df["Close"].iloc[-1]

        bm_df = get_price_history(benchmark, period=f"{DEFAULT_LOOKBACK_YEARS}y")

        # ── Step 3: Fundamentals & Valuation ──
        progress.progress(15, text="🔬 Obteniendo fundamentales...")
        fundamentals = get_fundamentals(ticker)
        df_results["fundamentals_raw"] = fundamentals
        valuation_data = get_valuation_data(ticker)

        # ── Step 4: Financial Statements (stocks/REITs only) ──
        if instrument_type in (INSTRUMENT_STOCK, INSTRUMENT_REIT):
            progress.progress(20, text="📊 Cargando estados financieros...")
            for stmt_type, key in [("income", "income"), ("balance", "balance"), ("cashflow", "cashflow")]:
                stmt = get_financial_statements(ticker, stmt_type)
                if stmt is not None and not stmt.empty:
                    statements[key] = stmt

        # ── Step 5: News ──
        progress.progress(25, text="📰 Recopilando noticias...")
        ticker_news = get_ticker_news(ticker)
        market_news_data = get_market_news()

        # ── Step 6: Macro ──
        progress.progress(30, text="🏛️ Leyendo blogs macroeconómicos...")
        macro_blogs = get_macro_blogs()

        # ── Step 7: Options ──
        progress.progress(35, text="🎯 Obteniendo cadena de opciones...")
        raw_options = get_options_data(ticker)
        options_analysis = compute_options_analysis(raw_options, rfr / 100) if raw_options else {}

        # ── Step 8: Technical Analysis ──
        progress.progress(40, text="📊 Calculando indicadores técnicos...")
        ta_data = generate_ta_summary(price_df, ticker)
        charts["technical"] = create_technical_chart(ta_data)

        if options_analysis:
            charts["options"] = create_options_charts(options_analysis, ticker)

        # ── Step 9: Risk Metrics ──
        progress.progress(45, text="⚠️ Calculando métricas de riesgo...")
        risk_metrics = compute_risk_metrics(ticker, price_df, bm_df, rfr)
        charts["drawdown"] = create_drawdown_chart(risk_metrics["drawdown_series"], ticker)

        # ── Step 10: Monte Carlo ──
        progress.progress(50, text="🎲 Ejecutando simulaciones Monte Carlo...")
        mc_data = run_gmm_montecarlo(price_df, days=sim_days, n_simulations=n_sims)
        charts["montecarlo"] = create_montecarlo_chart(mc_data, ticker)

        # ── Step 11: DCF (Baseline) ──
        progress.progress(52, text="💎 Calculando valoración intrínseca (DCF)...")
        from src.data_fetcher import parse_financial_val
        from src.analysis import compute_dcf
        
        fcf_raw = fundamentals.get("Free Cash Flow", 0)
        fcf_base = parse_financial_val(fcf_raw)
        ebitda_base = parse_financial_val(fundamentals.get("EBITDA", 0))
        shares = parse_financial_val(fundamentals.get("Shs Outstand", 1))
        
        dcf_baseline = compute_dcf(
            fcf_base=float(fcf_base) if fcf_base != 0 else 1e6, # Fallback
            growth_rate=0.10, # Baseline 10%
            wacc=0.15,        # Buffett baseline
            terminal_growth=0.02,
            ebitda_base=ebitda_base,
            exit_multiple=12.0,
            shares_outstanding=shares
        )
        results["dcf"] = dcf_baseline

        # ── Step 12: Earnings History ──
        progress.progress(54, text="📊 Obteniendo historial de ganancias...")
        earnings_data = get_earnings_history(ticker)
        st.session_state["earnings_data"] = earnings_data

        # ═══════════════════════════════════════════════════════════════
        #  LLM Analysis Calls
        # ═══════════════════════════════════════════════════════════════

        # ── Stock/Ticker News ──
        progress.progress(55, text="🤖 Analizando sentimiento de noticias...")
        if isinstance(ticker_news, pd.DataFrame) and not ticker_news.empty:
            news_text = "\n".join([f"• {row}" for row in ticker_news["Title"].head(20)])
        else:
            news_text = "No hay noticias recientes."
        results["news"] = call_llm(
            STOCK_NEWS_PROMPT.format(ticker=ticker, news=news_text), model=model,
        )

        # ── Market News ──
        progress.progress(58, text="🤖 Analizando condiciones de mercado...")
        mn = market_news_data if isinstance(market_news_data, dict) else {}

        # Fear & Greed
        fg_df = mn.get("fear_greed", pd.DataFrame())
        if isinstance(fg_df, pd.DataFrame) and not fg_df.empty:
            fg = str(round(fg_df["Fear and Greed Index"].iloc[-1], 1))
        else:
            fg = "N/A"

        # SPY performance
        spy_perf = mn.get("spy_perf", {})

        # Market headlines
        news_df = mn.get("news", pd.DataFrame())
        if isinstance(news_df, pd.DataFrame) and not news_df.empty:
            mkt_headlines = "\n".join([f"• {t}" for t in news_df["Title"].head(15)])
        else:
            mkt_headlines = "No hay noticias de mercado disponibles."

        results["market"] = call_llm(
            MARKET_NEWS_PROMPT.format(
                fear_greed=fg,
                perf_day=spy_perf.get("change", "N/A"),
                perf_week=spy_perf.get("perf_week", "N/A"),
                perf_month=spy_perf.get("perf_month", "N/A"),
                perf_quarter=spy_perf.get("perf_quarter", "N/A"),
                perf_half_y=spy_perf.get("perf_half_y", "N/A"),
                perf_ytd=spy_perf.get("perf_ytd", "N/A"),
                vol_week=spy_perf.get("vol_week", "N/A"),
                vol_month=spy_perf.get("vol_month", "N/A"),
                news=mkt_headlines,
            ), model=model,
        )

        # ── Macro ──
        progress.progress(61, text="🤖 Analizando entorno macroeconómico...")
        if macro_blogs and isinstance(macro_blogs, list):
            macro_text = "\n\n---\n\n".join([b.get("text", "") for b in macro_blogs[:3]])
        else:
            macro_text = "No hay datos macro disponibles."

        results["macro"] = call_llm(
            MACRO_PROMPT.format(macro_articles=macro_text[:8000]), model=model,
        )

        # ── Valuation ──
        progress.progress(65, text="🤖 Realizando análisis de valoración...")
        f = fundamentals or {}
        vd = valuation_data or {}
        industry_avgs = vd.get("industry_averages", {})

        price_val = overview.get("price", f.get("Price", "N/A"))
        results["valuation"] = call_llm(
            VALUATION_PROMPT.format(
                ticker=ticker, instrument_type=instrument_type,
                price=price_val,
                target_price=f.get("Target Price", "N/A"),
                industry=f.get("Industry", "N/A"),
                pe_ratio=f.get("P/E", "N/A"),
                forward_pe_ratio=f.get("Forward P/E", "N/A"),
                peg_ratio=f.get("PEG", "N/A"),
                ps_ratio=f.get("P/S", "N/A"),
                pb_ratio=f.get("P/B", "N/A"),
                pc_ratio=f.get("P/C", "N/A"),
                pfcf_ratio=f.get("P/FCF", "N/A"),
                industry_pe_avg=industry_avgs.get("P/E", "N/A"),
                industry_peg_avg=industry_avgs.get("PEG", "N/A"),
                industry_ps_avg=industry_avgs.get("P/S", "N/A"),
                industry_pb_avg=industry_avgs.get("P/B", "N/A"),
                industry_pc_avg=industry_avgs.get("P/C", "N/A"),
                industry_pfcf_avg=industry_avgs.get("P/FCF", "N/A"),
                eps_this_y=f.get("EPS this Y", "N/A"),
                eps_next_y=f.get("EPS next Y", "N/A"),
                eps_past_5y=f.get("EPS past 5Y", "N/A"),
                eps_next_5y=f.get("EPS next 5Y", "N/A"),
                sales_past_5y=f.get("Sales past 5Y", "N/A"),
            ), model=model,
        )

        # ── Fundamentals ──
        progress.progress(70, text="🤖 Analizando fundamentales...")
        results["fundamentals"] = call_llm(
            FUNDAMENTALS_PROMPT.format(
                ticker=ticker,
                insider_ownership=f.get("Insider Own", "N/A"),
                institutions_own=f.get("Inst Own", "N/A"),
                institutions_trans=f.get("Inst Trans", "N/A"),
                short_float=f.get("Short Float", "N/A"),
                short_ratio=f.get("Short Ratio", "N/A"),
                short_interest=f.get("Short Interest", "N/A"),
                options_shorts=f"{f.get('Optionable', 'N/A')} / {f.get('Shortable', 'N/A')}",
                shares_outstand=f.get("Shs Outstand", "N/A"),
                market_cap=f.get("Market Cap", "N/A"),
                beta=f.get("Beta", "N/A"),
                earnings_date=f.get("Earnings", "N/A"),
                gross_margin=f.get("Gross Margin", "N/A"),
                oper_margin=f.get("Oper. Margin", "N/A"),
                profit_margin=f.get("Profit Margin", "N/A"),
                roa=f.get("ROA", "N/A"),
                roe=f.get("ROE", "N/A"),
                roic=f.get("ROI", "N/A"),
                income=f.get("Income", "N/A"),
                sales=f.get("Sales", "N/A"),
                sales_past_5y=f.get("Sales past 5Y", "N/A"),
                eps_surprise=f.get("EPS Q/Q", "N/A"),
                sales_surprise=f.get("Sales Q/Q", "N/A"),
                book_sh=f.get("Book/sh", "N/A"),
                cash_sh=f.get("Cash/sh", "N/A"),
                quick_ratio=f.get("Quick Ratio", "N/A"),
                current_ratio=f.get("Current Ratio", "N/A"),
                debt_eq=f.get("Debt/Eq", "N/A"),
                lt_debt_eq=f.get("LT Debt/Eq", "N/A"),
            ), model=model,
        )

        # ── Financial Statements AI ──
        if statements:
            progress.progress(73, text="🤖 Analizando estados financieros...")
            for key, df in statements.items():
                try:
                    df_json = df.head(20).to_json(orient="records", indent=2)
                    df_results[key] = call_llm(
                        FINANCIAL_STATEMENT_PROMPT.format(df_input=df_json), model=model,
                    )
                except Exception:
                    df_results[key] = ""

        # ── Technicals ──
        progress.progress(77, text="🤖 Generando análisis técnico...")
        ind = ta_data.get("indicators", {})
        detected = ta_data.get("patterns", [])
        pattern_text = "\n".join([f"• {p.get('type', 'Unknown')}: {p.get('description', json.dumps(p))}" for p in detected]) if detected else "No significant patterns detected."
        ta_summary_text = "\n".join([f"• {k}: {v}" for k, v in ind.items()])

        results["technicals"] = call_llm(
            TECHNICAL_PROMPT.format(
                ticker=ticker,
                sma_20=f.get("SMA20", "N/A"),
                sma_50=f.get("SMA50", "N/A"),
                sma_200=f.get("SMA200", "N/A"),
                rsi_14=f.get("RSI (14)", ind.get("RSI", "N/A")),
                atr_14=ind.get("ATR", "N/A"),
                volatility_w=f.get("Volatility W", "N/A"),
                volatility_m=f.get("Volatility M", "N/A"),
                volume=f.get("Volume", "N/A"),
                rel_volume=f.get("Rel Volume", "N/A"),
                avg_volume=f.get("Avg Volume", "N/A"),
                low=f.get("52W Low", "N/A"),
                high=f.get("52W High", "N/A"),
                dist_high=f.get("from 52W High", "N/A"),
                dist_low=f.get("from 52W Low", "N/A"),
                detected_patterns=pattern_text,
                ta_summary=ta_summary_text,
            ), model=model,
        )

        # ── Risk ──
        progress.progress(82, text="🤖 Analizando perfil de riesgo...")
        results["risk"] = call_llm(
            RISK_METRICS_PROMPT.format(
                ticker=ticker, benchmark=benchmark,
                annualized_return=risk_metrics["annualized_return"],
                vol=risk_metrics["volatility"],
                downside_vol=risk_metrics["downside_volatility"],
                max_dd=risk_metrics["max_drawdown"],
                beta=risk_metrics["beta"],
                sharpe_ratio=risk_metrics["sharpe_ratio"],
                sortino_ratio=risk_metrics["sortino_ratio"],
                treynor_ratio=risk_metrics["treynor_ratio"],
                calmar_ratio=risk_metrics["calmar_ratio"],
                rfr=rfr,
            ), model=model,
        )

        # ── Monte Carlo ──
        progress.progress(86, text="🤖 Interpretando resultados Monte Carlo...")
        ps = mc_data["percentile_summary"]
        ps_text = "\n".join([f"• {k}: ${v:,.2f}" for k, v in ps.items()])
        results["montecarlo"] = call_llm(
            MONTECARLO_PROMPT.format(
                ticker=ticker,
                last_price=mc_data["last_price"],
                n_simulations=mc_data["n_simulations"],
                days=mc_data["days"],
                percentile_summary=ps_text,
            ), model=model,
        )

        # ── Options ──
        if options_analysis:
            progress.progress(90, text="🤖 Analizando posicionamiento de opciones...")
            oa = options_analysis
            calls_json = oa["calls"].head(15).to_json(orient="records", indent=2)
            puts_json = oa["puts"].head(15).to_json(orient="records", indent=2)
            options_data_str = f"CALLS:\n{calls_json}\n\nPUTS:\n{puts_json}"

            results["options"] = call_llm(
                OPTIONS_PROMPT.format(
                    ticker=ticker,
                    spot_price=oa["spot_price"],
                    expiration=oa["expiration"],
                    days_to_expiry=oa["days_to_expiry"],
                    iv_format=oa["iv_format"],
                    expected_move_return=oa["expected_move_return"],
                    range_lower_bound=oa["range_lower_bound"],
                    range_upper_bound=oa["range_upper_bound"],
                    max_pain_strike=oa["max_pain_strike"],
                    max_pain_distance=oa["max_pain_distance"],
                    put_call_oi_ratio=oa["put_call_oi_ratio"],
                    put_call_volume_ratio=oa["put_call_volume_ratio"],
                    total_call_oi=oa["total_call_oi"],
                    total_put_oi=oa["total_put_oi"],
                    options_data=options_data_str,
                ), model=model,
            )

        # ── ETF Analysis ──
        if instrument_type == INSTRUMENT_ETF:
            progress.progress(92, text="🤖 Analizando estructura del ETF...")
            holdings_text = json.dumps(overview.get("top_holdings", [])[:10], indent=2)
            sw = overview.get("sector_weights", {})
            sw_text = json.dumps(sw, indent=2) if sw else "N/A"

            results["etf"] = call_llm(
                ETF_ANALYSIS_PROMPT.format(
                    ticker=ticker,
                    name=overview.get("name", ticker),
                    category=overview.get("category", "N/A"),
                    total_assets=overview.get("total_assets", "N/A"),
                    expense_ratio=overview.get("expense_ratio", "N/A"),
                    yield_val=overview.get("yield", "N/A"),
                    ytd_return=overview.get("ytd_return", "N/A"),
                    three_year_return=overview.get("three_year_return", "N/A"),
                    five_year_return=overview.get("five_year_return", "N/A"),
                    beta_3y=overview.get("beta_3y", "N/A"),
                    top_holdings=holdings_text,
                    sector_weights=sw_text,
                ), model=model,
            )

        # ── Conclusion ──
        progress.progress(95, text="🤖 Generando conclusión de inversión...")
        conclusion = call_llm(
            CONCLUSION_PROMPT.format(
                company=overview.get("name", ticker),
                ticker=ticker,
                instrument_type=instrument_type,
                price=current_price_data,
                news=results.get("news", "N/A"),
                valuation=results.get("valuation", "N/A"),
                df_i_j=df_results.get("income", "N/A"),
                df_b_j=df_results.get("balance", "N/A"),
                df_c_j=df_results.get("cashflow", "N/A"),
                fundamentals=results.get("fundamentals", "N/A"),
                risk_metrics=results.get("risk", "N/A"),
                gmm_montecarlo=results.get("montecarlo", "N/A"),
                options_short_term_analysis=results.get("options", "N/A"),
                technicals=results.get("technicals", "N/A"),
            ), model=model,
        )

        # ── PDF Report ──
        progress.progress(98, text="📄 Generando reporte PDF...")
        try:
            import plotly.io as pio
            import tempfile
            chart_paths = []
            
            # Save the main charts as temporary images for the PDF
            for key in ["technical", "montecarlo", "drawdown"]:
                if key in charts and charts[key]:
                    try:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        pio.write_image(charts[key], tmp.name, format="png", width=1200, height=800, scale=2)
                        chart_paths.append(tmp.name)
                    except Exception:
                        continue

            results["current_price_data"] = current_price_data
            
            pdf_path = generate_report(
                ticker=ticker,
                overview=overview,
                results=results,
                df_results=df_results,
                conclusion=conclusion,
                statements=statements,
                chart_paths=chart_paths
            )
        except Exception as e:
            pdf_path = None
            st.warning(f"PDF generation failed: {e}")
            traceback.print_exc()

        progress.progress(100, text="✅ ¡Análisis completado!")

        # ── Store everything in session state ──
        st.session_state.update({
            "analysis_complete": True,
            "analysis_running": False,
            "overview": overview,
            "instrument_type": instrument_type,
            "ta_data": ta_data,
            "risk_metrics": risk_metrics,
            "mc_data": mc_data,
            "options_analysis": options_analysis,
            "results": results,
            "df_results": df_results,
            "statements": statements,
            "charts": charts,
            "conclusion": conclusion,
            "pdf_path": pdf_path,
        })

    except Exception as e:
        progress.empty()
        st.error(f"❌ Analysis failed: {str(e)}")
        st.expander("Error Details").code(traceback.format_exc())
        st.session_state["analysis_running"] = False


# ═══════════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Sidebar
    config = render_sidebar()

    # Title
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:30px;">
            <h1 style="background: linear-gradient(135deg, #00D4AA, #6C5CE7);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                font-size: 2.5rem; font-weight: 800; margin-bottom: 5px;">
                📊 Stock AI Agent
            </h1>
            <p style="color: #888; font-size: 0.95rem;">
                Análisis de Grado Institucional potenciado por IA
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Run Analysis ──
    if config["run_analysis"]:
        if not get_api_key():
            st.error("⚠️ OpenAI API key not configured. Set it as an environment variable `OPENAI_API_KEY` or in `.streamlit/secrets.toml`.")
            return
        st.session_state["analysis_running"] = True
        st.session_state["analysis_complete"] = False
        run_full_analysis(config["ticker"], config)

    # ── Display Results ──
    if st.session_state.get("analysis_complete"):
        overview = st.session_state["overview"]
        instrument_type = st.session_state["instrument_type"]
        results = st.session_state["results"]
        df_results = st.session_state["df_results"]
        statements = st.session_state["statements"]
        charts = st.session_state["charts"]
        ta_data = st.session_state["ta_data"]
        risk_metrics = st.session_state["risk_metrics"]
        mc_data = st.session_state["mc_data"]
        options_analysis = st.session_state["options_analysis"]
        conclusion = st.session_state["conclusion"]
        pdf_path = st.session_state.get("pdf_path")

        # Tabs
        tab_names = [
            "📋 Resumen", "💰 Financiero", "📈 Técnico",
            "⚠️ Riesgo y MC", "🎯 Opciones", "📊 Ganancias",
            "💸 Análisis DCF", "📄 Reporte Final"
        ]
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_names)

        with tab1:
            etf_holdings = {
                "top_holdings": overview.get("top_holdings", []),
                "sector_weights": overview.get("sector_weights", {}),
            } if instrument_type == INSTRUMENT_ETF else None

            render_overview_tab(
                overview=overview,
                news_analysis=results.get("news", ""),
                market_analysis=results.get("market", ""),
                macro_analysis=results.get("macro", ""),
                etf_holdings=etf_holdings,
            )

        with tab2:
            render_financials_tab(
                instrument_type=instrument_type,
                valuation_analysis=results.get("valuation", ""),
                fundamentals_analysis=results.get("fundamentals", ""),
                statements=statements,
                statement_analyses=df_results,
                etf_analysis=results.get("etf"),
            )

        with tab3:
            render_technicals_tab(
                ta_data=ta_data,
                chart=charts.get("technical"),
                technical_analysis=results.get("technicals", ""),
            )

        with tab4:
            render_risk_tab(
                risk_metrics=risk_metrics,
                risk_analysis=results.get("risk", ""),
                mc_data=mc_data,
                mc_chart=charts.get("montecarlo"),
                mc_analysis=results.get("montecarlo", ""),
                dd_chart=charts.get("drawdown"),
            )

        with tab5:
            render_options_tab(
                ticker=config["ticker"],
                options_analysis=options_analysis,
                options_chart=charts.get("options"),
                options_ai_analysis=results.get("options", ""),
            )

        with tab6:
            render_earnings_tab(
                ticker=config["ticker"],
                earnings_data=st.session_state.get("earnings_data", {}),
            )

        with tab7:
            render_dcf_tab(
                ticker=config["ticker"],
                overview=overview,
                fundamentals=df_results.get("fundamentals_raw", {}),
            )

        with tab8:
            render_report_tab(
                conclusion=conclusion,
                pdf_path=pdf_path,
            )

    else:
        # Welcome state
        st.markdown(
            """
            <div style="text-align:center; padding: 60px 20px;">
                <div style="font-size: 4rem; margin-bottom: 20px;">🚀</div>
                <h2 style="color: #ddd; margin-bottom: 10px;">Listo para Analizar</h2>
                <p style="color: #888; max-width: 500px; margin: 0 auto;">
                    Ingrese un símbolo (ticker) en la barra lateral y haga clic en 
                    <strong style="color: #00D4AA;">Ejecutar Análisis Completo</strong> para comenzar.
                    <br><br>
                    Soportamos <strong>Acciones</strong>, <strong>ETFs</strong> y <strong>REITs</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
