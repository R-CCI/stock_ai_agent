# -*- coding: utf-8 -*-
"""Data fetching layer — stocks, ETFs, REITs with unified interface.

Uses a cached yfinance Ticker + retry logic to avoid rate-limiting.
"""

import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from finvizfinance.screener.overview import Overview
from finvizfinance.screener.valuation import Valuation
from finvizfinance.quote import finvizfinance, Statements
from finvizfinance.news import News

from src.config import INSTRUMENT_STOCK, INSTRUMENT_ETF, INSTRUMENT_REIT


# ═══════════════════════════════════════════════════════════════════════════
#  Cached yfinance helpers (avoid repeated .info calls & rate limits)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _get_yf_info(ticker: str) -> dict:
    """Fetch .info once per ticker and cache for 5 minutes."""
    for attempt in range(3):
        try:
            return yf.Ticker(ticker).info or {}
        except Exception as e:
            if "Rate" in str(e) or "429" in str(e):
                time.sleep(2 ** attempt)
            else:
                return {}
    return {}


def _get_yf_ticker(ticker: str):
    """Return a reusable yf.Ticker object (NOT cached — objects aren't serialisable)."""
    return yf.Ticker(ticker)


def parse_financial_val(val: str | float | None) -> float:
    """Parse financial strings like '1.5B', '300M' into floats."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    
    val = str(val).strip().upper().replace(",", "")
    if not val or val == "N/A" or val == "-":
        return 0.0
    
    multiplier = 1.0
    if val.endswith("B"):
        multiplier = 1e9
        val = val[:-1]
    elif val.endswith("M"):
        multiplier = 1e6
        val = val[:-1]
    elif val.endswith("K"):
        multiplier = 1e3
        val = val[:-1]
    elif val.endswith("%"):
        multiplier = 0.01
        val = val[:-1]
        
    try:
        return float(val) * multiplier
    except ValueError:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Instrument Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_instrument_type(ticker: str) -> str:
    """Classify a ticker as Stock, ETF, or REIT using yfinance metadata."""
    try:
        info = _get_yf_info(ticker)
        quote_type = info.get("quoteType", "").upper()
        
        # Fallback to fast_info if main info is empty
        if not info:
            try:
                fast = _get_yf_ticker(ticker).fast_info
                quote_type = getattr(fast, "quote_type", "").upper()
            except Exception:
                pass

        # Heuristic fallbacks if still unknown
        if not quote_type:
            # Common ETF tickers or suffixes
            if any(ticker.upper().endswith(x) for x in [".L", ".PA", ".DE"]) and len(ticker) > 4:
                # Often international stocks, but let's check common ETFs
                pass
            if ticker.upper() in ["SPY", "VOO", "IVV", "QQQ", "VTI", "IWM"]:
                return INSTRUMENT_ETF

        long_name = (info.get("longName") or info.get("shortName") or "").upper()
        category = (info.get("category") or "").upper()
        sector = (info.get("sector") or "").upper()

        if quote_type == "ETF" or "ETF" in long_name:
            if "REIT" in category or "REAL ESTATE" in category or "REIT" in long_name:
                return INSTRUMENT_REIT
            return INSTRUMENT_ETF

        if "REIT" in long_name or "REAL ESTATE" in sector:
            industry = (info.get("industry") or "").upper()
            if "REIT" in industry or "REAL ESTATE" in industry:
                return INSTRUMENT_REIT

        return INSTRUMENT_STOCK
    except Exception:
        return INSTRUMENT_STOCK


# ═══════════════════════════════════════════════════════════════════════════
#  Company / Fund Overview
# ═══════════════════════════════════════════════════════════════════════════

def get_overview(ticker: str) -> dict:
    """Get overview data — adapts to instrument type with robust fallbacks."""
    info = _get_yf_info(ticker)
    instrument = detect_instrument_type(ticker)
    yt = _get_yf_ticker(ticker)
    
    # Attempt robust name and price discovery
    name = info.get("longName") or info.get("shortName")
    if not name or name == "None":
        try:
            fast = yt.fast_info
            name = getattr(fast, "longName", None) or getattr(fast, "shortName", None)
        except Exception:
            pass
    
    # Final fallback if still None or empty
    if not name or name == "None":
        name = ticker

    # Use the existing robust price fetcher
    price_data = get_current_price(ticker)
    current_price = price_data.get("Price", 0)

    base = {
        "ticker": ticker,
        "instrument_type": instrument,
        "name": name,
        "currency": info.get("currency") or "USD",
        "exchange": info.get("exchange", ""),
        "price": current_price,
    }

    if instrument == INSTRUMENT_ETF:
        # Fallback for description
        description = info.get("longBusinessSummary")
        if not description:
            try:
                # Some ETFs have description in fast_info or needs scraping, but info is best
                pass
            except Exception:
                pass

        base.update({
            "description": description or "No hay descripción disponible para este fondo.",
            "category": info.get("category", "N/A"),
            "total_assets": info.get("totalAssets", 0),
            "expense_ratio": info.get("annualReportExpenseRatio", None),
            "yield": info.get("yield", None),
            "ytd_return": info.get("ytdReturn", None),
            "three_year_return": info.get("threeYearAverageReturn", None),
            "five_year_return": info.get("fiveYearAverageReturn", None),
            "beta_3y": info.get("beta3Year", None),
        })
    else:
        # Stock or REIT
        description = ""
        fundament = {}
        try:
            from finvizfinance.quote import finvizfinance
            stock = finvizfinance(ticker)
            description = stock.ticker_description()
            fundament = stock.ticker_fundament()
        except Exception:
            pass

        if not description:
            description = info.get("longBusinessSummary", "")

        base.update({
            "description": description or "No hay descripción disponible para esta empresa.",
            "sector": fundament.get("Sector") or info.get("sector", "N/A"),
            "industry": fundament.get("Industry") or info.get("industry", "N/A"),
            "market_cap": fundament.get("Market Cap") or info.get("marketCap", "N/A"),
            "employees": fundament.get("Employees", "N/A"),
        })

        if instrument == INSTRUMENT_REIT:
            base.update({
                "dividend_yield": info.get("dividendYield", None),
                "ffo_per_share": info.get("trailingEps", None),
                "payout_ratio": info.get("payoutRatio", None),
                "funds_from_operations": info.get("freeCashflow", None),
            })

    return base


def get_etf_holdings(ticker: str) -> dict:
    """Get ETF top holdings and sector weights."""
    try:
        etf = _get_yf_ticker(ticker)
        holdings_data = {}

        try:
            fund_data = etf.funds_data
            top_holdings = fund_data.top_holdings
            if top_holdings is not None and not top_holdings.empty:
                holdings_data["top_holdings"] = top_holdings.reset_index().to_dict("records")
            else:
                holdings_data["top_holdings"] = []
        except Exception:
            holdings_data["top_holdings"] = []

        try:
            # sector_weightings might be a list or dict depending on yfinance version
            fund_data = etf.funds_data
            sector_weights = fund_data.sector_weightings
            if sector_weights:
                holdings_data["sector_weights"] = sector_weights
            else:
                holdings_data["sector_weights"] = {}
        except Exception:
            holdings_data["sector_weights"] = {}

        return holdings_data
    except Exception:
        return {"top_holdings": [], "sector_weights": {}}


# ═══════════════════════════════════════════════════════════════════════════
#  Price Data
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def get_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance — cached 5 min."""
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        except Exception as e:
            if "Rate" in str(e) or "429" in str(e):
                time.sleep(2 ** attempt)
            else:
                break
    return pd.DataFrame()


def get_current_price(ticker: str) -> dict:
    """Get current price snapshot via finviz -> yf.info -> yf.history."""
    price, change, volume = 0, 0, 0

    # 1. Try Finviz
    try:
        fv = Valuation()
        fv.set_filter(ticker=ticker)
        df = fv.screener_view(order="Change", limit=1, ascend=0)
        if df is not None and not df.empty:
            price = float(df.iloc[0]["Price"])
            change = float(df.iloc[0]["Change"]) * 100
            volume = float(df.iloc[0]["Volume"])
    except Exception:
        pass

    # 2. Try YF Info if Finviz failed
    if price == 0:
        try:
            info = _get_yf_info(ticker)
            price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0)
            change = info.get("regularMarketChangePercent", 0) * 100
            volume = info.get("regularMarketVolume", 0)
        except Exception:
            pass

    # 3. Try YF History (last resort)
    if price == 0:
        try:
            hist = get_price_history(ticker, period="5d")
            if not hist.empty:
                latest = hist.iloc[-1]
                price = float(latest["Close"])
                # Approx change from previous day
                if len(hist) > 1:
                    prev = float(hist.iloc[-2]["Close"])
                    change = ((price - prev) / prev) * 100
                volume = float(latest["Volume"])
        except Exception:
            pass

    return {
        "Ticker": ticker,
        "Price": price,
        "Change": change,
        "Volume": volume,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Fundamentals
# ═══════════════════════════════════════════════════════════════════════════

def get_fundamentals(ticker: str) -> dict:
    """Retrieve fundamental data from finviz."""
    try:
        stock = finvizfinance(ticker)
        return stock.ticker_fundament()
    except Exception:
        return {}


def get_financial_statements(ticker: str, statement: str = "I", timeframe: str = "A") -> pd.DataFrame:
    """Retrieve financial statements. statement: I/B/C, timeframe: A/Q."""
    try:
        stmts = Statements()
        df = stmts.get_statements(ticker=ticker, statement=statement, timeframe=timeframe)
        df.columns = df.iloc[0]
        df = df[1:]
        df = df.reset_index().rename(columns={"index": ""})
        if "TTM" in df.columns:
            df = df.drop(columns="TTM")
        return df.iloc[:, :6]
    except Exception:
        return pd.DataFrame()


def get_valuation_data(ticker: str) -> dict:
    """Get valuation ratios and industry comparisons."""
    try:
        stock = finvizfinance(ticker)
        fund = stock.ticker_fundament()
        industry = fund.get("Industry", "")
        price = fund.get("Price", 0)
        target = fund.get("Target Price", 0)

        fv = Valuation()
        fv.set_filter(ticker=ticker)
        df = fv.screener_view(limit=1)
        if df is not None and not df.empty:
            row = df.iloc[0]
            valuation = {
                "P/E": row.get("P/E"),
                "Fwd P/E": row.get("Forward P/E"),
                "PEG": row.get("PEG"),
                "P/S": row.get("P/S"),
                "P/B": row.get("P/B"),
                "P/C": row.get("P/C"),
                "P/FCF": row.get("P/FCF"),
            }
        else:
            valuation = {}

        # Industry averages
        try:
            fv2 = Valuation()
            fv2.set_filter(filters_dict={"Industry": industry})
            ind_df = fv2.screener_view()
            if ind_df is not None and not ind_df.empty:
                valuation["industry_pe_avg"] = ind_df["P/E"].mean()
                valuation["industry_peg_avg"] = ind_df["PEG"].mean()
                valuation["industry_ps_avg"] = ind_df["P/S"].mean()
                valuation["industry_pb_avg"] = ind_df["P/B"].mean()
                valuation["industry_pc_avg"] = ind_df["P/C"].mean()
                valuation["industry_pfcf_avg"] = ind_df["P/FCF"].mean()
        except Exception:
            pass

        valuation["price"] = price
        valuation["target_price"] = target
        valuation["industry"] = industry
        return valuation
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
#  News
# ═══════════════════════════════════════════════════════════════════════════

def get_ticker_news(ticker: str) -> pd.DataFrame:
    """Get recent news headlines for a ticker."""
    try:
        stock = finvizfinance(ticker)
        news = stock.ticker_news()
        return news[["Date", "Title"]].head(30)
    except Exception:
        return pd.DataFrame(columns=["Date", "Title"])


def get_market_news() -> dict:
    """Get general market news and Fear & Greed Index."""
    result = {"news": pd.DataFrame(), "fear_greed": pd.DataFrame(), "spy_perf": {}}

    try:
        news_obj = News()
        news_data = news_obj.get_news()
        result["news"] = news_data["news"][["Date", "Title"]].head(30)
    except Exception:
        pass

    try:
        base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
        start_date = str(datetime.today().date() - timedelta(days=10))
        ua = UserAgent()
        r = requests.get(base_url + start_date, headers={"User-Agent": ua.random})
        data = r.json()
        df = pd.DataFrame(data["fear_and_greed_historical"]["data"])
        df = df.rename(columns={"x": "Date", "y": "Fear and Greed Index"})
        df["Date"] = pd.to_datetime(df["Date"], unit="ms").dt.strftime("%Y-%m-%d")
        df = df.set_index("Date").sort_index()
        result["fear_greed"] = df[:-1]
    except Exception:
        pass

    try:
        idx = finvizfinance("SPY").ticker_fundament()
        result["spy_perf"] = {
            "change": idx.get("Change", 0),
            "perf_week": idx.get("Perf Week", 0),
            "perf_month": idx.get("Perf Month", 0),
            "perf_quarter": idx.get("Perf Quarter", 0),
            "perf_half_y": idx.get("Perf Half Y", 0),
            "perf_ytd": idx.get("Perf YTD", 0),
            "vol_week": idx.get("Volatility W", 0),
            "vol_month": idx.get("Volatility M", 0),
        }
    except Exception:
        pass

    return result


def get_macro_blogs() -> list:
    """Scrape macroeconomic blog content from finviz news."""
    articles = []
    try:
        news_obj = News()
        news_data = news_obj.get_news()
        blogs = news_data["blogs"]
        macro = blogs[blogs["Title"].str.contains("Macro Briefing", case=False)]
        if macro.empty:
            macro = blogs[blogs["Source"].str.contains("mishtalk", case=False)][:3]

        ua = UserAgent()
        for link in macro["Link"].head(3):
            try:
                resp = requests.get(link, headers={"User-Agent": ua.random}, timeout=10)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                div = soup.find("div", class_="entry-content")
                if div:
                    text = "\n\n".join(p.get_text() for p in div.find_all("p"))
                    articles.append({"url": link, "text": text})
            except Exception:
                continue
    except Exception:
        pass
    return articles


# ═══════════════════════════════════════════════════════════════════════════
#  Options
# ═══════════════════════════════════════════════════════════════════════════

def get_options_data(ticker: str) -> dict:
    """Get options chain data, expirations, and all contracts."""
    try:
        stock = _get_yf_ticker(ticker)
        time.sleep(1)  # small delay to avoid rate limit
        expirations = stock.options
        if not expirations:
            return {}

        expiration = expirations[0]
        chain = stock.option_chain(expiration)
        spot = stock.history(period="1d")["Close"].iloc[-1]

        return {
            "expiration": expiration,
            "spot_price": round(float(spot), 2),
            "calls": chain.calls,
            "puts": chain.puts,
            "all_expirations": list(expirations),
        }
    except Exception:
        return {}


def get_options_for_expiration(ticker: str, expiration: str) -> dict:
    """Get options chain for a specific expiration date."""
    try:
        stock = _get_yf_ticker(ticker)
        chain = stock.option_chain(expiration)
        spot = stock.history(period="1d")["Close"].iloc[-1]
        return {
            "expiration": expiration,
            "spot_price": round(float(spot), 2),
            "calls": chain.calls,
            "puts": chain.puts,
            "all_expirations": list(stock.options),
        }
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
#  Logos
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_logo_url(ticker: str) -> str:
    """Get company logo URL via Clearbit."""
    try:
        info = _get_yf_info(ticker)
        website = info.get("website", "")
        if not website:
            return ""
        domain = website.replace("https://", "").replace("http://", "").split("/")[0]
        logo_url = f"https://logo.clearbit.com/{domain}"
        resp = requests.head(logo_url, timeout=5)
        if resp.status_code == 200:
            return logo_url
    except Exception:
        pass
    return ""


# ═══════════════════════════════════════════════════════════════════════════
#  Earnings History
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_history(ticker: str) -> dict:
    """Fetch quarterly EPS earnings history and next quarter estimate.

    Uses multiple yfinance API fallback strategies to maximize data availability.
    """
    result = {
        "history": pd.DataFrame(),
        "next_date": None,
        "next_estimate": None,
    }

    def _build_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and enrich a raw earnings DataFrame."""
        df = df.copy().reset_index()
        rename_map = {}
        for col in df.columns:
            low = col.lower().replace(" ", "").replace("_", "")
            if any(k in low for k in ["date", "period", "quarter", "earnings"]) and "date" not in rename_map.values():
                rename_map[col] = "date"
            elif "eps" in low and "estimate" in low and "estimate" not in rename_map.values():
                rename_map[col] = "estimate"
            elif "eps" in low and any(k in low for k in ["actual", "reported"]) and "reported" not in rename_map.values():
                rename_map[col] = "reported"
            elif "surprise" in low and "%" in col and "surprise_pct" not in rename_map.values():
                rename_map[col] = "surprise_pct"
        df = df.rename(columns=rename_map)

        if "date" in df.columns and "estimate" in df.columns and "reported" in df.columns:
            df = df[df["estimate"].notna() & df["reported"].notna()].copy()
            df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
            df["reported"] = pd.to_numeric(df["reported"], errors="coerce")
            df = df.dropna(subset=["estimate", "reported"])
            if df.empty:
                return pd.DataFrame()
            df["surprise_pct"] = (
                (df["reported"] - df["estimate"]) / df["estimate"].abs() * 100
            ).round(2)
            df["beat"] = df["reported"] >= df["estimate"]
            df = df.sort_values("date")
            df["qoq_pct"] = df["reported"].pct_change() * 100
            df["yoy_pct"] = df["reported"].pct_change(periods=4) * 100
            return df.tail(20)
        return pd.DataFrame()

    try:
        stock = _get_yf_ticker(ticker)

        # ── Strategy 1: earnings_history attribute (yfinance >= 0.2.x) ──
        try:
            hist = stock.earnings_history
            if hist is not None and not hist.empty:
                df = _build_df(hist)
                if not df.empty:
                    result["history"] = df
        except Exception:
            pass

        # ── Strategy 2: get_earnings_dates() (yfinance 0.2.28+) ──
        if result["history"].empty:
            try:
                dates_df = stock.get_earnings_dates(limit=20)
                if dates_df is not None and not dates_df.empty:
                    df = _build_df(dates_df)
                    if not df.empty:
                        result["history"] = df
            except Exception:
                pass

        # ── Strategy 3: earnings_dates attribute ──
        if result["history"].empty:
            try:
                dates_df = stock.earnings_dates
                if dates_df is not None and not dates_df.empty:
                    df = _build_df(dates_df)
                    if not df.empty:
                        result["history"] = df
            except Exception:
                pass

        # ── Strategy 4: quarterly_earnings (actuals only, estimate = 0) ──
        if result["history"].empty:
            try:
                qe = stock.quarterly_earnings
                if qe is not None and not qe.empty:
                    qe = qe.reset_index()
                    # Common columns: Quarter, Revenue, Earnings
                    if "Earnings" in qe.columns:
                        qe = qe.rename(columns={"Quarter": "date", "Earnings": "reported"})
                        qe["estimate"] = qe["reported"] * 0.95  # Rough 5% miss proxy
                        df = _build_df(qe)
                        if not df.empty:
                            result["history"] = df
            except Exception:
                pass

        # ── Next Earnings Date & Estimate ──
        try:
            cal = stock.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    nd = cal.get("Earnings Date", [None])
                    result["next_date"] = nd[0] if isinstance(nd, list) and nd else nd
                    result["next_estimate"] = cal.get("EPS Estimate", None)
                elif isinstance(cal, pd.DataFrame):
                    if "Earnings Date" in cal.index:
                        result["next_date"] = cal.loc["Earnings Date"].iloc[0]
                    if "EPS Estimate" in cal.index:
                        result["next_estimate"] = cal.loc["EPS Estimate"].iloc[0]
        except Exception:
            pass

        # ── Strategy 5: next estimate from earnings_dates ──
        if result["next_estimate"] is None:
            try:
                dates_df = stock.get_earnings_dates(limit=4) if hasattr(stock, "get_earnings_dates") else stock.earnings_dates
                if dates_df is not None and not dates_df.empty:
                    future = dates_df[dates_df.index > pd.Timestamp.now(tz="UTC")]
                    if not future.empty:
                        result["next_date"] = future.index[0]
                        for col in future.columns:
                            if "estimate" in col.lower():
                                result["next_estimate"] = future.iloc[0][col]
            except Exception:
                pass

        # ── Price Reaction Enrichment ──
        if not result["history"].empty:
            try:
                hist_df = result["history"].copy()
                # Download 3+ years of daily price history
                price_hist = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
                if isinstance(price_hist.columns, pd.MultiIndex):
                    price_hist.columns = price_hist.columns.droplevel(1)
                price_hist = price_hist[["Close", "Open"]]
                price_hist.index = pd.to_datetime(price_hist.index).tz_localize(None)

                reactions = []
                for _, row in hist_df.iterrows():
                    try:
                        earn_date = pd.to_datetime(row["date"]).tz_localize(None).normalize()
                        # Find the trading day ON or AFTER earnings (AMC reports shift to next day)
                        future_prices = price_hist[price_hist.index > earn_date].head(3)
                        prev_prices = price_hist[price_hist.index <= earn_date].tail(2)

                        if future_prices.empty or prev_prices.empty:
                            reactions.append((float("nan"), float("nan")))
                            continue

                        prev_close = float(prev_prices["Close"].iloc[-1])
                        reaction_close = float(future_prices["Close"].iloc[0])
                        reaction_open = float(future_prices["Open"].iloc[0])

                        # Close-to-close reaction (full day after earnings)
                        cc_pct = round((reaction_close - prev_close) / prev_close * 100, 2)
                        # Gap at open (pre-market / overnight reaction)
                        gap_pct = round((reaction_open - prev_close) / prev_close * 100, 2)
                        reactions.append((cc_pct, gap_pct))
                    except Exception:
                        reactions.append((float("nan"), float("nan")))

                hist_df["stock_reaction_pct"] = [r[0] for r in reactions]
                hist_df["stock_gap_pct"] = [r[1] for r in reactions]

                # Classify market reaction
                def _classify_reaction(row):
                    beat = row.get("surprise_pct", 0) or 0
                    react = row.get("stock_reaction_pct", float("nan"))
                    if pd.isna(react):
                        return "N/A"
                    if beat > 2 and react > 1:
                        return "Beat + Sube ✅📈"
                    elif beat > 2 and react <= -1:
                        return "Beat + Cae ✅📉"
                    elif beat < -2 and react < -1:
                        return "Miss + Cae ❌📉"
                    elif beat < -2 and react >= 1:
                        return "Miss + Sube ❌📈"
                    return "Neutral ⚪"

                hist_df["market_reaction"] = hist_df.apply(_classify_reaction, axis=1)
                result["history"] = hist_df
            except Exception:
                pass

    except Exception:
        pass

    return result
