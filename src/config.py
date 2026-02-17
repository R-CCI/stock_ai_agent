# -*- coding: utf-8 -*-
"""Financial analysis — risk metrics, Monte Carlo, options, valuation logic."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm
from statsmodels import regression

from src.config import get_risk_free_rate


# ═══════════════════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def _linreg(x, y):
    """Simple linear regression returning (alpha, beta)."""
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    x = x[:, 1]
    return model.params[0], model.params[1]


def compute_drawdown(return_series: pd.Series) -> pd.DataFrame:
    """Compute wealth index, previous peaks, and drawdown series."""
    if isinstance(return_series, pd.DataFrame):
        return_series = return_series.iloc[:, 0]
    wealth = 1000 * (1 + return_series).cumprod()
    peaks = wealth.cummax()
    dd = (wealth - peaks) / peaks
    return pd.DataFrame({
        "Wealth": wealth.values,
        "Previous Peak": peaks.values,
        "Drawdown": dd.values,
    }, index=return_series.index)


def _gmm_pdf(x, weights, means, covariances, n_components):
    """Gaussian Mixture Model probability density."""
    return np.sum([
        weights[i] * norm.pdf(x, means[i], np.sqrt(covariances[i]))
        for i in range(n_components)
    ], axis=0)


def _project_percentiles(df):
    """Add percentile columns to simulation DataFrame."""
    sim_cols = [c for c in df.columns if c.startswith("S")]
    for p in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        df[f"P{int(p * 100)}"] = df[sim_cols].quantile(p, axis=1)
    return df


def _black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility_call(S, K, T, r, market_price):
    """Compute implied vol for a call option via Brent's method."""
    try:
        return brentq(
            lambda sigma: _black_scholes_call(S, K, T, r, sigma) - market_price,
            1e-5, 5.0, maxiter=500,
        )
    except Exception:
        return np.nan


def compute_max_pain(calls_df, puts_df):
    """Compute max pain strike from options chain."""
    strikes = sorted(set(calls_df["strike"]).union(set(puts_df["strike"])))
    total_pain = []
    for s in strikes:
        call_pain = ((s - calls_df["strike"]).clip(lower=0) * calls_df["openInterest"]).sum()
        put_pain = ((puts_df["strike"] - s).clip(lower=0) * puts_df["openInterest"]).sum()
        total_pain.append((s, call_pain + put_pain))
    pain_df = pd.DataFrame(total_pain, columns=["strike", "Total Pain"])
    return pain_df.loc[pain_df["Total Pain"].idxmin(), "strike"]


# ═══════════════════════════════════════════════════════════════════════════
#  Risk Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_risk_metrics(
    ticker: str,
    price_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    rfr: float | None = None,
) -> dict:
    """Compute comprehensive risk metrics vs benchmark."""
    if rfr is None:
        rfr = get_risk_free_rate()

    close = price_df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = np.log(1 + close.pct_change()).dropna()

    bm_close = benchmark_df["Close"]
    if isinstance(bm_close, pd.DataFrame):
        bm_close = bm_close.iloc[:, 0]
    bm_returns = np.log(1 + bm_close.pct_change()).dropna()

    n = len(returns)
    daily_return = (returns + 1).prod() ** (1 / n) - 1
    ann_return = round(((daily_return + 1) ** 252 - 1) * 100, 2)
    vol = round(float(returns.std() * np.sqrt(252)) * 100, 2)
    downside_vol = round(float(returns[returns < 0].std() * np.sqrt(252)) * 100, 2)

    dd = compute_drawdown(returns)
    max_dd = round(float(dd["Drawdown"].min()) * 100, 2)

    Y = returns.values
    X = bm_returns.values[-len(Y):]
    if len(X) > len(Y):
        X = X[:len(Y)]
    elif len(Y) > len(X):
        Y = Y[-len(X):]

    alpha, beta = _linreg(X, Y)
    beta = round(float(beta), 4)

    sharpe = round((ann_return - rfr) / vol, 4) if vol != 0 else 0
    sortino = round((ann_return - rfr) / downside_vol, 4) if downside_vol != 0 else 0
    treynor = round((ann_return - rfr) / beta, 4) if beta != 0 else 0
    calmar = round((ann_return - rfr) / (-max_dd), 4) if max_dd != 0 else 0

    return {
        "annualized_return": ann_return,
        "volatility": vol,
        "downside_volatility": downside_vol,
        "beta": beta,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "treynor_ratio": treynor,
        "calmar_ratio": calmar,
        "drawdown_series": dd,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  GMM Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════

def run_gmm_montecarlo(
    price_df: pd.DataFrame,
    days: int = 60,
    n_simulations: int = 1000,
) -> dict:
    """Run Gaussian Mixture Model Monte Carlo simulations."""
    close = price_df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    # Use log returns explicitly for the simulation
    returns = np.log(close / close.shift(1)).dropna()
    X = returns.values.reshape(-1, 1)

    # ── BIC-based Component Selection ──
    best_gmm = None
    best_bic = np.inf
    # Try 1 to 4 components and pick the best fit based on BIC
    for n in range(1, 5):
        try:
            gmm = GaussianMixture(n_components=n, random_state=42, max_iter=100)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            continue
    
    gmm = best_gmm if best_gmm else GaussianMixture(n_components=1).fit(X)

    last_price = float(close.iloc[-1])
    last_date = close.index[-1]
    dates_list = [last_date + timedelta(days=i + 1) for i in range(days)]

    # ── Vectorized Path Generation ──
    # Sample all returns at once: (days * n_simulations, 1)
    sim_returns, _ = gmm.sample(days * n_simulations)
    sim_returns = sim_returns.reshape(days, n_simulations)
    
    # Correct formula for log returns: Price_t = Price_0 * exp(cumulative sum of returns)
    price_paths = last_price * np.exp(np.cumsum(sim_returns, axis=0))
    
    sim_df = pd.DataFrame(
        price_paths, 
        index=dates_list, 
        columns=[f"S{i + 1}" for i in range(n_simulations)]
    )

    sim_df = _project_percentiles(sim_df)

    # Build percentile summary
    percentile_cols = [c for c in sim_df.columns if c.startswith("P")]
    end_row = sim_df.iloc[-1]
    percentile_summary = {
        col: round(float(end_row[col]), 2) for col in percentile_cols
    }

    return {
        "simulation_df": sim_df,
        "dates": dates_list,
        "last_price": last_price,
        "percentile_summary": percentile_summary,
        "n_simulations": n_simulations,
        "days": days,
        "n_components": gmm.n_components,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Options Analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_options_analysis(options_data: dict, rfr: float | None = None) -> dict:
    """Comprehensive options analysis from raw chain data."""
    if not options_data:
        return {}
    if rfr is None:
        rfr = get_risk_free_rate() / 100.0

    spot = options_data["spot_price"]
    expiration = options_data["expiration"]
    calls = options_data["calls"]
    puts = options_data["puts"]

    from dateutil import parser as dateparser
    T = (dateparser.parse(expiration).date() - datetime.today().date()).days / 365
    days_to_expiry = T * 365

    # ATM implied volatility
    atm_idx = calls["openInterest"].idxmax()
    atm_strike = calls.loc[atm_idx, "strike"]
    atm_iv = calls.loc[calls["strike"] == atm_strike, "impliedVolatility"].mean()

    # Expected move
    expected_move = atm_strike * atm_iv * np.sqrt(days_to_expiry / 365)
    range_lower = round(spot - expected_move, 2)
    range_upper = round(spot + expected_move, 2)
    expected_move_pct = round(expected_move * 100 / spot, 2)

    # Max pain
    max_pain_strike = compute_max_pain(calls, puts)
    max_pain_dist = round((max_pain_strike - spot) / spot * 100, 2)

    # Put/Call ratios
    total_call_oi = calls["openInterest"].sum()
    total_put_oi = puts["openInterest"].sum()
    total_call_vol = calls["volume"].sum()
    total_put_vol = puts["volume"].sum()
    pc_oi = round(total_put_oi * 100 / max(total_call_oi, 1), 2)
    pc_vol = round(total_put_vol * 100 / max(total_call_vol, 1), 2)

    # IV for middle strike (representative)
    mid_call = calls.iloc[len(calls) // 2]
    K = mid_call["strike"]
    mkt_price = mid_call["lastPrice"]
    iv = implied_volatility_call(spot, K, T, rfr, mkt_price)
    iv_pct = round(iv * 100, 2) if not np.isnan(iv) else round(atm_iv * 100, 2)

    return {
        "expiration": expiration,
        "spot_price": spot,
        "days_to_expiry": round(days_to_expiry),
        "iv_format": iv_pct,
        "atm_iv": round(atm_iv * 100, 2),
        "expected_move_return": expected_move_pct,
        "range_lower_bound": range_lower,
        "range_upper_bound": range_upper,
        "max_pain_strike": max_pain_strike,
        "max_pain_distance": max_pain_dist,
        "put_call_oi_ratio": pc_oi,
        "put_call_volume_ratio": pc_vol,
        "total_call_oi": int(total_call_oi),
        "total_put_oi": int(total_put_oi),
        "calls": calls,
        "puts": puts,
    }

