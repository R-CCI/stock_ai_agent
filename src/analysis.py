# -*- coding: utf-8 -*-
"""Financial analysis — risk metrics, Monte Carlo, options, improved Markov & DCF."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import brentq
from scipy.stats import norm, dirichlet
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

    returns = np.log(close / close.shift(1)).dropna()
    X = returns.values.reshape(-1, 1)

    # BIC-based Component Selection
    best_gmm = None
    best_bic = np.inf
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

    sim_returns, _ = gmm.sample(days * n_simulations)
    sim_returns = sim_returns.reshape(days, n_simulations)
    price_paths = last_price * np.exp(np.cumsum(sim_returns, axis=0))

    sim_df = pd.DataFrame(
        price_paths,
        index=dates_list,
        columns=[f"S{i + 1}" for i in range(n_simulations)]
    )
    sim_df = _project_percentiles(sim_df)

    percentile_cols = [c for c in sim_df.columns if c.startswith("P")]
    end_row = sim_df.iloc[-1]
    percentile_summary = {col: round(float(end_row[col]), 2) for col in percentile_cols}

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

    calls = options_data.get("calls", pd.DataFrame()).copy()
    puts = options_data.get("puts", pd.DataFrame()).copy()
    if calls.empty and puts.empty:
        return {}

    spot = float(options_data.get("spot_price", 0) or 0)
    expiration = options_data.get("expiration", "")

    reference_chain = calls if not calls.empty else puts
    if reference_chain.empty:
        return {}
    if spot <= 0:
        spot = float(reference_chain["strike"].median())

    from dateutil import parser as dateparser
    try:
        days_to_expiry = max((dateparser.parse(expiration).date() - datetime.today().date()).days, 1)
    except Exception:
        days_to_expiry = 1
    T = max(days_to_expiry / 365, 1 / 365)

    atm_idx = (reference_chain["strike"] - spot).abs().idxmin()
    atm_strike = float(reference_chain.loc[atm_idx, "strike"])
    atm_slice = reference_chain.loc[reference_chain["strike"] == atm_strike, "impliedVolatility"]
    atm_iv = pd.to_numeric(atm_slice, errors="coerce").mean()
    if pd.isna(atm_iv) or atm_iv <= 0:
        atm_iv = pd.to_numeric(reference_chain.get("impliedVolatility"), errors="coerce").dropna().median()
    if pd.isna(atm_iv) or atm_iv <= 0:
        atm_iv = 0.25

    safe_spot = max(spot, 0.01)
    expected_move = atm_strike * atm_iv * np.sqrt(days_to_expiry / 365)
    range_lower = round(spot - expected_move, 2)
    range_upper = round(spot + expected_move, 2)
    expected_move_pct = round(expected_move * 100 / safe_spot, 2)

    max_pain_strike = compute_max_pain(calls, puts) if not (calls.empty and puts.empty) else atm_strike
    max_pain_dist = round((max_pain_strike - spot) / safe_spot * 100, 2)

    total_call_oi = pd.to_numeric(calls.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    total_put_oi = pd.to_numeric(puts.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    total_call_vol = pd.to_numeric(calls.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    total_put_vol = pd.to_numeric(puts.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    pc_oi = round(total_put_oi * 100 / max(total_call_oi, 1), 2)
    pc_vol = round(total_put_vol * 100 / max(total_call_vol, 1), 2)

    pricing_chain = calls if not calls.empty else reference_chain
    mid_call = pricing_chain.iloc[len(pricing_chain) // 2]
    K = float(mid_call["strike"])
    mkt_price = float(mid_call.get("lastPrice", 0) or 0)
    iv = implied_volatility_call(spot, K, T, rfr, mkt_price) if mkt_price > 0 else np.nan
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
        "all_expirations": options_data.get("all_expirations", []),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Valuation: Multi-Stage DCF with Scenario Analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_wacc(
    equity_weight: float,
    debt_weight: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float = 0.21,
) -> float:
    """
    Calculate Weighted Average Cost of Capital (WACC).

    Args:
        equity_weight: E / (E + D)
        debt_weight:   D / (E + D)
        cost_of_equity: required return on equity (e.g. from CAPM)
        cost_of_debt:   pre-tax interest rate on debt
        tax_rate:       effective corporate tax rate (default 21%)
    """
    return equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)


def compute_capm_cost_of_equity(
    beta: float,
    risk_free_rate: float,
    equity_risk_premium: float = 0.055,
) -> float:
    """
    CAPM: E(r) = Rf + beta * ERP

    Args:
        beta: systematic risk vs market
        risk_free_rate: 10Y US Treasury yield (decimal, e.g. 0.045)
        equity_risk_premium: market ERP, default Damodaran estimate 5.5%
    """
    return risk_free_rate + beta * equity_risk_premium


def compute_dcf(
    fcf_base: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    ebitda_base: float | None = None,
    exit_multiple: float | None = None,
    net_debt: float = 0,
    shares_outstanding: float = 1,
    years: int = 5
) -> dict:
    """
    Single-stage DCF (Gordon Growth + Exit Multiple).

    Used as the baseline from the main pipeline.
    """
    projections = []
    current_fcf = fcf_base

    for i in range(1, years + 1):
        current_fcf *= (1 + growth_rate)
        discount_factor = 1 / ((1 + wacc) ** i)
        pv_fcf = current_fcf * discount_factor
        projections.append({
            "year": i,
            "fcf": round(current_fcf, 2),
            "pv_fcf": round(pv_fcf, 2)
        })

    sum_pv_cf = sum(p["pv_fcf"] for p in projections)
    final_fcf = projections[-1]["fcf"]

    tv_gordon = (final_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    pv_tv_gordon = tv_gordon / ((1 + wacc) ** years)

    ev_gordon = sum_pv_cf + pv_tv_gordon
    equity_val_gordon = ev_gordon - net_debt
    price_gordon = equity_val_gordon / shares_outstanding if shares_outstanding > 0 else 0

    price_exit = 0
    ev_exit = 0
    pv_tv_exit = 0
    if ebitda_base is not None and exit_multiple is not None:
        projected_ebitda = ebitda_base * ((1 + growth_rate) ** years)
        tv_exit = projected_ebitda * exit_multiple
        pv_tv_exit = tv_exit / ((1 + wacc) ** years)
        ev_exit = sum_pv_cf + pv_tv_exit
        equity_val_exit = ev_exit - net_debt
        price_exit = equity_val_exit / shares_outstanding if shares_outstanding > 0 else 0

    return {
        "projections": projections,
        "sum_pv_cf": round(sum_pv_cf, 2),
        "gordon": {
            "terminal_value": round(tv_gordon, 2),
            "pv_terminal_value": round(pv_tv_gordon, 2),
            "enterprise_value": round(ev_gordon, 2),
            "equity_value": round(equity_val_gordon, 2),
            "implied_price": round(price_gordon, 2),
            "pct_from_tv": round(pv_tv_gordon * 100 / ev_gordon, 1) if ev_gordon > 0 else 0
        },
        "exit": {
            "terminal_value": round(pv_tv_exit * ((1 + wacc) ** years), 2) if pv_tv_exit > 0 else 0,
            "pv_terminal_value": round(pv_tv_exit, 2),
            "enterprise_value": round(ev_exit, 2),
            "equity_value": round(ev_exit - net_debt, 2),
            "implied_price": round(price_exit, 2),
            "pct_from_tv": round(pv_tv_exit * 100 / ev_exit, 1) if ev_exit > 0 else 0
        }
    }


def compute_multistage_dcf(
    fcf_base: float,
    stage1_growth: float,
    stage2_growth: float,
    terminal_growth: float,
    wacc: float,
    stage1_years: int = 5,
    stage2_years: int = 5,
    net_debt: float = 0,
    shares_outstanding: float = 1,
    ebitda_base: float | None = None,
    exit_multiple: float | None = None,
) -> dict:
    """
    Three-stage DCF model:
      Stage 1 (High Growth): years 1–stage1_years at stage1_growth
      Stage 2 (Transition):  years (stage1_years+1)–(stage1_years+stage2_years)
                             at linearly declining rate from stage1_growth → terminal_growth
      Stage 3 (Terminal):    Gordon Growth or Exit Multiple

    Returns per-year projections + summary metrics.
    """
    projections = []
    current_fcf = fcf_base
    total_years = stage1_years + stage2_years

    for i in range(1, total_years + 1):
        if i <= stage1_years:
            growth = stage1_growth
        else:
            # Linear interpolation from stage1_growth → terminal_growth
            t = (i - stage1_years) / stage2_years
            growth = stage1_growth + t * (terminal_growth - stage1_growth)

        current_fcf *= (1 + growth)
        discount_factor = 1 / ((1 + wacc) ** i)
        pv_fcf = current_fcf * discount_factor

        projections.append({
            "year": i,
            "stage": 1 if i <= stage1_years else 2,
            "growth_rate": round(growth * 100, 2),
            "fcf": round(current_fcf, 2),
            "discount_factor": round(discount_factor, 4),
            "pv_fcf": round(pv_fcf, 2),
        })

    sum_pv_cf = sum(p["pv_fcf"] for p in projections)
    final_fcf = projections[-1]["fcf"]

    # Terminal Value — Gordon Growth
    if wacc > terminal_growth:
        tv_gordon = (final_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    else:
        tv_gordon = final_fcf * 15  # Safety cap

    pv_tv_gordon = tv_gordon / ((1 + wacc) ** total_years)
    ev_gordon = sum_pv_cf + pv_tv_gordon
    equity_gordon = ev_gordon - net_debt
    price_gordon = equity_gordon / shares_outstanding if shares_outstanding > 0 else 0

    # Terminal Value — Exit Multiple
    price_exit = ev_exit = pv_tv_exit = 0.0
    if ebitda_base is not None and exit_multiple is not None:
        projected_ebitda = ebitda_base * ((1 + stage1_growth) ** stage1_years) * \
                           ((1 + (stage1_growth + terminal_growth) / 2) ** stage2_years)
        tv_exit = projected_ebitda * exit_multiple
        pv_tv_exit = tv_exit / ((1 + wacc) ** total_years)
        ev_exit = sum_pv_cf + pv_tv_exit
        price_exit = (ev_exit - net_debt) / shares_outstanding if shares_outstanding > 0 else 0

    return {
        "model": "multistage",
        "projections": projections,
        "stage1_years": stage1_years,
        "stage2_years": stage2_years,
        "total_years": total_years,
        "sum_pv_cf": round(sum_pv_cf, 2),
        "gordon": {
            "terminal_value": round(tv_gordon, 2),
            "pv_terminal_value": round(pv_tv_gordon, 2),
            "enterprise_value": round(ev_gordon, 2),
            "equity_value": round(equity_gordon, 2),
            "implied_price": round(price_gordon, 2),
            "pct_from_tv": round(pv_tv_gordon * 100 / ev_gordon, 1) if ev_gordon > 0 else 0,
        },
        "exit": {
            "terminal_value": round(pv_tv_exit * ((1 + wacc) ** total_years), 2),
            "pv_terminal_value": round(pv_tv_exit, 2),
            "enterprise_value": round(ev_exit, 2),
            "equity_value": round(ev_exit - net_debt, 2),
            "implied_price": round(price_exit, 2),
            "pct_from_tv": round(pv_tv_exit * 100 / ev_exit, 1) if ev_exit > 0 else 0,
        },
    }


def compute_dcf_scenarios(
    fcf_base: float,
    wacc_base: float,
    terminal_growth_base: float,
    net_debt: float = 0,
    shares_outstanding: float = 1,
    ebitda_base: float | None = None,
    exit_multiple: float = 12.0,
) -> dict:
    """
    Run Bull / Base / Bear DCF scenarios.

    Scenario parameters:
        Bull:  high FCF growth, slightly lower WACC
        Base:  moderate FCF growth, base WACC
        Bear:  low FCF growth, higher WACC (stress)

    Returns a dict with per-scenario results and a sensitivity table.
    """
    scenarios = {
        "bull": {
            "label": "Alcista (Bull)",
            "color": "#00D4AA",
            "stage1_growth": 0.20,
            "stage2_growth": 0.12,
            "wacc": max(wacc_base - 0.02, 0.08),
            "terminal_growth": min(terminal_growth_base + 0.005, 0.04),
        },
        "base": {
            "label": "Base",
            "color": "#74B9FF",
            "stage1_growth": 0.12,
            "stage2_growth": 0.07,
            "wacc": wacc_base,
            "terminal_growth": terminal_growth_base,
        },
        "bear": {
            "label": "Bajista (Bear)",
            "color": "#FF4757",
            "stage1_growth": 0.04,
            "stage2_growth": 0.02,
            "wacc": wacc_base + 0.03,
            "terminal_growth": max(terminal_growth_base - 0.01, 0.005),
        },
    }

    results = {}
    for key, params in scenarios.items():
        dcf = compute_multistage_dcf(
            fcf_base=fcf_base,
            stage1_growth=params["stage1_growth"],
            stage2_growth=params["stage2_growth"],
            terminal_growth=params["terminal_growth"],
            wacc=params["wacc"],
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
            ebitda_base=ebitda_base,
            exit_multiple=exit_multiple,
        )
        results[key] = {
            **params,
            "gordon_price": dcf["gordon"]["implied_price"],
            "exit_price": dcf["exit"]["implied_price"],
            "ev": dcf["gordon"]["enterprise_value"],
            "pct_from_tv": dcf["gordon"]["pct_from_tv"],
            "projections": dcf["projections"],
            "sum_pv_cf": dcf["sum_pv_cf"],
        }

    # Sensitivity: WACC vs Terminal Growth grid
    wacc_range = np.arange(wacc_base - 0.04, wacc_base + 0.05, 0.01)
    tg_range = np.arange(
        max(terminal_growth_base - 0.02, 0.005),
        terminal_growth_base + 0.03,
        0.005,
    )

    sensitivity = {}
    for w in wacc_range:
        row = {}
        for tg in tg_range:
            if w > tg:
                dcf_s = compute_multistage_dcf(
                    fcf_base=fcf_base,
                    stage1_growth=0.12,
                    stage2_growth=0.07,
                    terminal_growth=round(tg, 4),
                    wacc=round(w, 4),
                    net_debt=net_debt,
                    shares_outstanding=shares_outstanding,
                )
                row[round(tg * 100, 2)] = round(dcf_s["gordon"]["implied_price"], 2)
        if row:
            sensitivity[round(w * 100, 2)] = row

    return {
        "scenarios": results,
        "sensitivity_wacc_tg": sensitivity,
        "wacc_base": wacc_base,
        "terminal_growth_base": terminal_growth_base,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Improved Markov Chain — Bayesian Smoothing + Multi-Step Prediction
# ═══════════════════════════════════════════════════════════════════════════

def _bayesian_smooth_transition_matrix(
    counts: pd.DataFrame,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Apply Dirichlet-Laplace Bayesian smoothing to a raw transition counts matrix.

    Each row is treated as a Dirichlet posterior with concentration parameter α.
    α = 1.0 → add-one (Laplace) smoothing (uniform prior)
    α < 1.0 → sparser prior; α > 1.0 → stronger toward uniform

    Returns a row-stochastic probability matrix.
    """
    smoothed = counts.copy().astype(float)
    smoothed += alpha  # Add Dirichlet concentration parameter to each cell
    row_sums = smoothed.sum(axis=1)
    return smoothed.div(row_sums, axis=0)


def _stationary_distribution(tm: pd.DataFrame) -> pd.Series:
    """
    Compute the stationary (long-run) distribution π of a Markov chain.

    Solves π P = π, ∑π = 1 using eigenvalue decomposition.
    """
    try:
        P = tm.values.T
        eigenvalues, eigenvectors = np.linalg.eig(P)
        # Stationary dist = eigenvector for eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi) / np.abs(pi).sum()
        return pd.Series(pi, index=tm.index)
    except Exception:
        n = len(tm)
        return pd.Series([1 / n] * n, index=tm.index)


def _multistep_probabilities(tm: pd.DataFrame, current_state: str, steps: int = 4) -> pd.DataFrame:
    """
    Compute n-step ahead probability distributions by matrix exponentiation.
    P(state at t+n | current_state) = e_current · P^n

    Returns a DataFrame with rows=step, columns=states.
    """
    P = tm.values
    states = list(tm.index)

    if current_state not in states:
        # Fallback: use stationary distribution
        pi = _stationary_distribution(tm)
        rows = [pi.values] * steps
    else:
        idx = states.index(current_state)
        e = np.zeros(len(states))
        e[idx] = 1.0

        rows = []
        Pn = np.eye(len(states))
        for _ in range(steps):
            Pn = Pn @ P
            rows.append((e @ Pn).tolist())

    return pd.DataFrame(rows, columns=states, index=[f"Q+{i+1}" for i in range(steps)])


def compute_markov_earnings(history_df: pd.DataFrame) -> dict:
    """
    Enhanced Markov Chain model from historical earnings data.

    Improvements vs original:
    - Bayesian (Dirichlet) smoothing for sparse transition matrices
    - Multi-step (4-quarter ahead) probability predictions via P^n
    - Stationary distribution (long-run expected behavior)
    - Rolling window transition matrix to detect regime shifts
    - Confidence scores based on sample size

    Model 1 — EPS only (3 states):
        'B' = Beat  (surprise_pct > +2%)
        'M' = Miss  (surprise_pct < -2%)
        'N' = Near  (-2% ≤ surprise_pct ≤ +2%)

    Model 2 — EPS + Market Reaction (5 states):
        'BR' = Beat + Mercado Sube
        'BF' = Beat + Mercado Cae
        'MR' = Miss + Mercado Sube
        'MF' = Miss + Mercado Cae
        'N'  = Neutral
    """
    if history_df.empty or "beat" not in history_df.columns:
        return {}

    df = history_df.dropna(subset=["surprise_pct"]).copy()
    if len(df) < 2:
        return {}

    # ── Model 1: EPS States ────────────────────────────────────────────────
    def classify_eps(s_pct):
        if s_pct > 2:    return "B"
        elif s_pct < -2: return "M"
        return "N"

    df["state_eps"] = df["surprise_pct"].apply(classify_eps)
    seq_eps = df["state_eps"].tolist()
    states_eps = ["B", "M", "N"]

    # Raw counts
    counts_eps = pd.DataFrame(0, index=states_eps, columns=states_eps)
    for i in range(len(seq_eps) - 1):
        counts_eps.loc[seq_eps[i], seq_eps[i + 1]] += 1

    # Bayesian smoothing (Laplace / Dirichlet prior)
    sample_size = len(seq_eps)
    alpha_smooth = 1.0 if sample_size >= 12 else 0.5  # Lighter prior if short history
    tm_eps = _bayesian_smooth_transition_matrix(counts_eps, alpha=alpha_smooth)

    current_eps = seq_eps[-1]
    prob_eps = tm_eps.loc[current_eps] if current_eps in tm_eps.index else pd.Series({s: 1/3 for s in states_eps})
    beat_prob = round(float(prob_eps.get("B", 0)) * 100, 1)
    miss_prob = round(float(prob_eps.get("M", 0)) * 100, 1)
    near_prob = round(float(prob_eps.get("N", 0)) * 100, 1)

    # Multi-step prediction (next 4 quarters)
    multistep = _multistep_probabilities(tm_eps, current_eps, steps=4)
    multistep_pct = (multistep * 100).round(1)

    # Stationary distribution
    stationary = _stationary_distribution(tm_eps)
    stationary_pct = (stationary * 100).round(1)

    # Rolling window transition matrix (last 8 quarters if available)
    rolling_tm = None
    if len(seq_eps) >= 8:
        recent_seq = seq_eps[-8:]
        rolling_counts = pd.DataFrame(0, index=states_eps, columns=states_eps)
        for i in range(len(recent_seq) - 1):
            rolling_counts.loc[recent_seq[i], recent_seq[i + 1]] += 1
        rolling_tm = _bayesian_smooth_transition_matrix(rolling_counts, alpha=0.5)

    # Streaks
    beat_streak = miss_streak = 0
    for s in reversed(seq_eps):
        if s == "B":
            if miss_streak == 0: beat_streak += 1
            else: break
        elif s == "M":
            if beat_streak == 0: miss_streak += 1
            else: break
        else: break

    historical_beat_rate = round(100 * float(df["beat"].sum()) / len(df), 1)

    # Confidence: based on sample size
    if sample_size >= 16:    confidence = "Alta"
    elif sample_size >= 8:   confidence = "Moderada"
    else:                    confidence = "Baja (pocos datos)"

    if beat_prob >= 60:    verdict = "🟢 Alta probabilidad de BEAT"
    elif beat_prob >= 45:  verdict = "🟡 Probabilidad moderada de beat"
    elif miss_prob >= 60:  verdict = "🔴 Alta probabilidad de MISS"
    else:                  verdict = "⚪ Sin señal clara"

    # ── Model 2: EPS + Market Reaction ────────────────────────────────────
    markov_reaction = {}
    has_react = "stock_reaction_pct" in df.columns and df["stock_reaction_pct"].notna().any()

    if has_react:
        df_r = df.dropna(subset=["stock_reaction_pct"]).copy()

        def classify_combined(row):
            s = row["surprise_pct"] or 0
            r = row["stock_reaction_pct"] or 0
            if s > 2   and r > 1:   return "BR"
            if s > 2   and r <= -1: return "BF"
            if s < -2  and r >= 1:  return "MR"
            if s < -2  and r < -1:  return "MF"
            return "N"

        df_r["state_combined"] = df_r.apply(classify_combined, axis=1)
        seq_cmb = df_r["state_combined"].tolist()
        states_cmb = ["BR", "BF", "MR", "MF", "N"]

        counts_cmb = pd.DataFrame(0, index=states_cmb, columns=states_cmb)
        for i in range(len(seq_cmb) - 1):
            counts_cmb.loc[seq_cmb[i], seq_cmb[i + 1]] += 1

        active = [s for s in states_cmb if counts_cmb.loc[s].sum() > 0 or counts_cmb[s].sum() > 0]
        if not active:
            active = states_cmb
        counts_cmb = counts_cmb.loc[active, active]

        # Bayesian smoothing
        tm_cmb = _bayesian_smooth_transition_matrix(counts_cmb, alpha=0.5)

        current_cmb = seq_cmb[-1]
        if current_cmb in tm_cmb.index:
            prob_cmb = tm_cmb.loc[current_cmb]
        else:
            prob_cmb = pd.Series({s: 1 / len(active) for s in active})

        label_map = {
            "BR": "Beat+Sube ✅📈", "BF": "Beat+Cae ✅📉",
            "MR": "Miss+Sube ❌📈", "MF": "Miss+Cae ❌📉", "N": "Neutral ⚪",
        }
        probs_cmb = {label_map.get(s, s): round(float(v) * 100, 1) for s, v in prob_cmb.items()}

        tm_display = tm_cmb.copy()
        tm_display.index   = [label_map.get(s, s) for s in tm_display.index]
        tm_display.columns = [label_map.get(s, s) for s in tm_display.columns]

        best_outcome = max(probs_cmb, key=probs_cmb.get)
        best_prob = probs_cmb[best_outcome]

        # Multi-step for combined model
        multistep_cmb = _multistep_probabilities(tm_cmb, current_cmb, steps=4)
        multistep_cmb.index   = [f"Q+{i+1}" for i in range(4)]
        multistep_cmb.columns = [label_map.get(s, s) for s in multistep_cmb.columns]
        multistep_cmb_pct = (multistep_cmb * 100).round(1)

        # Stationary distribution for combined model
        stat_cmb = _stationary_distribution(tm_cmb)
        stat_cmb.index = [label_map.get(s, s) for s in stat_cmb.index]

        paradox_n = sum(1 for s in seq_cmb if s in ("BF", "MR"))
        paradox_rate = round(100 * paradox_n / len(seq_cmb), 1) if seq_cmb else 0

        color_map = {
            "Beat+Sube ✅📈": "🟢", "Beat+Cae ✅📉": "🟡",
            "Miss+Sube ❌📈": "🟠", "Miss+Cae ❌📉": "🔴", "Neutral ⚪": "⚪"
        }
        icon = color_map.get(best_outcome, "⚪")
        combined_verdict = f"{icon} Resultado más probable: **{best_outcome}** ({best_prob}%)"

        markov_reaction = {
            "transition_matrix": tm_display.round(3),
            "current_state": label_map.get(current_cmb, current_cmb),
            "probabilities": probs_cmb,
            "multistep": multistep_cmb_pct,
            "stationary": (stat_cmb * 100).round(1),
            "best_outcome": best_outcome,
            "best_probability": best_prob,
            "paradox_rate": paradox_rate,
            "combined_verdict": combined_verdict,
            "total_quarters": len(df_r),
        }

    return {
        # Model 1
        "transition_matrix": tm_eps.round(3),
        "current_state": current_eps,
        "beat_probability": beat_prob,
        "miss_probability": miss_prob,
        "near_probability": near_prob,
        "beat_streak": beat_streak,
        "miss_streak": miss_streak,
        "historical_beat_rate": historical_beat_rate,
        "total_quarters": len(df),
        "verdict": verdict,
        "confidence": confidence,
        # Multi-step prediction
        "multistep_prediction": multistep_pct,
        # Long-run behavior
        "stationary_distribution": (stationary_pct).to_dict(),
        # Recent regime (rolling window)
        "rolling_transition_matrix": rolling_tm.round(3) if rolling_tm is not None else None,
        # Model 2
        "markov_reaction": markov_reaction,
    }
