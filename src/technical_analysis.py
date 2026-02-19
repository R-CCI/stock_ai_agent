# -*- coding: utf-8 -*-
"""Enhanced technical analysis with chart pattern detection and indicator computation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from scipy.signal import argrelextrema


# ═══════════════════════════════════════════════════════════════════════════
#  Indicator Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a comprehensive set of technical indicators."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]

    out = df.copy()

    # Moving Averages
    out["SMA20"] = ta.trend.sma_indicator(close, window=20)
    out["SMA50"] = ta.trend.sma_indicator(close, window=50)
    out["SMA200"] = ta.trend.sma_indicator(close, window=200)
    out["EMA12"] = ta.trend.ema_indicator(close, window=12)
    out["EMA26"] = ta.trend.ema_indicator(close, window=26)

    # MACD
    macd = ta.trend.MACD(close)
    out["MACD"] = macd.macd()
    out["MACD_Signal"] = macd.macd_signal()
    out["MACD_Hist"] = macd.macd_diff()

    # RSI
    out["RSI"] = ta.momentum.rsi(close, window=14)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["BB_Upper"] = bb.bollinger_hband()
    out["BB_Middle"] = bb.bollinger_mavg()
    out["BB_Lower"] = bb.bollinger_lband()
    out["BB_Width"] = bb.bollinger_wband()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    out["Stoch_K"] = stoch.stoch()
    out["Stoch_D"] = stoch.stoch_signal()

    # ADX
    adx = ta.trend.ADXIndicator(high, low, close)
    out["ADX"] = adx.adx()
    out["ADX_Pos"] = adx.adx_pos()
    out["ADX_Neg"] = adx.adx_neg()

    # ATR
    out["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)

    # OBV
    out["OBV"] = ta.volume.on_balance_volume(close, volume)

    # CCI
    out["CCI"] = ta.trend.cci(high, low, close, window=20)

    # Williams %R
    out["Williams_R"] = ta.momentum.williams_r(high, low, close, lbp=14)

    # Volume SMA
    out["Vol_SMA20"] = volume.rolling(20).mean()

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Pattern Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_support_resistance(df: pd.DataFrame, window: int = 20, num_levels: int = 5) -> dict:
    """Detect support and resistance levels using local extrema."""
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Find local maxima (resistance) and minima (support)
    local_max_idx = argrelextrema(close.values, np.greater, order=window)[0]
    local_min_idx = argrelextrema(close.values, np.less, order=window)[0]

    resistance_levels = sorted(close.iloc[local_max_idx].values, reverse=True)[:num_levels]
    support_levels = sorted(close.iloc[local_min_idx].values)[:num_levels]

    return {
        "resistance": [round(float(r), 2) for r in resistance_levels],
        "support": [round(float(s), 2) for s in support_levels],
    }


def detect_trend(df: pd.DataFrame, lookback: int = 50) -> dict:
    """Classify trend using linear regression on recent prices."""
    close = df["Close"].tail(lookback)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    x = np.arange(len(close))
    slope, intercept = np.polyfit(x, close.values, 1)
    r_squared = 1 - np.sum((close.values - (slope * x + intercept)) ** 2) / np.sum(
        (close.values - close.mean()) ** 2
    )

    pct_change = slope * len(close) / close.iloc[0] * 100

    if pct_change > 5:
        direction = "Strong Uptrend"
    elif pct_change > 1:
        direction = "Moderate Uptrend"
    elif pct_change > -1:
        direction = "Sideways / Consolidation"
    elif pct_change > -5:
        direction = "Moderate Downtrend"
    else:
        direction = "Strong Downtrend"

    return {
        "direction": direction,
        "slope_pct": round(pct_change, 2),
        "r_squared": round(float(r_squared), 3),
        "lookback_days": lookback,
    }


def detect_head_and_shoulders(df: pd.DataFrame, window: int = 10) -> list:
    """Detect Head & Shoulders and Inverse H&S patterns."""
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    patterns = []
    local_max_idx = argrelextrema(close.values, np.greater, order=window)[0]
    local_min_idx = argrelextrema(close.values, np.less, order=window)[0]

    if len(local_max_idx) >= 3:
        # Check last 3 peaks for H&S
        for i in range(len(local_max_idx) - 2):
            left = close.iloc[local_max_idx[i]]
            head = close.iloc[local_max_idx[i + 1]]
            right = close.iloc[local_max_idx[i + 2]]

            # Head higher than both shoulders, shoulders roughly equal
            if head > left and head > right:
                shoulder_diff = abs(left - right) / max(left, right)
                if shoulder_diff < 0.05:  # shoulders within 5% of each other
                    neckline_idx = local_min_idx[(local_min_idx > local_max_idx[i]) & (local_min_idx < local_max_idx[i + 2])]
                    neckline_vals = close.iloc[neckline_idx].values if len(neckline_idx) > 0 else []
                    neckline = min(neckline_vals) if len(neckline_vals) > 0 else None
                    
                    patterns.append({
                        "type": "Head & Shoulders (Bearish)",
                        "head_price": round(float(head), 2),
                        "head_date": str(df.index[local_max_idx[i + 1]].date()),
                        "left_shoulder": round(float(left), 2),
                        "right_shoulder": round(float(right), 2),
                        "neckline": round(float(neckline), 2) if neckline is not None else None,
                        "coords": [
                            (str(df.index[local_max_idx[i]].date()), float(left)),
                            (str(df.index[local_max_idx[i + 1]].date()), float(head)),
                            (str(df.index[local_max_idx[i + 2]].date()), float(right))
                        ],
                        "neckline_coords": [(str(df.index[idx].date()), float(close.iloc[idx])) for idx in neckline_idx] if len(neckline_idx) > 0 else [],
                        "reliability": "High" if shoulder_diff < 0.03 else "Moderate",
                    })

    if len(local_min_idx) >= 3:
        # Check for inverse H&S
        for i in range(len(local_min_idx) - 2):
            left = close.iloc[local_min_idx[i]]
            head = close.iloc[local_min_idx[i + 1]]
            right = close.iloc[local_min_idx[i + 2]]

            if head < left and head < right:
                shoulder_diff = abs(left - right) / max(left, right)
                if shoulder_diff < 0.05:
                    patterns.append({
                        "type": "Inverse Head & Shoulders (Bullish)",
                        "head_price": round(float(head), 2),
                        "head_date": str(df.index[local_min_idx[i + 1]].date()),
                        "left_shoulder": round(float(left), 2),
                        "right_shoulder": round(float(right), 2),
                        "coords": [
                            (str(df.index[local_min_idx[i]].date()), float(left)),
                            (str(df.index[local_min_idx[i + 1]].date()), float(head)),
                            (str(df.index[local_min_idx[i + 2]].date()), float(right))
                        ],
                        "reliability": "High" if shoulder_diff < 0.03 else "Moderate",
                    })

    return patterns[-3:] if patterns else []  # Return last 3 at most


def detect_double_top_bottom(df: pd.DataFrame, window: int = 15, tolerance: float = 0.03) -> list:
    """Detect Double Top and Double Bottom formations."""
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    patterns = []
    local_max_idx = argrelextrema(close.values, np.greater, order=window)[0]
    local_min_idx = argrelextrema(close.values, np.less, order=window)[0]

    # Double Top
    if len(local_max_idx) >= 2:
        for i in range(len(local_max_idx) - 1):
            p1 = close.iloc[local_max_idx[i]]
            p2 = close.iloc[local_max_idx[i + 1]]
            diff = abs(p1 - p2) / max(p1, p2)
            if diff < tolerance:
                patterns.append({
                    "type": "Double Top (Bearish)",
                    "price_level": round((float(p1) + float(p2)) / 2, 2),
                    "first_peak": str(df.index[local_max_idx[i]].date()),
                    "second_peak": str(df.index[local_max_idx[i + 1]].date()),
                    "coords": [
                        (str(df.index[local_max_idx[i]].date()), float(p1)),
                        (str(df.index[local_max_idx[i + 1]].date()), float(p2))
                    ],
                    "similarity": round((1 - diff) * 100, 1),
                })

    # Double Bottom
    if len(local_min_idx) >= 2:
        for i in range(len(local_min_idx) - 1):
            p1 = close.iloc[local_min_idx[i]]
            p2 = close.iloc[local_min_idx[i + 1]]
            diff = abs(p1 - p2) / max(p1, p2)
            if diff < tolerance:
                patterns.append({
                    "type": "Double Bottom (Bullish)",
                    "price_level": round((float(p1) + float(p2)) / 2, 2),
                    "first_trough": str(df.index[local_min_idx[i]].date()),
                    "second_trough": str(df.index[local_min_idx[i + 1]].date()),
                    "coords": [
                        (str(df.index[local_min_idx[i]].date()), float(p1)),
                        (str(df.index[local_min_idx[i + 1]].date()), float(p2))
                    ],
                    "similarity": round((1 - diff) * 100, 1),
                })

    return patterns[-3:] if patterns else []


def detect_divergences(df: pd.DataFrame, lookback: int = 50) -> list:
    """Detect price/RSI divergences (bullish and bearish)."""
    close = df["Close"].tail(lookback)
    rsi = df["RSI"].tail(lookback) if "RSI" in df.columns else None

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    if rsi is None or rsi.isna().all():
        return []

    if isinstance(rsi, pd.DataFrame):
        rsi = rsi.iloc[:, 0]

    divergences = []

    local_min_idx = argrelextrema(close.values, np.less, order=5)[0]
    local_max_idx = argrelextrema(close.values, np.greater, order=5)[0]

    # Bullish divergence: price makes lower low, RSI makes higher low
    if len(local_min_idx) >= 2:
        i, j = local_min_idx[-2], local_min_idx[-1]
        if close.iloc[j] < close.iloc[i] and rsi.iloc[j] > rsi.iloc[i]:
            divergences.append({
                "type": "Bullish Divergence",
                "description": "Price made a lower low while RSI made a higher low — potential reversal signal",
                "date": str(close.index[j].date()),
            })

    # Bearish divergence: price makes higher high, RSI makes lower high
    if len(local_max_idx) >= 2:
        i, j = local_max_idx[-2], local_max_idx[-1]
        if close.iloc[j] > close.iloc[i] and rsi.iloc[j] < rsi.iloc[i]:
            divergences.append({
                "type": "Bearish Divergence",
                "description": "Price made a higher high while RSI made a lower high — potential reversal signal",
                "date": str(close.index[j].date()),
            })

    return divergences


def detect_sma_crossovers(df: pd.DataFrame) -> list:
    """Detect Golden Cross and Death Cross events."""
    crossovers = []

    if "SMA50" not in df.columns or "SMA200" not in df.columns:
        return crossovers

    sma50 = df["SMA50"]
    sma200 = df["SMA200"]

    if isinstance(sma50, pd.DataFrame):
        sma50 = sma50.iloc[:, 0]
    if isinstance(sma200, pd.DataFrame):
        sma200 = sma200.iloc[:, 0]

    # Check last 60 days for crossovers
    recent = df.tail(60)
    for i in range(1, len(recent)):
        idx = recent.index[i]
        prev = recent.index[i - 1]

        s50_now = sma50.loc[idx]
        s50_prev = sma50.loc[prev]
        s200_now = sma200.loc[idx]
        s200_prev = sma200.loc[prev]

        if pd.isna(s50_now) or pd.isna(s200_now) or pd.isna(s50_prev) or pd.isna(s200_prev):
            continue

        if s50_prev <= s200_prev and s50_now > s200_now:
            crossovers.append({
                "type": "Golden Cross (Bullish)",
                "date": str(idx.date()),
                "description": "SMA50 crossed above SMA200",
            })
        elif s50_prev >= s200_prev and s50_now < s200_now:
            crossovers.append({
                "type": "Death Cross (Bearish)",
                "date": str(idx.date()),
                "description": "SMA50 crossed below SMA200",
            })

    return crossovers


# ═══════════════════════════════════════════════════════════════════════════
#  Combined TA Summary
# ═══════════════════════════════════════════════════════════════════════════

def generate_ta_summary(df: pd.DataFrame, ticker: str) -> dict:
    """Generate a comprehensive technical analysis summary with all patterns."""
    df = compute_indicators(df)
    last = df.iloc[-1]

    close_val = float(last["Close"]) if not isinstance(last["Close"], pd.Series) else float(last["Close"].iloc[0])

    # Signal assessments
    rsi_val = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50
    if rsi_val > 70:
        rsi_signal = "Overbought"
    elif rsi_val < 30:
        rsi_signal = "Oversold"
    elif rsi_val > 60:
        rsi_signal = "Bullish momentum"
    elif rsi_val < 40:
        rsi_signal = "Bearish momentum"
    else:
        rsi_signal = "Neutral"

    macd_val = float(last["MACD"]) if not pd.isna(last["MACD"]) else 0
    macd_sig = float(last["MACD_Signal"]) if not pd.isna(last["MACD_Signal"]) else 0
    macd_signal = "Bullish" if macd_val > macd_sig else "Bearish"

    adx_val = float(last["ADX"]) if not pd.isna(last["ADX"]) else 0
    if adx_val > 40:
        adx_signal = "Strong trend"
    elif adx_val > 25:
        adx_signal = "Trending"
    else:
        adx_signal = "Weak/No trend"

    # Bollinger Band position
    bb_upper = float(last["BB_Upper"]) if not pd.isna(last["BB_Upper"]) else close_val
    bb_lower = float(last["BB_Lower"]) if not pd.isna(last["BB_Lower"]) else close_val
    bb_range = bb_upper - bb_lower
    bb_position = ((close_val - bb_lower) / bb_range * 100) if bb_range > 0 else 50

    # Pattern detection
    patterns = []
    patterns.extend(detect_head_and_shoulders(df))
    patterns.extend(detect_double_top_bottom(df))
    patterns.extend(detect_divergences(df))
    patterns.extend(detect_sma_crossovers(df))

    sr_levels = detect_support_resistance(df)
    trend = detect_trend(df)

    return {
        "ticker": ticker,
        "df": df,
        "indicators": {
            "RSI": round(rsi_val, 2),
            "RSI_Signal": rsi_signal,
            "MACD": round(macd_val, 4),
            "MACD_Signal_Line": round(macd_sig, 4),
            "MACD_Direction": macd_signal,
            "ADX": round(adx_val, 2),
            "ADX_Signal": adx_signal,
            "ATR": round(float(last["ATR"]), 2) if not pd.isna(last["ATR"]) else 0,
            "CCI": round(float(last["CCI"]), 2) if not pd.isna(last["CCI"]) else 0,
            "Williams_R": round(float(last["Williams_R"]), 2) if not pd.isna(last["Williams_R"]) else 0,
            "BB_Position": round(bb_position, 1),
            "BB_Width": round(float(last["BB_Width"]), 4) if not pd.isna(last["BB_Width"]) else 0,
            "Stoch_K": round(float(last["Stoch_K"]), 2) if not pd.isna(last["Stoch_K"]) else 0,
            "Stoch_D": round(float(last["Stoch_D"]), 2) if not pd.isna(last["Stoch_D"]) else 0,
        },
        "trend": trend,
        "patterns": patterns,
        "support_resistance": sr_levels,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Interactive Charts (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

def create_technical_chart(ta_data: dict, lookback: int = 200) -> go.Figure:
    """Create an interactive Plotly chart with overlayed indicators and patterns."""
    df = ta_data["df"].tail(lookback).copy()
    sr = ta_data["support_resistance"]
    patterns = ta_data["patterns"]

    close_col = df["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    open_col = df["Open"]
    if isinstance(open_col, pd.DataFrame):
        open_col = open_col.iloc[:, 0]
    high_col = df["High"]
    if isinstance(high_col, pd.DataFrame):
        high_col = high_col.iloc[:, 0]
    low_col = df["Low"]
    if isinstance(low_col, pd.DataFrame):
        low_col = low_col.iloc[:, 0]
    vol_col = df["Volume"]
    if isinstance(vol_col, pd.DataFrame):
        vol_col = vol_col.iloc[:, 0]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=("Price & Overlays", "RSI", "MACD", "Volume"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=open_col, high=high_col, low=low_col, close=close_col,
        name="Price", increasing_line_color="#00D4AA", decreasing_line_color="#FF4757",
    ), row=1, col=1)

    # SMAs
    for sma, color in [("SMA20", "#FFD93D"), ("SMA50", "#6C5CE7"), ("SMA200", "#FF6B6B")]:
        if sma in df.columns:
            sma_series = df[sma]
            if isinstance(sma_series, pd.DataFrame):
                sma_series = sma_series.iloc[:, 0]
            fig.add_trace(go.Scatter(
                x=df.index, y=sma_series, name=sma,
                line=dict(color=color, width=1.2), opacity=0.8,
            ), row=1, col=1)

    # Bollinger Bands
    for bb, color in [("BB_Upper", "#636e72"), ("BB_Lower", "#636e72")]:
        if bb in df.columns:
            bb_series = df[bb]
            if isinstance(bb_series, pd.DataFrame):
                bb_series = bb_series.iloc[:, 0]
            fig.add_trace(go.Scatter(
                x=df.index, y=bb_series, name=bb,
                line=dict(color=color, width=0.8, dash="dot"), opacity=0.5,
            ), row=1, col=1)

    # Support / Resistance lines
    for level in sr.get("resistance", [])[:3]:
        fig.add_hline(y=level, line_dash="dash", line_color="#FF4757",
                      opacity=0.4, row=1, col=1, annotation_text=f"R: ${level}")
    for level in sr.get("support", [])[:3]:
        fig.add_hline(y=level, line_dash="dash", line_color="#00D4AA",
                      opacity=0.4, row=1, col=1, annotation_text=f"S: ${level}")

    # RSI
    if "RSI" in df.columns:
        rsi_series = df["RSI"]
        if isinstance(rsi_series, pd.DataFrame):
            rsi_series = rsi_series.iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi_series, name="RSI",
            line=dict(color="#FFD93D", width=1.5),
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#FF4757", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00D4AA", opacity=0.5, row=2, col=1)

    # MACD
    if "MACD" in df.columns:
        macd_s = df["MACD"]
        macd_sig_s = df["MACD_Signal"]
        macd_h = df["MACD_Hist"]
        if isinstance(macd_s, pd.DataFrame):
            macd_s = macd_s.iloc[:, 0]
        if isinstance(macd_sig_s, pd.DataFrame):
            macd_sig_s = macd_sig_s.iloc[:, 0]
        if isinstance(macd_h, pd.DataFrame):
            macd_h = macd_h.iloc[:, 0]

        colors = ["#00D4AA" if v >= 0 else "#FF4757" for v in macd_h.fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=macd_h, name="MACD Hist",
                             marker_color=colors, opacity=0.6), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_s, name="MACD",
                                 line=dict(color="#6C5CE7", width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_sig_s, name="Signal",
                                 line=dict(color="#FFD93D", width=1.2)), row=3, col=1)

    # Volume
    vol_colors = ["#00D4AA" if c >= o else "#FF4757"
                  for c, o in zip(close_col.fillna(0), open_col.fillna(0))]
    fig.add_trace(go.Bar(x=df.index, y=vol_col, name="Volume",
                         marker_color=vol_colors, opacity=0.6), row=4, col=1)

    if "Vol_SMA20" in df.columns:
        vol_sma = df["Vol_SMA20"]
        if isinstance(vol_sma, pd.DataFrame):
            vol_sma = vol_sma.iloc[:, 0]
        fig.add_trace(go.Scatter(x=df.index, y=vol_sma, name="Vol SMA20",
                                 line=dict(color="#FFD93D", width=1)), row=4, col=1)

    # Pattern Drawings & Annotations
    for p in patterns:
        try:
            # 1. Draw Silhouette (if coords exist)
            if "coords" in p and p["coords"]:
                # Head & Shoulders or Double Top/Bottom silhouettes
                dates = [c[0] for c in p["coords"]]
                prices = [c[1] for c in p["coords"]]
                
                # Filter coords to only show if within current view
                if any(pd.to_datetime(d) in df.index for d in dates):
                    fig.add_trace(go.Scatter(
                        x=dates, y=prices,
                        mode="lines+markers",
                        name=p["type"],
                        line=dict(color="#FFD93D", width=2, dash="dot"),
                        marker=dict(size=6, symbol="diamond"),
                        showlegend=False,
                    ), row=1, col=1)

                # 1b. Draw Neckline (if exists)
                if "neckline_coords" in p and p["neckline_coords"]:
                    n_dates = [c[0] for c in p["neckline_coords"]]
                    n_prices = [c[1] for c in p["neckline_coords"]]
                    if len(n_dates) >= 2:
                        fig.add_trace(go.Scatter(
                            x=n_dates, y=n_prices,
                            mode="lines",
                            name="Neckline",
                            line=dict(color="#6C5CE7", width=1.5, dash="dash"),
                            showlegend=False,
                        ), row=1, col=1)

            # 2. Add Annotation
            # Determine best X/Y for annotation
            ann_x = p.get("date") or p.get("head_date") or p.get("first_peak") or p.get("first_trough")
            ann_y = p.get("head_price") or p.get("price_level") or (p["coords"][0][1] if "coords" in p else 0)
            
            if ann_x and pd.to_datetime(ann_x) in df.index:
                fig.add_annotation(
                    x=ann_x, y=ann_y, xref="x", yref="y",
                    text=f"⚡ {p['type']}", 
                    showarrow=True, arrowhead=2,
                    font=dict(size=10, color="#FFD93D"),
                    bgcolor="rgba(0,0,0,0.8)", bordercolor="#FFD93D",
                    ay=-40,
                )
        except Exception:
            continue

    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
    )

    return fig


def create_options_charts(options_analysis: dict, ticker: str) -> go.Figure:
    """Create interactive options analysis charts."""
    calls = options_analysis["calls"]
    puts = options_analysis["puts"]
    spot = options_analysis["spot_price"]
    max_pain = options_analysis["max_pain_strike"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Call Open Interest", "Put Open Interest",
            "Implied Volatility Smile", "Net OI Pressure"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Call OI
    fig.add_trace(go.Bar(
        x=calls["strike"], y=calls["openInterest"],
        name="Call OI", marker_color="#00D4AA", opacity=0.7,
    ), row=1, col=1)

    # Put OI
    fig.add_trace(go.Bar(
        x=puts["strike"], y=puts["openInterest"],
        name="Put OI", marker_color="#FF4757", opacity=0.7,
    ), row=1, col=2)

    # IV Smile
    fig.add_trace(go.Scatter(
        x=calls["strike"], y=calls["impliedVolatility"],
        name="Call IV", mode="lines+markers",
        line=dict(color="#00D4AA"), marker=dict(size=4),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=puts["strike"], y=puts["impliedVolatility"],
        name="Put IV", mode="lines+markers",
        line=dict(color="#FF4757"), marker=dict(size=4),
    ), row=2, col=1)

    # Net OI
    merged = pd.merge(
        calls[["strike", "openInterest"]].rename(columns={"openInterest": "call_oi"}),
        puts[["strike", "openInterest"]].rename(columns={"openInterest": "put_oi"}),
        on="strike", how="outer",
    ).fillna(0)
    merged["net_oi"] = merged["call_oi"] - merged["put_oi"]
    colors = ["#00D4AA" if v > 0 else "#FF4757" for v in merged["net_oi"]]
    fig.add_trace(go.Bar(
        x=merged["strike"], y=merged["net_oi"],
        name="Net OI", marker_color=colors, opacity=0.7,
    ), row=2, col=2)

    # Vertical lines for spot and max pain on all subplots
    for row in [1, 2]:
        for col in [1, 2]:
            fig.add_vline(x=spot, line_dash="solid", line_color="#FFD93D",
                          opacity=0.6, row=row, col=col)
            fig.add_vline(x=max_pain, line_dash="dash", line_color="#FFA502",
                          opacity=0.6, row=row, col=col)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_montecarlo_chart(mc_data: dict, ticker: str) -> go.Figure:
    """Create interactive Monte Carlo simulation fan chart."""
    sim_df = mc_data["simulation_df"]
    dates = mc_data["dates"]
    last_price = mc_data["last_price"]

    fig = go.Figure()

    # Individual simulation paths
    sim_cols = [c for c in sim_df.columns if c.startswith("S")]
    for col in sim_cols:
        fig.add_trace(go.Scatter(
            x=dates, y=sim_df[col], mode="lines",
            line=dict(color="rgba(200,200,200,0.15)", width=0.5),
            showlegend=False, hoverinfo="skip",
        ))

    # Percentile bands
    percentile_config = [
        ("P1", "#FF4757", "dash", "P1 (Extreme Bear)"),
        ("P5", "#FF6B81", "dash", "P5 (Bear)"),
        ("P25", "#FFA502", "dot", "P25"),
        ("P50", "#00D4AA", "solid", "P50 (Median)"),
        ("P75", "#6C5CE7", "dot", "P75"),
        ("P95", "#74B9FF", "dash", "P95 (Bull)"),
        ("P99", "#55EFC4", "dash", "P99 (Extreme Bull)"),
    ]

    for col, color, dash, name in percentile_config:
        if col in sim_df.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=sim_df[col], mode="lines",
                name=name, line=dict(color=color, width=2, dash=dash),
            ))

    # Starting price line
    fig.add_hline(y=last_price, line_dash="dot", line_color="#FFD93D",
                  opacity=0.5, annotation_text=f"Current: ${last_price:.2f}")

    fig.update_layout(
        template="plotly_dark",
        title=f"{ticker} — GMM Monte Carlo Simulations",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        height=500,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def create_drawdown_chart(dd_series: pd.DataFrame, ticker: str) -> go.Figure:
    """Create an interactive drawdown chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series["Drawdown"] * 100,
        fill="tozeroy", fillcolor="rgba(255,71,87,0.2)",
        line=dict(color="#FF4757", width=1.5),
        name="Drawdown (%)",
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"{ticker} — Drawdown History",
        yaxis_title="Drawdown (%)",
        height=350,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
    )

    return fig


def create_dcf_charts(dcf_results: dict, method: str = "gordon") -> list[go.Figure]:
    """Create visualizations for DCF projections and value breakdown."""
    method_data = dcf_results.get(method, {})
    projections = dcf_results.get("projections", [])
    
    if not method_data or not projections:
        return []

    # 1. Projections Chart (Bar)
    years = [f"Año {p['year']}" for p in projections]
    fcf_vals = [p["fcf"] for p in projections]
    pv_fcf_vals = [p["pv_fcf"] for p in projections]

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Bar(
        x=years, y=fcf_vals, name="Flujo de Caja Libre (FCF)",
        marker_color="#74B9FF", opacity=0.8
    ))
    fig_proj.add_trace(go.Bar(
        x=years, y=pv_fcf_vals, name="PV del FCF",
        marker_color="#00D4AA", opacity=0.9
    ))

    fig_proj.update_layout(
        template="plotly_dark",
        title="Proyección de Flujos de Caja Futuros",
        xaxis_title="Período",
        yaxis_title="Monto ($)",
        barmode="group",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
    )

    # 2. Value Concentration (Pie)
    pv_sum_cf = dcf_results.get("sum_pv_cf", 0)
    pv_tv = method_data.get("pv_terminal_value", 0)

    fig_pie = go.Figure(data=[go.Pie(
        labels=["PV de Flujos 5 Años", "PV del Valor Terminal"],
        values=[pv_sum_cf, pv_tv],
        hole=.4,
        marker_colors=["#6C5CE7", "#FFD93D"]
    )])

    fig_pie.update_layout(
        template="plotly_dark",
        title="Distribución del Valor Intrínseco",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=400,
    )

    return [fig_proj, fig_pie]
