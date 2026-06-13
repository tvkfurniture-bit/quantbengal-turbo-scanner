# ============================================================
# QuantBengal Pro SMC Terminal  |  app.py
# Senior Quant Dev & Market Microstructure Analyst build
# Strategy: Smart Money Concepts (SMC) Liquidity Sweep Detection
# Stack: Python 3.11+ | Streamlit | Plotly | Pandas | yfinance
# Architecture: Single-file. No local imports.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantBengal Pro | SMC Terminal",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE  (dark terminal aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {
      background-color: #0a0e14;
      color: #cdd6f4;
      font-family: 'JetBrains Mono', 'Courier New', monospace;
  }
  [data-testid="stSidebar"] {
      background-color: #0d1117;
      border-right: 1px solid #1e2a3a;
  }

  /* ── Header strip ── */
  .terminal-header {
      background: linear-gradient(90deg, #0d1117 0%, #0f2027 60%, #0d1117 100%);
      border-bottom: 1px solid #1e6fa5;
      padding: 14px 24px 10px 24px;
      display: flex;
      align-items: baseline;
      gap: 12px;
  }
  .terminal-header h1 {
      font-size: 1.45rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      color: #89dceb;
      margin: 0;
  }
  .terminal-header span {
      font-size: 0.72rem;
      color: #6c7086;
      letter-spacing: 0.15em;
      text-transform: uppercase;
  }

  /* ── KPI tiles ── */
  .kpi-grid { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
  .kpi-tile {
      background: #0d1117;
      border: 1px solid #1e2a3a;
      border-radius: 6px;
      padding: 10px 18px;
      min-width: 140px;
      flex: 1;
  }
  .kpi-tile .label {
      font-size: 0.65rem;
      color: #6c7086;
      letter-spacing: 0.12em;
      text-transform: uppercase;
  }
  .kpi-tile .value {
      font-size: 1.35rem;
      font-weight: 700;
      color: #cdd6f4;
      margin-top: 2px;
  }
  .kpi-tile .value.up   { color: #a6e3a1; }
  .kpi-tile .value.down { color: #f38ba8; }
  .kpi-tile .value.neutral { color: #89dceb; }

  /* ── Signal badge ── */
  .badge {
      display: inline-block;
      border-radius: 4px;
      padding: 2px 8px;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.08em;
  }
  .badge.buy  { background: #1a3a2a; color: #a6e3a1; border: 1px solid #a6e3a1; }
  .badge.sell { background: #3a1a2a; color: #f38ba8; border: 1px solid #f38ba8; }
  .badge.none { background: #1a1e2a; color: #6c7086; border: 1px solid #313244; }

  /* ── Table ── */
  .signal-table { font-size: 0.78rem; }
  [data-testid="stDataFrame"] { border: 1px solid #1e2a3a; border-radius: 6px; }

  /* ── Sidebar labels ── */
  .stSelectbox label, .stSlider label, .stNumberInput label {
      font-size: 0.72rem !important;
      color: #89dceb !important;
      letter-spacing: 0.08em;
      text-transform: uppercase;
  }

  /* ── Tab strip ── */
  [data-testid="stTab"] { color: #6c7086 !important; font-size: 0.82rem; }
  [aria-selected="true"] { color: #89dceb !important; border-bottom: 2px solid #89dceb !important; }

  /* ── Divider ── */
  hr { border-color: #1e2a3a; margin: 8px 0; }

  /* ── Backtest stat cards ── */
  .bt-stat-grid { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
  .bt-stat {
      background: #0d1117;
      border: 1px solid #1e2a3a;
      border-radius: 6px;
      padding: 8px 14px;
      min-width: 120px;
      flex: 1;
  }
  .bt-stat .s-label {
      font-size: 0.62rem;
      color: #6c7086;
      letter-spacing: 0.1em;
      text-transform: uppercase;
  }
  .bt-stat .s-value {
      font-size: 1.1rem;
      font-weight: 700;
      color: #cdd6f4;
      margin-top: 2px;
  }
  .bt-stat .s-value.pos { color: #a6e3a1; }
  .bt-stat .s-value.neg { color: #f38ba8; }
  .bt-stat .s-value.neu { color: #89dceb; }

  /* ── Trade log table result colouring ── */
  .win-row  { background-color: #1a3a2a !important; }
  .loss-row { background-color: #3a1a1a !important; }
  .open-row { background-color: #1a1e2a !important; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ROLLING_WINDOW: int = 20          # periods for support/resistance/vol SMA
VOLUME_MULT: float  = 1.2         # institutional volume threshold multiplier
RR_RATIO: float     = 2.5         # Risk:Reward — target = SL * RR_RATIO
LOOKBACK_DAYS: int  = 30          # yfinance download window
INTERVAL: str       = "15m"       # bar interval

# Fixed-point backtest parameters (spec: BUY +100 / -40, SELL -100 / +40)
BT_TARGET_PTS: float = 100.0      # points gained on a winning trade
BT_SL_PTS: float     = 40.0       # points risked on a losing trade
# Derived RR check: 100 / 40 = 2.5  ✓ matches RR_RATIO constant above
BT_MAX_BARS: int     = 96         # forward-scan cap per trade (96 × 15m = 24 h)

ASSET_MAP: dict = {
    "BankNifty (^NSEBANK)":  "^NSEBANK",
    "Nifty 50 (^NSEI)":      "^NSEI",
    "Nifty IT (^CNXIT)":     "^CNXIT",
    "S&P 500 (^GSPC)":       "^GSPC",
    "NASDAQ 100 (^NDX)":     "^NDX",
    "Gold (GC=F)":           "GC=F",
    "Crude Oil (CL=F)":      "CL=F",
    "Bitcoin (BTC-USD)":     "BTC-USD",
}


# ─────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────
def fetch_ohlcv(ticker: str, days: int = LOOKBACK_DAYS, interval: str = INTERVAL) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance.

    Safeguards applied:
      1. Multi-index column flattening — yfinance ≥0.2.x returns MultiIndex columns.
      2. Data sufficiency check — requires ≥50 rows for rolling calculations.
      3. Timezone normalization — converts to UTC-naive for Plotly compatibility.
    """
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)

    raw = yf.download(
        tickers=ticker,
        start=start_dt,
        end=end_dt,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw.empty:
        return pd.DataFrame()

    # ── Safeguard 1: flatten MultiIndex columns ──────────────────────────────
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # ── Normalize column names ───────────────────────────────────────────────
    raw.columns = [c.strip().capitalize() for c in raw.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(raw.columns)):
        return pd.DataFrame()

    df = raw[list(required)].copy()
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    # ── Timezone normalization ───────────────────────────────────────────────
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # ── Safeguard 2: minimum row check ──────────────────────────────────────
    if len(df) < 50:
        return pd.DataFrame()

    return df


# ─────────────────────────────────────────────
# SMC SIGNAL ENGINE
# ─────────────────────────────────────────────
def compute_smc_signals(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    vol_mult: float = VOLUME_MULT,
    rr_ratio: float = RR_RATIO,
) -> pd.DataFrame:
    """
    Core SMC Liquidity Sweep detection.

    Levels (shift=1 prevents lookahead bias):
      support    = rolling(window).min(Low) .shift(1)
      resistance = rolling(window).max(High).shift(1)
      vol_sma    = rolling(window).mean(Volume).shift(1)

    BUY signal (Bullish Sweep):
      candle.Low  < support   AND
      candle.Close > support  AND
      candle.Volume > vol_mult * vol_sma

    SELL signal (Bearish Sweep):
      candle.High  > resistance AND
      candle.Close < resistance AND
      candle.Volume > vol_mult * vol_sma

    Risk management (1 : rr_ratio):
      BUY  → SL = support level at signal bar
              TP = Close + (Close - SL) * rr_ratio
      SELL → SL = resistance level at signal bar
              TP = Close - (SL - Close) * rr_ratio
    """
    out = df.copy()

    # ── Level computation (fully vectorized) ────────────────────────────────
    out["support"]    = out["Low"].rolling(window).min().shift(1)
    out["resistance"] = out["High"].rolling(window).max().shift(1)
    out["vol_sma"]    = out["Volume"].rolling(window).mean().shift(1)
    out["vol_threshold"] = out["vol_sma"] * vol_mult

    # ── Signal masks ────────────────────────────────────────────────────────
    high_vol = out["Volume"] > out["vol_threshold"]

    bull_mask = (
        (out["Low"]   < out["support"])     &   # sweep below support
        (out["Close"] > out["support"])     &   # recovery close above
        high_vol
    )

    bear_mask = (
        (out["High"]  > out["resistance"])  &   # spike above resistance
        (out["Close"] < out["resistance"])  &   # rejection close below
        high_vol
    )

    out["signal"] = np.where(bull_mask, "BUY", np.where(bear_mask, "SELL", ""))

    # ── Risk/Reward levels ───────────────────────────────────────────────────
    # BUY  : SL at support, TP = Close + (Close - SL) * RR
    # SELL : SL at resistance, TP = Close - (SL - Close) * RR
    buy_sl  = out["support"]
    buy_tp  = out["Close"] + (out["Close"] - buy_sl) * rr_ratio

    sell_sl = out["resistance"]
    sell_tp = out["Close"] - (sell_sl - out["Close"]) * rr_ratio

    out["sl"] = np.where(bull_mask, buy_sl,  np.where(bear_mask, sell_sl, np.nan))
    out["tp"] = np.where(bull_mask, buy_tp,  np.where(bear_mask, sell_tp, np.nan))

    # ── Candle range & relative volume ──────────────────────────────────────
    out["candle_range"]  = out["High"] - out["Low"]
    out["rel_volume"]    = (out["Volume"] / out["vol_sma"]).round(2)

    return out


# ─────────────────────────────────────────────
# CHARTING ENGINE  (v2 — precision rebuild)
# ─────────────────────────────────────────────
def build_chart(df: pd.DataFrame, ticker: str, n_candles: int = 100) -> go.Figure:
    """
    Dual-panel Plotly chart — last `n_candles` bars only.

    Panel 1 (75%):
      · Candlestick — green/red OHLC bodies
      · Support     — dashed green line  (rolling 20p Low min, shift 1)
      · Resistance  — dashed red line    (rolling 20p High max, shift 1)
      · BUY markers — green ▲ below candle Low  (offset = mean_range × 0.35)
      · SELL markers — red ▼ above candle High  (offset = mean_range × 0.35)
      · SL/TP extension lines — rendered as Scatter traces, legend-only by default
        so they don't clutter the default view but are togglable.

    Panel 2 (25%):
      · Volume bars  — colour-matched to candle direction
      · Vol SMA (20) — cyan reference line

    Design constraints honoured:
      · template="plotly_dark"
      · margin=dict(l=0, r=0, t=36, b=0)
      · xaxis_rangeslider_visible=False on the price panel (xaxis1)
      · Marker offset is price-range-relative, not a fixed multiplier —
        prevents overlap on high-absolute-value indices (BankNifty ~50 000)
        and crypto (BTC ~60 000) alike.
    """
    # ── Slice to last n_candles ───────────────────────────────────────────────
    view = df.tail(n_candles).copy()

    # Price-range-relative offset for signal markers.
    # Using mean candle range of the visible window × 0.35 means the triangle
    # tip clears the wick by roughly one-third of a typical candle's range —
    # large enough to be distinct, small enough not to float into adjacent bars.
    mean_range: float = float(view["candle_range"].mean())
    marker_offset: float = mean_range * 0.35

    # ── Subplots ──────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.018,
    )

    # ────────────────────────────────────────────────────────────────────────
    # LAYER 1 — Candlestick
    # ────────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            name="Price",
            increasing_line_color="#a6e3a1",
            decreasing_line_color="#f38ba8",
            increasing_fillcolor="#a6e3a1",
            decreasing_fillcolor="#f38ba8",
            whiskerwidth=0.3,
            line=dict(width=1),
            hoverinfo="x+y",
        ),
        row=1, col=1,
    )

    # ────────────────────────────────────────────────────────────────────────
    # LAYER 2 — Support  (dashed green — retail buy-stop cluster floor)
    # ────────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["support"],
            mode="lines",
            name="Support (20p Low)",
            line=dict(
                color="#a6e3a1",   # green — institutional demand zone
                width=1.5,
                dash="dash",       # clearly distinguishable from price
            ),
            opacity=0.85,
            hovertemplate="Support: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ────────────────────────────────────────────────────────────────────────
    # LAYER 3 — Resistance  (dashed red — retail sell-stop cluster ceiling)
    # ────────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["resistance"],
            mode="lines",
            name="Resistance (20p High)",
            line=dict(
                color="#f38ba8",   # red — institutional supply zone
                width=1.5,
                dash="dash",
            ),
            opacity=0.85,
            hovertemplate="Resistance: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ────────────────────────────────────────────────────────────────────────
    # LAYER 4 — BUY signal triangles  (▲ below Low)
    # ────────────────────────────────────────────────────────────────────────
    buys = view[view["signal"] == "BUY"]

    # Always add the trace — empty x/y renders nothing but keeps legend entry
    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=(buys["Low"] - marker_offset) if not buys.empty else pd.Series(dtype=float),
            mode="markers+text",
            name="Bullish Sweep ▲",
            marker=dict(
                symbol="triangle-up",
                size=15,
                color="#a6e3a1",
                line=dict(color="#0d1117", width=1.5),
            ),
            text=["BUY"] * len(buys),
            textposition="bottom center",
            textfont=dict(
                size=8,
                color="#a6e3a1",
                family="JetBrains Mono, monospace",
            ),
            hovertemplate=(
                "<b>BULLISH SWEEP</b><br>"
                "Time : %{x}<br>"
                "Close: %{customdata[0]:.2f}<br>"
                "SL   : %{customdata[1]:.2f}<br>"
                "TP   : %{customdata[2]:.2f}<br>"
                "RelVol: %{customdata[3]:.2f}×"
                "<extra></extra>"
            ),
            customdata=(
                buys[["Close", "sl", "tp", "rel_volume"]].values
                if not buys.empty
                else np.empty((0, 4))
            ),
        ),
        row=1, col=1,
    )

    # ────────────────────────────────────────────────────────────────────────
    # LAYER 5 — SELL signal triangles  (▼ above High)
    # ────────────────────────────────────────────────────────────────────────
    sells = view[view["signal"] == "SELL"]

    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=(sells["High"] + marker_offset) if not sells.empty else pd.Series(dtype=float),
            mode="markers+text",
            name="Bearish Sweep ▼",
            marker=dict(
                symbol="triangle-down",
                size=15,
                color="#f38ba8",
                line=dict(color="#0d1117", width=1.5),
            ),
            text=["SELL"] * len(sells),
            textposition="top center",
            textfont=dict(
                size=8,
                color="#f38ba8",
                family="JetBrains Mono, monospace",
            ),
            hovertemplate=(
                "<b>BEARISH SWEEP</b><br>"
                "Time : %{x}<br>"
                "Close: %{customdata[0]:.2f}<br>"
                "SL   : %{customdata[1]:.2f}<br>"
                "TP   : %{customdata[2]:.2f}<br>"
                "RelVol: %{customdata[3]:.2f}×"
                "<extra></extra>"
            ),
            customdata=(
                sells[["Close", "sl", "tp", "rel_volume"]].values
                if not sells.empty
                else np.empty((0, 4))
            ),
        ),
        row=1, col=1,
    )

    # ────────────────────────────────────────────────────────────────────────
    # LAYER 6 — SL / TP horizontal extensions
    # Rendered as Scatter traces (not shapes) so they are legend-togglable.
    # Default visibility = "legendonly" — clean default view, opt-in to see levels.
    # Each signal gets its own SL segment and TP segment extending to the
    # right edge of the visible window.
    # ────────────────────────────────────────────────────────────────────────
    right_edge = view.index[-1]

    # Collect all SL coords and all TP coords into two single multi-segment
    # traces using None as a gap marker — avoids per-signal add_trace() loops.
    sl_x, sl_y, tp_x, tp_y = [], [], [], []

    sig_bars = view[view["signal"] != ""]
    for ts, bar in sig_bars.iterrows():
        if pd.notna(bar["sl"]):
            sl_x.extend([ts, right_edge, None])
            sl_y.extend([bar["sl"], bar["sl"], None])
        if pd.notna(bar["tp"]):
            tp_x.extend([ts, right_edge, None])
            tp_y.extend([bar["tp"], bar["tp"], None])

    if sl_x:
        fig.add_trace(
            go.Scatter(
                x=sl_x, y=sl_y,
                mode="lines",
                name="Stop Loss levels",
                line=dict(color="#f38ba8", width=0.9, dash="longdash"),
                opacity=0.55,
                visible="legendonly",   # hidden by default — toggle in legend
                hoverinfo="skip",
            ),
            row=1, col=1,
        )
    if tp_x:
        fig.add_trace(
            go.Scatter(
                x=tp_x, y=tp_y,
                mode="lines",
                name="Take Profit levels",
                line=dict(color="#a6e3a1", width=0.9, dash="longdash"),
                opacity=0.55,
                visible="legendonly",
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # ────────────────────────────────────────────────────────────────────────
    # LAYER 7 — Volume bars (colour-matched to candle direction)
    # ────────────────────────────────────────────────────────────────────────
    vol_colors = np.where(view["Close"] >= view["Open"], "#a6e3a1", "#f38ba8").tolist()

    fig.add_trace(
        go.Bar(
            x=view.index,
            y=view["Volume"],
            name="Volume",
            marker=dict(
                color=vol_colors,
                line=dict(width=0),
            ),
            opacity=0.50,
            showlegend=False,
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # ────────────────────────────────────────────────────────────────────────
    # LAYER 8 — Volume SMA reference line
    # ────────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["vol_sma"],
            mode="lines",
            name=f"Vol SMA ({ROLLING_WINDOW})",
            line=dict(color="#89dceb", width=1.4),
            hovertemplate="Vol SMA: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # ────────────────────────────────────────────────────────────────────────
    # LAYOUT
    # ────────────────────────────────────────────────────────────────────────
    # Build the title string dynamically so it reflects the actual visible range
    t_start = view.index[0].strftime("%d %b")
    t_end   = view.index[-1].strftime("%d %b '%y")
    n_buy_vis  = int((view["signal"] == "BUY").sum())
    n_sell_vis = int((view["signal"] == "SELL").sum())

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e14",
        plot_bgcolor="#0a0e14",
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(
            text=(
                f"<b style='color:#89dceb'>{ticker}</b>"
                f"<span style='color:#6c7086'>  ·  SMC Liquidity Sweep  ·  {INTERVAL}  ·  "
                f"last {n_candles} bars  ({t_start} → {t_end})  ·  "
                f"<span style='color:#a6e3a1'>{n_buy_vis}▲</span> "
                f"<span style='color:#f38ba8'>{n_sell_vis}▼</span></span>"
            ),
            font=dict(family="JetBrains Mono, monospace", size=12),
            x=0.005,
            xanchor="left",
        ),
        legend=dict(
            orientation="h",
            x=0,
            y=1.055,
            font=dict(size=10, color="#6c7086", family="JetBrains Mono, monospace"),
            bgcolor="rgba(0,0,0,0)",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0d1117",
            bordercolor="#1e2a3a",
            font_size=11,
            font_family="JetBrains Mono, monospace",
        ),
        # Suppress the default rangeslider on the price panel (xaxis)
        xaxis=dict(rangeslider=dict(visible=False)),
    )

    # Y-axis styling — price panel
    fig.update_yaxes(
        row=1, col=1,
        gridcolor="#1a2235",
        gridwidth=0.5,
        zerolinecolor="#1e2a3a",
        tickfont=dict(size=10, color="#6c7086", family="JetBrains Mono, monospace"),
        tickformat=",.2f",
        side="right",            # price axis on the right — Bloomberg/TradingView convention
        showgrid=True,
    )

    # Y-axis styling — volume panel
    fig.update_yaxes(
        row=2, col=1,
        gridcolor="#1a2235",
        gridwidth=0.5,
        zerolinecolor="#1e2a3a",
        tickfont=dict(size=9, color="#6c7086", family="JetBrains Mono, monospace"),
        tickformat=".2s",        # e.g. "1.2M" instead of "1200000"
        side="right",
        showgrid=True,
    )

    # X-axis styling — shared
    fig.update_xaxes(
        gridcolor="#1a2235",
        gridwidth=0.5,
        tickfont=dict(size=9, color="#6c7086", family="JetBrains Mono, monospace"),
        showgrid=False,
        zeroline=False,
        # Ensure rangeslider is off on all x-axes (make_subplots can create xaxis2)
        rangeslider=dict(visible=False),
    )

    return fig


# ─────────────────────────────────────────────
# KPI HELPER
# ─────────────────────────────────────────────
def build_kpi_html(df: pd.DataFrame) -> str:
    last  = df.iloc[-1]
    close = last["Close"]
    open_ = df.iloc[-2]["Close"] if len(df) > 1 else last["Open"]
    chg   = close - open_
    pct   = (chg / open_) * 100 if open_ != 0 else 0
    direction_cls = "up" if chg >= 0 else "down"
    arrow = "▲" if chg >= 0 else "▼"

    n_buy  = int((df["signal"] == "BUY").sum())
    n_sell = int((df["signal"] == "SELL").sum())
    last_sig = df[df["signal"] != ""].iloc[-1]["signal"] if (df["signal"] != "").any() else "—"
    sig_cls = "up" if last_sig == "BUY" else ("down" if last_sig == "SELL" else "neutral")

    support_val    = f"{last['support']:.2f}"
    resistance_val = f"{last['resistance']:.2f}"
    rel_vol        = f"{last['rel_volume']:.2f}×"

    tiles = [
        ("LTP", f"{close:,.2f}", direction_cls),
        (f"Change {arrow}", f"{chg:+.2f} ({pct:+.2f}%)", direction_cls),
        ("Last Signal", last_sig, sig_cls),
        ("Buy Sweeps", str(n_buy), "up"),
        ("Sell Sweeps", str(n_sell), "down"),
        ("Support", support_val, "neutral"),
        ("Resistance", resistance_val, "neutral"),
        ("Rel Volume", rel_vol, "neutral"),
    ]

    html = '<div class="kpi-grid">'
    for label, value, cls in tiles:
        html += f'''
        <div class="kpi-tile">
          <div class="label">{label}</div>
          <div class="value {cls}">{value}</div>
        </div>'''
    html += '</div>'
    return html


# ─────────────────────────────────────────────
# SIGNAL LOG TABLE
# ─────────────────────────────────────────────
def build_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    signals = df[df["signal"] != ""].copy()
    if signals.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "Datetime":   signals.index.strftime("%Y-%m-%d %H:%M"),
        "Signal":     signals["signal"],
        "Close":      signals["Close"].round(2),
        "SL":         signals["sl"].round(2),
        "TP":         signals["tp"].round(2),
        "Support":    signals["support"].round(2),
        "Resistance": signals["resistance"].round(2),
        "Rel Vol":    signals["rel_volume"],
    })
    out = out.sort_values("Datetime", ascending=False).reset_index(drop=True)
    return out



# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def run_backtest(
    df: pd.DataFrame,
    target_pts: float = BT_TARGET_PTS,
    sl_pts: float     = BT_SL_PTS,
    max_bars: int     = BT_MAX_BARS,
) -> pd.DataFrame:
    """
    Fixed-point walk-forward backtest over SMC Liquidity Sweep signals.

    Entry  : Close price of the signal candle (next-bar open is the realistic
             alternative, but Close is used here per spec and is labelled clearly).
    Exit scan: iterate forward bar-by-bar from entry+1 up to max_bars.

    BUY trade:
      tp_price = entry + target_pts      (limit fill — exact level)
      sl_price = entry - sl_pts          (stop fill  — exact level)
      Each forward candle is tested:
        · hit_tp  → candle High  >= tp_price
        · hit_sl  → candle Low   <= sl_price
        · Both in same candle     → Loss  (conservative: assume SL filled first)
        · Only TP                 → Win
        · Only SL                 → Loss
        · Neither within max_bars → Open (P&L at last close)

    SELL trade:
      tp_price = entry - target_pts
      sl_price = entry + sl_pts
        · hit_tp  → candle Low   <= tp_price
        · hit_sl  → candle High  >= sl_price
        · Both same candle        → Loss

    Implementation note:
      The inner scan uses pre-extracted NumPy arrays (highs, lows, closes,
      timestamps) indexed by integer position. This avoids per-row .loc /
      .iloc overhead and is the correct pattern when each trade's horizon
      starts at a different position — standard Pandas vectorization cannot
      express variable-origin forward scans.

    Returns a DataFrame with one row per completed or open trade, columns:
      entry_time, exit_time, type, entry_px, exit_px, pnl_pts, result
    """
    # ── Guard ────────────────────────────────────────────────────────────────
    signal_rows = df[df["signal"] != ""]
    if signal_rows.empty:
        return pd.DataFrame()

    # ── Extract NumPy arrays once — avoids repeated pandas indexing overhead ─
    all_highs:  np.ndarray = df["High"].to_numpy(dtype=np.float64)
    all_lows:   np.ndarray = df["Low"].to_numpy(dtype=np.float64)
    all_closes: np.ndarray = df["Close"].to_numpy(dtype=np.float64)
    all_times:  np.ndarray = df.index.to_numpy()   # datetime64 or Timestamp array
    n_bars: int = len(df)

    # Map datetime index → integer position for O(1) lookup
    time_to_pos: dict = {t: i for i, t in enumerate(df.index)}

    records: list[dict] = []

    for entry_time, sig_row in signal_rows.iterrows():
        sig_type: str    = sig_row["signal"]           # "BUY" or "SELL"
        entry_px: float  = float(sig_row["Close"])

        # ── Level definitions ────────────────────────────────────────────────
        if sig_type == "BUY":
            tp_price = entry_px + target_pts
            sl_price = entry_px - sl_pts
        else:  # SELL
            tp_price = entry_px - target_pts
            sl_price = entry_px + sl_pts

        # ── Forward scan ─────────────────────────────────────────────────────
        entry_pos: int = time_to_pos[entry_time]
        scan_start: int = entry_pos + 1                # never look at signal bar itself
        scan_end: int   = min(scan_start + max_bars, n_bars)

        result:   str   = "Open"
        exit_px:  float = all_closes[-1]               # default: last close
        exit_time        = all_times[-1]

        for j in range(scan_start, scan_end):
            bar_high:  float = all_highs[j]
            bar_low:   float = all_lows[j]

            if sig_type == "BUY":
                hit_tp = bar_high >= tp_price
                hit_sl = bar_low  <= sl_price
            else:
                hit_tp = bar_low  <= tp_price
                hit_sl = bar_high >= sl_price

            if hit_tp and hit_sl:
                # Both levels breached in the same candle →
                # conservative assumption: stop loss filled first.
                result    = "Loss"
                exit_px   = sl_price
                exit_time = all_times[j]
                break
            elif hit_tp:
                result    = "Win"
                exit_px   = tp_price
                exit_time = all_times[j]
                break
            elif hit_sl:
                result    = "Loss"
                exit_px   = sl_price
                exit_time = all_times[j]
                break
        # end forward scan

        pnl_pts: float = (exit_px - entry_px) if sig_type == "BUY" else (entry_px - exit_px)

        records.append({
            "entry_time": entry_time,
            "exit_time":  exit_time,
            "type":       sig_type,
            "entry_px":   round(entry_px, 2),
            "exit_px":    round(exit_px,  2),
            "pnl_pts":    round(pnl_pts,  2),
            "result":     result,
        })

    if not records:
        return pd.DataFrame()

    trades = pd.DataFrame(records)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"])

    # ── Cumulative P&L column (excluding open trades from running sum) ────────
    closed = trades["result"].isin(["Win", "Loss"])
    trades["cum_pnl"] = trades.loc[closed, "pnl_pts"].cumsum().reindex(trades.index).ffill().fillna(0)

    return trades


# ─────────────────────────────────────────────
# BACKTEST SUMMARY STATS
# ─────────────────────────────────────────────
def compute_bt_stats(trades: pd.DataFrame) -> dict:
    """
    Derives key performance metrics from the trades DataFrame.
    Only closed trades (Win / Loss) are used for rate and expectancy
    calculations; open positions are excluded to avoid survivorship bias.
    """
    closed = trades[trades["result"].isin(["Win", "Loss"])].copy()
    open_  = trades[trades["result"] == "Open"]

    n_total  = len(trades)
    n_closed = len(closed)
    n_open   = len(open_)

    if n_closed == 0:
        return {
            "n_total": n_total, "n_closed": 0, "n_open": n_open,
            "n_wins": 0, "n_losses": 0,
            "win_rate": 0.0, "total_pnl": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0,
            "profit_factor": 0.0, "max_consec_loss": 0,
        }

    wins   = closed[closed["result"] == "Win"]
    losses = closed[closed["result"] == "Loss"]

    n_wins   = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_closed if n_closed else 0.0

    total_pnl  = float(closed["pnl_pts"].sum())
    avg_win    = float(wins["pnl_pts"].mean())   if n_wins   else 0.0
    avg_loss   = float(losses["pnl_pts"].mean()) if n_losses else 0.0   # negative number

    # Expectancy = (win_rate × avg_win) + (loss_rate × avg_loss)
    # Positive expectancy → edge exists over this sample.
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Profit factor = gross_profit / |gross_loss|
    gross_profit = float(wins["pnl_pts"].sum())   if n_wins   else 0.0
    gross_loss   = abs(float(losses["pnl_pts"].sum())) if n_losses else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Maximum consecutive losses — important for drawdown intuition
    consec = 0
    max_consec = 0
    for r in closed["result"].tolist():
        if r == "Loss":
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    return {
        "n_total":        n_total,
        "n_closed":       n_closed,
        "n_open":         n_open,
        "n_wins":         n_wins,
        "n_losses":       n_losses,
        "win_rate":       win_rate,
        "total_pnl":      total_pnl,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "expectancy":     expectancy,
        "profit_factor":  profit_factor,
        "max_consec_loss": max_consec,
    }


# ─────────────────────────────────────────────
# BACKTEST CHART HELPERS
# ─────────────────────────────────────────────
def build_equity_curve(trades: pd.DataFrame) -> go.Figure:
    """
    Step-line equity curve of cumulative P&L (points) over trade sequence.
    Only closed trades are plotted. Open trades are annotated separately.
    """
    closed = trades[trades["result"].isin(["Win", "Loss"])].copy().reset_index(drop=True)

    if closed.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0e14",
                          plot_bgcolor="#0a0e14", margin=dict(l=0, r=0, t=28, b=0))
        return fig

    cum_pnl = closed["pnl_pts"].cumsum()
    colors  = ["#a6e3a1" if v >= 0 else "#f38ba8" for v in cum_pnl]

    fig = go.Figure()

    # Zero baseline
    fig.add_hline(y=0, line=dict(color="#313244", width=1, dash="dot"))

    # Equity curve — filled area + line
    fig.add_trace(go.Scatter(
        x=closed.index + 1,   # trade number (1-indexed)
        y=cum_pnl,
        mode="lines+markers",
        name="Cumulative P&L",
        line=dict(color="#89dceb", width=2, shape="hv"),   # step-line
        fill="tozeroy",
        fillcolor="rgba(137,220,235,0.08)",
        marker=dict(
            color=colors,
            size=7,
            symbol=["triangle-up" if r == "Win" else "triangle-down"
                    for r in closed["result"]],
            line=dict(color="#0d1117", width=1),
        ),
        hovertemplate=(
            "Trade #%{x}<br>"
            "Cum P&L : %{y:+.1f} pts<br>"
            "%{customdata[0]} @ %{customdata[1]}<extra></extra>"
        ),
        customdata=closed[["type", "entry_time"]].apply(
            lambda r: [r["type"], r["entry_time"].strftime("%d %b %H:%M")], axis=1
        ).tolist(),
    ))

    # Per-trade P&L bars as secondary reference (right axis)
    bar_colors = ["#a6e3a1" if p > 0 else "#f38ba8" for p in closed["pnl_pts"]]
    fig.add_trace(go.Bar(
        x=closed.index + 1,
        y=closed["pnl_pts"],
        name="Trade P&L",
        marker=dict(color=bar_colors, opacity=0.40, line=dict(width=0)),
        yaxis="y2",
        hovertemplate="Trade #%{x}  P&L: %{y:+.1f} pts<extra></extra>",
    ))

    last_pnl = float(cum_pnl.iloc[-1])
    pnl_color = "#a6e3a1" if last_pnl >= 0 else "#f38ba8"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e14",
        plot_bgcolor="#0a0e14",
        margin=dict(l=0, r=0, t=32, b=0),
        title=dict(
            text=(
                f"Equity Curve  ·  {len(closed)} closed trades  ·  "
                f"<span style='color:{pnl_color}'>{last_pnl:+.1f} pts cumulative</span>"
            ),
            font=dict(family="JetBrains Mono, monospace", size=12, color="#89dceb"),
            x=0.005, xanchor="left",
        ),
        legend=dict(
            orientation="h", x=0, y=1.06,
            font=dict(size=10, color="#6c7086"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0d1117", font_size=11,
                        font_family="JetBrains Mono, monospace"),
        xaxis=dict(
            title="Trade #",
            gridcolor="#1a2235", tickfont=dict(size=9, color="#6c7086"),
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="Cumulative P&L (pts)",
            gridcolor="#1a2235", tickfont=dict(size=9, color="#6c7086"),
            side="left", tickformat="+,.1f",
        ),
        yaxis2=dict(
            overlaying="y", side="right",
            showgrid=False,
            tickfont=dict(size=9, color="#6c7086"),
            tickformat="+,.1f",
            title="Per-trade P&L (pts)",
        ),
        bargap=0.3,
    )
    return fig


def build_result_donut(stats: dict) -> go.Figure:
    """Compact win/loss/open donut chart for the stats panel."""
    labels = ["Wins", "Losses", "Open"]
    values = [stats["n_wins"], stats["n_losses"], stats["n_open"]]
    colors = ["#a6e3a1", "#f38ba8", "#89dceb"]

    # Guard: Plotly raises a division error if all values are zero.
    # Return a blank dark figure instead of crashing.
    if sum(values) == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e14",
            plot_bgcolor="#0a0e14", margin=dict(l=0, r=0, t=28, b=0),
        )
        fig.add_annotation(
            text="No closed trades",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#6c7086", family="JetBrains Mono, monospace"),
            xanchor="center", yanchor="middle",
        )
        return fig

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0a0e14", width=3)),
        textinfo="label+percent",
        textfont=dict(size=11, family="JetBrains Mono, monospace"),
        hovertemplate="%{label}: %{value} trades (%{percent})<extra></extra>",
        sort=False,
    ))

    wr = stats["win_rate"] * 100
    wr_color = "#a6e3a1" if wr >= 50 else "#f38ba8"
    fig.add_annotation(
        text=f"<b style='color:{wr_color}'>{wr:.1f}%</b><br>"
             f"<span style='color:#6c7086;font-size:10px'>Win Rate</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, family="JetBrains Mono, monospace"),
        xanchor="center", yanchor="middle",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e14",
        plot_bgcolor="#0a0e14",
        margin=dict(l=0, r=0, t=28, b=0),
        title=dict(
            text="Outcome Distribution",
            font=dict(family="JetBrains Mono, monospace", size=12, color="#89dceb"),
            x=0.5, xanchor="center",
        ),
        legend=dict(
            orientation="v", x=1.02, y=0.5,
            font=dict(size=10, color="#6c7086"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
    )
    return fig

# ─────────────────────────────────────────────
# LIVE RADAR UI HELPERS
# ─────────────────────────────────────────────
def render_radar_banner(last_row: pd.Series) -> None:
    """
    Renders a full-width coloured banner when an active SMC signal
    exists on the most-recent candle. Silent (no banner) when flat.

    BUY  → green banner with pulse animation
    SELL → red banner with pulse animation
    Flat → subdued grey status bar
    """
    sig = last_row.get("signal", "")

    if sig == "BUY":
        bg      = "linear-gradient(90deg, #0d2b1a 0%, #1a4a2a 50%, #0d2b1a 100%)"
        border  = "#a6e3a1"
        icon    = "▲"
        label   = "ACTIVE BUY SIGNAL"
        sub     = "Bullish Liquidity Sweep Confirmed — Institutional Demand Detected"
        txt_col = "#a6e3a1"
        pulse   = "#a6e3a1"
    elif sig == "SELL":
        bg      = "linear-gradient(90deg, #2b0d0d 0%, #4a1a1a 50%, #2b0d0d 100%)"
        border  = "#f38ba8"
        icon    = "▼"
        label   = "ACTIVE SELL SIGNAL"
        sub     = "Bearish Liquidity Sweep Confirmed — Institutional Supply Detected"
        txt_col = "#f38ba8"
        pulse   = "#f38ba8"
    else:
        bg      = "#0d1117"
        border  = "#1e2a3a"
        icon    = "◉"
        label   = "MONITORING — NO ACTIVE SIGNAL"
        sub     = "Last candle closed neutral. Watching for next SMC sweep."
        txt_col = "#6c7086"
        pulse   = "#6c7086"

    ts_str = ""
    if hasattr(last_row.name, "strftime"):
        ts_str = last_row.name.strftime("Last bar: %d %b %Y  %H:%M UTC")

    st.markdown(
        f"""
        <style>
          @keyframes radar-pulse {{
            0%   {{ box-shadow: 0 0 0 0   {pulse}44; }}
            70%  {{ box-shadow: 0 0 0 10px {pulse}00; }}
            100% {{ box-shadow: 0 0 0 0   {pulse}00; }}
          }}
          .radar-banner {{
            background: {bg};
            border: 1px solid {border};
            border-left: 4px solid {border};
            border-radius: 6px;
            padding: 14px 22px;
            margin-bottom: 12px;
            {"animation: radar-pulse 1.8s infinite;" if sig else ""}
            display: flex;
            align-items: center;
            justify-content: space-between;
          }}
          .radar-banner .rb-left  {{ display:flex; align-items:center; gap:14px; }}
          .radar-banner .rb-icon  {{ font-size:2rem; color:{txt_col}; line-height:1; }}
          .radar-banner .rb-label {{ font-size:1.05rem; font-weight:700;
                                     color:{txt_col}; letter-spacing:0.1em;
                                     font-family:'JetBrains Mono',monospace; }}
          .radar-banner .rb-sub   {{ font-size:0.68rem; color:#6c7086;
                                     margin-top:3px; letter-spacing:0.06em; }}
          .radar-banner .rb-ts    {{ font-size:0.62rem; color:#45475a;
                                     font-family:'JetBrains Mono',monospace; }}
        </style>
        <div class="radar-banner">
          <div class="rb-left">
            <div class="rb-icon">{icon}</div>
            <div>
              <div class="rb-label">{label}</div>
              <div class="rb-sub">{sub}</div>
            </div>
          </div>
          <div class="rb-ts">{ts_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_radar_metrics(last_row: pd.Series, bt_target: float, bt_sl: float) -> None:
    """
    Four st.metric boxes scoped to the current signal state on the latest bar.

    Boxes:
      · Live Price  — last Close, delta vs previous close
      · Target      — fixed-point TP from entry (bt_target pts)
      · Stop Loss   — fixed-point SL from entry (bt_sl pts)
      · Risk:Reward — bt_target / bt_sl (dimensionless ratio)

    When no signal is active, Target and Stop Loss display the dynamic
    SMC TP/SL levels from the signal engine (not the fixed BT levels).
    This gives the trader actionable levels even outside a confirmed entry.
    """
    sig        = last_row.get("signal", "")
    close      = float(last_row["Close"])
    prev_close = float(last_row.get("prev_close", close))
    delta_pts  = close - prev_close
    delta_pct  = (delta_pts / prev_close * 100) if prev_close != 0 else 0.0

    # Derive TP / SL from the latest candle's signal state
    if sig == "BUY":
        tp_display  = close + bt_target
        sl_display  = close - bt_sl
        rr_display  = bt_target / bt_sl if bt_sl > 0 else 0.0
        tp_label    = f"Target  (+{bt_target:.0f} pts)"
        sl_label    = f"Stop Loss  (−{bt_sl:.0f} pts)"
    elif sig == "SELL":
        tp_display  = close - bt_target
        sl_display  = close + bt_sl
        rr_display  = bt_target / bt_sl if bt_sl > 0 else 0.0
        tp_label    = f"Target  (−{bt_target:.0f} pts)"
        sl_label    = f"Stop Loss  (+{bt_sl:.0f} pts)"
    else:
        # No active signal — show dynamic SMC engine levels
        tp_display  = float(last_row["tp"])  if pd.notna(last_row.get("tp"))  else float("nan")
        sl_display  = float(last_row["sl"])  if pd.notna(last_row.get("sl"))  else float("nan")
        rr_display  = bt_target / bt_sl if bt_sl > 0 else 0.0
        tp_label    = "Last TP Level"
        sl_label    = "Last SL Level"

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        label="Live Price",
        value=f"{close:,.2f}",
        delta=f"{delta_pts:+.2f} ({delta_pct:+.2f}%)",
    )
    c2.metric(
        label=tp_label,
        value=f"{tp_display:,.2f}" if not np.isnan(tp_display) else "—",
        delta=f"{'+' if sig == 'BUY' else ''}{bt_target:.0f} pts" if sig else None,
        delta_color="normal" if sig == "BUY" else ("inverse" if sig == "SELL" else "off"),
    )
    c3.metric(
        label=sl_label,
        value=f"{sl_display:,.2f}" if not np.isnan(sl_display) else "—",
        delta=f"{'−' if sig == 'BUY' else '+'}{bt_sl:.0f} pts" if sig else None,
        delta_color="inverse",
    )
    c4.metric(
        label="Risk : Reward",
        value=f"1 : {rr_display:.2f}",
        delta=f"{bt_target:.0f} / {bt_sl:.0f} pts",
        delta_color="off",
    )


def build_equity_curve_timeseries(trades: pd.DataFrame) -> go.Figure:
    """
    Point Equity Curve plotted against calendar time (entry_time on x-axis),
    not trade sequence number. This matches the spec: 'cumulative points
    gained over the 30 days'.

    Only closed trades contribute to the curve. Open trades are shown as
    ghost markers at the right edge with an 'OPEN' annotation.

    Chart anatomy:
      · Primary y  — cumulative P&L line (step-hv, filled area)
      · Secondary y — per-trade P&L bars (colour-coded win/loss)
      · Zero baseline
      · Drawdown shading — negative-equity regions filled red
    """
    closed = trades[trades["result"].isin(["Win", "Loss"])].copy()
    open_  = trades[trades["result"] == "Open"].copy()

    if closed.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e14",
            plot_bgcolor="#0a0e14", margin=dict(l=0, r=0, t=36, b=0),
            title=dict(
                text="Point Equity Curve — no closed trades yet",
                font=dict(family="JetBrains Mono, monospace", size=12, color="#6c7086"),
                x=0.005, xanchor="left",
            ),
        )
        fig.add_annotation(
            text="No closed trades to plot.",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#6c7086", family="JetBrains Mono, monospace"),
            xanchor="center", yanchor="middle",
        )
        return fig

    closed = closed.sort_values("entry_time").reset_index(drop=True)
    cum_pnl = closed["pnl_pts"].cumsum()
    last_pnl   = float(cum_pnl.iloc[-1])
    pnl_color  = "#a6e3a1" if last_pnl >= 0 else "#f38ba8"
    point_colors = ["#a6e3a1" if v >= 0 else "#f38ba8" for v in cum_pnl]

    fig = go.Figure()

    # ── Zero baseline ─────────────────────────────────────────────────────
    fig.add_hline(y=0, line=dict(color="#313244", width=1, dash="dot"))

    # ── Equity curve: filled area under the step-line ─────────────────────
    fig.add_trace(go.Scatter(
        x=closed["entry_time"],
        y=cum_pnl,
        mode="lines+markers",
        name="Cumulative P&L",
        line=dict(color="#89dceb", width=2.5, shape="hv"),
        fill="tozeroy",
        fillcolor="rgba(137,220,235,0.07)",
        marker=dict(
            color=point_colors,
            size=8,
            symbol=["triangle-up" if r == "Win" else "triangle-down"
                    for r in closed["result"]],
            line=dict(color="#0a0e14", width=1.2),
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b>  %{x|%d %b %H:%M}<br>"
            "P&L this trade : <b>%{customdata[1]:+.1f} pts</b><br>"
            "Cumulative     : <b>%{y:+.1f} pts</b>"
            "<extra></extra>"
        ),
        customdata=list(zip(closed["type"], closed["pnl_pts"])),
    ))

    # ── Per-trade bars on secondary y ─────────────────────────────────────
    bar_colors = ["#a6e3a1" if p > 0 else "#f38ba8" for p in closed["pnl_pts"]]
    fig.add_trace(go.Bar(
        x=closed["entry_time"],
        y=closed["pnl_pts"],
        name="Per-trade P&L",
        marker=dict(color=bar_colors, opacity=0.35, line=dict(width=0)),
        yaxis="y2",
        hovertemplate="%{x|%d %b %H:%M}  <b>%{y:+.1f} pts</b><extra></extra>",
    ))

    # ── Open trades — ghost markers at their entry times ──────────────────
    if not open_.empty:
        fig.add_trace(go.Scatter(
            x=open_["entry_time"],
            y=[last_pnl] * len(open_),
            mode="markers+text",
            name="Open (unresolved)",
            marker=dict(
                symbol="circle-open",
                size=10,
                color="#89dceb",
                line=dict(color="#89dceb", width=2),
            ),
            text=["OPEN"] * len(open_),
            textposition="top center",
            textfont=dict(size=8, color="#89dceb", family="JetBrains Mono, monospace"),
            hovertemplate=(
                "OPEN trade  %{x|%d %b %H:%M}<br>"
                "Not yet resolved within scan window<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e14",
        plot_bgcolor="#0a0e14",
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(
            text=(
                f"Point Equity Curve  ·  30-day SMC Backtest  ·  "
                f"{len(closed)} closed trades  ·  "
                f"<span style='color:{pnl_color}'>{last_pnl:+.1f} pts net</span>"
            ),
            font=dict(family="JetBrains Mono, monospace", size=12, color="#89dceb"),
            x=0.005, xanchor="left",
        ),
        legend=dict(
            orientation="h", x=0, y=1.055,
            font=dict(size=10, color="#6c7086"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0d1117", bordercolor="#1e2a3a",
            font_size=11, font_family="JetBrains Mono, monospace",
        ),
        xaxis=dict(
            title=None,
            gridcolor="#1a2235",
            tickfont=dict(size=9, color="#6c7086"),
            tickformat="%d %b\n%H:%M",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="Cumulative P&L (pts)",
            gridcolor="#1a2235",
            tickfont=dict(size=9, color="#6c7086"),
            side="left",
            tickformat="+,.0f",
            zeroline=True,
            zerolinecolor="#313244",
            zerolinewidth=1,
        ),
        yaxis2=dict(
            overlaying="y", side="right",
            showgrid=False,
            tickfont=dict(size=9, color="#6c7086"),
            tickformat="+,.0f",
            title="Per-trade P&L (pts)",
        ),
        bargap=0.2,
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐯 QuantBengal Pro")
    st.markdown("**SMC Terminal**  `v1.0`")
    st.markdown("---")

    selected_label = st.selectbox(
        "Asset",
        list(ASSET_MAP.keys()),
        index=0,
    )
    ticker = ASSET_MAP[selected_label]

    st.markdown("---")
    st.markdown("**Strategy Parameters**")

    roll_window = st.slider("Rolling Window (bars)", 10, 50, ROLLING_WINDOW, step=5)
    vol_multiplier = st.slider("Volume Multiplier", 1.0, 2.5, VOLUME_MULT, step=0.1,
                               format="%.1f×")
    rr = st.slider("Risk : Reward Ratio", 1.5, 5.0, RR_RATIO, step=0.5,
                   format="1 : %.1f")

    st.markdown("---")
    st.markdown("**Backtest Parameters**")
    bt_target = st.number_input("Target (pts)", min_value=10.0, max_value=500.0,
                                value=BT_TARGET_PTS, step=10.0,
                                help="Fixed-point profit target per trade.")
    bt_sl     = st.number_input("Stop Loss (pts)", min_value=5.0, max_value=200.0,
                                value=BT_SL_PTS, step=5.0,
                                help="Fixed-point stop loss per trade.")
    bt_rr_display = bt_target / bt_sl if bt_sl > 0 else 0
    st.markdown(
        f"<div style='font-size:0.68rem; color:#89dceb; margin-top:-6px'>"
        f"Implied RR → 1 : {bt_rr_display:.2f}</div>",
        unsafe_allow_html=True,
    )
    bt_max_bars = st.slider("Max Hold (bars)", 12, 200, BT_MAX_BARS, step=4,
                            help="Max 15-min bars to scan forward for exit. "
                                 f"Default {BT_MAX_BARS} bars = 24 h.")

    st.markdown("---")
    refresh = st.button("⟳  Refresh Data", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.65rem; color:#6c7086; line-height:1.7'>
    <b style='color:#89dceb'>BUY Trigger</b><br>
    Low < Support(20) AND<br>
    Close > Support(20) AND<br>
    Vol > {:.1f}× Vol SMA<br><br>
    <b style='color:#89dceb'>SELL Trigger</b><br>
    High > Resistance(20) AND<br>
    Close < Resistance(20) AND<br>
    Vol > {:.1f}× Vol SMA<br><br>
    <b style='color:#89dceb'>Risk Management</b><br>
    Ratio  1 : {:.1f}<br>
    SL → liquidity level<br>
    TP → Close ± (risk × RR)
    </div>
    """.format(vol_multiplier, vol_multiplier, rr), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="terminal-header">
  <h1>🐯 QuantBengal Pro SMC Terminal</h1>
  <span>Smart Money Concepts · Liquidity Sweep Detection · {INTERVAL} bars · 30-day window</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA FETCH + PROCESSING  (no @st.cache_data)
# ─────────────────────────────────────────────
with st.spinner(f"Fetching {ticker} · {INTERVAL} · 30d …"):
    raw_df = fetch_ohlcv(ticker, days=LOOKBACK_DAYS, interval=INTERVAL)

if raw_df.empty:
    st.error(
        f"**Data pipeline returned no usable rows for `{ticker}`.**  "
        "Possible causes: market closed, ticker unsupported for intraday, "
        "or yfinance rate limit. Try a different asset or refresh."
    )
    st.stop()

processed_df = compute_smc_signals(
    raw_df,
    window=roll_window,
    vol_mult=vol_multiplier,
    rr_ratio=rr,
)
# prev_close needed by render_radar_metrics for live delta calculation
processed_df["prev_close"] = processed_df["Close"].shift(1)


# ─────────────────────────────────────────────
# KPI BAR
# ─────────────────────────────────────────────
st.markdown(build_kpi_html(processed_df), unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BACKTEST  (run once, reused across tabs)
# ─────────────────────────────────────────────
trades_df = run_backtest(
    processed_df,
    target_pts=bt_target,
    sl_pts=bt_sl,
    max_bars=bt_max_bars,
)
bt_stats = compute_bt_stats(trades_df) if not trades_df.empty else {}


# ─────────────────────────────────────────────
# MAIN TABS  (final spec layout)
# ─────────────────────────────────────────────
tab_radar, tab_backtest, tab_signals, tab_data = st.tabs([
    "📡  Institutional Live Radar",
    "🔬  SMC Performance Backtest",
    "⚡  Signal Log",
    "🗂  Raw Data",
])

# ══════════════════════════════════════════════
# TAB 1 — INSTITUTIONAL LIVE RADAR
# ══════════════════════════════════════════════
with tab_radar:
    last_bar = processed_df.iloc[-1]

    # ── Signal banner ─────────────────────────────────────────────────────
    render_radar_banner(last_bar)

    # ── Four metric boxes: Live Price / Target / SL / RR ─────────────────
    render_radar_metrics(last_bar, bt_target=bt_target, bt_sl=bt_sl)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Candles slider (scoped to this tab) ───────────────────────────────
    n_candles_radar = st.slider(
        "Visible candles",
        min_value=50, max_value=300, value=100, step=10,
        key="radar_candles",
        help="Adjusts how many 15-min bars are rendered. "
             "Signal engine always uses the full 30-day history.",
    )

    # ── Candlestick chart ─────────────────────────────────────────────────
    st.plotly_chart(
        build_chart(processed_df, ticker, n_candles=n_candles_radar),
        use_container_width=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )

    # ── Legend caption ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='font-size:0.67rem; color:#45475a; margin-top:-8px; line-height:1.9'>
        <span style='color:#a6e3a1'>&#9650; green triangle</span> = Bullish Liquidity Sweep (BUY)
        &nbsp;&middot;&nbsp;
        <span style='color:#f38ba8'>&#9660; red triangle</span> = Bearish Liquidity Sweep (SELL)
        &nbsp;&middot;&nbsp;
        <span style='color:#a6e3a1'>&#8212;&#8212; dashed green</span> = Support (20p rolling Low)
        &nbsp;&middot;&nbsp;
        <span style='color:#f38ba8'>&#8212;&#8212; dashed red</span> = Resistance (20p rolling High)
        &nbsp;&middot;&nbsp;
        SL / TP levels hidden by default &mdash; toggle via chart legend.
        &nbsp;&middot;&nbsp;
        <em>Simulation only. Not financial advice.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════
# TAB 2 — SMC PERFORMANCE BACKTEST
# ══════════════════════════════════════════════
with tab_backtest:
    if trades_df.empty or not bt_stats:
        st.info(
            "No SMC signals detected in the current 30-day window — "
            "backtest has no trades to simulate. "
            "Try reducing the Volume Multiplier or Rolling Window in the sidebar."
        )
    else:
        s  = bt_stats
        pf = s["profit_factor"]

        # ── Four headline KPIs (spec-required) ───────────────────────────
        st.markdown(
            "<div style='font-size:0.72rem; color:#89dceb; letter-spacing:0.1em; "
            "text-transform:uppercase; margin-bottom:6px'>Performance Summary</div>",
            unsafe_allow_html=True,
        )

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric(
            label="Total Trades",
            value=str(s["n_total"]),
            delta=f"{s['n_closed']} closed · {s['n_open']} open",
            delta_color="off",
        )
        win_rate_pct = s["win_rate"] * 100
        kc2.metric(
            label="Win Rate",
            value=f"{win_rate_pct:.1f}%",
            delta=f"{s['n_wins']}W / {s['n_losses']}L",
            delta_color="normal" if win_rate_pct >= 50 else "inverse",
        )
        kc3.metric(
            label="Net Points Accrued",
            value=f"{s['total_pnl']:+.1f} pts",
            delta=f"Expectancy {s['expectancy']:+.2f} pts/trade",
            delta_color="normal" if s["total_pnl"] >= 0 else "inverse",
        )
        kc4.metric(
            label="Profit Factor",
            value=f"{pf:.2f}×" if pf != float("inf") else "∞",
            delta="Gross Wins ÷ Gross Losses",
            delta_color="normal" if pf >= 1.0 else "inverse",
        )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ── Secondary stat cards ──────────────────────────────────────────
        sec_cards = [
            ("Avg Win",          f"{s['avg_win']:+.1f} pts",          "pos"),
            ("Avg Loss",         f"{s['avg_loss']:+.1f} pts",          "neg"),
            ("Expectancy",       f"{s['expectancy']:+.2f} pts/trade",
             "pos" if s["expectancy"] >= 0 else "neg"),
            ("Max Consec. Loss", str(s["max_consec_loss"]),             "neg"),
            ("Target / SL",      f"{bt_target:.0f} / {bt_sl:.0f} pts","neu"),
            ("Implied RR",       f"1 : {bt_rr_display:.2f}",          "neu"),
        ]
        sec_html = '<div class="bt-stat-grid">'
        for lbl, val, cls in sec_cards:
            sec_html += (
                f'<div class="bt-stat">'
                f'<div class="s-label">{lbl}</div>'
                f'<div class="s-value {cls}">{val}</div>'
                f'</div>'
            )
        sec_html += '</div>'
        st.markdown(sec_html, unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Methodology callout ───────────────────────────────────────────
        st.markdown(
            f"""
            <div style='font-size:0.67rem; color:#6c7086; background:#0d1117;
                        border:1px solid #1e2a3a; border-radius:6px;
                        padding:8px 16px; line-height:1.9; margin-bottom:12px'>
            <b style='color:#89dceb'>Simulation methodology</b> &nbsp;&middot;&nbsp;
            Entry at <b>signal candle Close</b> &nbsp;&middot;&nbsp;
            Target <b style='color:#a6e3a1'>+{bt_target:.0f} pts</b> limit fill &nbsp;&middot;&nbsp;
            Stop <b style='color:#f38ba8'>&#8722;{bt_sl:.0f} pts</b> stop fill &nbsp;&middot;&nbsp;
            Same-candle TP+SL breach &rarr; <b style='color:#f38ba8'>Loss</b> (conservative)
            &nbsp;&middot;&nbsp;
            Max hold <b>{bt_max_bars} bars ({bt_max_bars * 15 // 60}h)</b> &nbsp;&middot;&nbsp;
            No slippage or commission modelled &nbsp;&middot;&nbsp;
            Open trades excluded from Win Rate and Expectancy.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Point Equity Curve (time-series, x = calendar date) ──────────
        st.markdown(
            "<div style='font-size:0.72rem; color:#89dceb; letter-spacing:0.1em; "
            "text-transform:uppercase; margin-bottom:4px'>Point Equity Curve — 30 Days</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_equity_curve_timeseries(trades_df),
            use_container_width=True,
            config={"scrollZoom": False, "displayModeBar": False},
        )

        # ── Detailed Trade Log ────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.72rem; color:#89dceb; letter-spacing:0.1em; "
            "text-transform:uppercase; margin:10px 0 4px'>Detailed Trade Log</div>",
            unsafe_allow_html=True,
        )

        display_trades = trades_df.copy()
        display_trades["entry_time"] = display_trades["entry_time"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        display_trades["exit_time"] = display_trades["exit_time"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        display_trades.index = range(1, len(display_trades) + 1)
        display_trades.index.name = "#"
        display_trades.columns = [
            "Entry Time", "Exit Time", "Type",
            "Entry Px", "Exit Px", "P&L (pts)", "Result", "Cum P&L",
        ]

        def _style_trades(row):
            if row["Result"] == "Win":
                return ["background-color:#1a3a2a"] * len(row)
            if row["Result"] == "Loss":
                return ["background-color:#3a1a1a"] * len(row)
            return ["background-color:#1a1e2a"] * len(row)

        def _colour_pnl(val):
            try:
                return (
                    "color:#a6e3a1;font-weight:700"
                    if float(val) >= 0
                    else "color:#f38ba8;font-weight:700"
                )
            except (ValueError, TypeError):
                return ""

        def _colour_result(val):
            if val == "Win":  return "color:#a6e3a1;font-weight:700"
            if val == "Loss": return "color:#f38ba8;font-weight:700"
            return "color:#89dceb"

        def _colour_type(val):
            if val == "BUY":  return "color:#a6e3a1;font-weight:700"
            if val == "SELL": return "color:#f38ba8;font-weight:700"
            return ""

        st.dataframe(
            display_trades.style
            .apply(_style_trades, axis=1)
            .map(_colour_pnl,    subset=["P&L (pts)", "Cum P&L"])
            .map(_colour_result, subset=["Result"])
            .map(_colour_type,   subset=["Type"])
            .format({
                "Entry Px":  "{:.2f}",
                "Exit Px":   "{:.2f}",
                "P&L (pts)": "{:+.2f}",
                "Cum P&L":   "{:+.2f}",
            }),
            use_container_width=True,
            height=420,
        )

        st.markdown(
            f"<div style='font-size:0.62rem; color:#45475a; margin-top:4px'>"
            f"{len(trades_df)} total trades &nbsp;&middot;&nbsp; "
            f"{s['n_closed']} closed ({s['n_wins']}W / {s['n_losses']}L) "
            f"&nbsp;&middot;&nbsp; {s['n_open']} open at data end &nbsp;&middot;&nbsp; "
            f"Period: {processed_df.index[0].strftime('%d %b')} "
            f"&rarr; {processed_df.index[-1].strftime('%d %b %Y')}"
            f"</div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════
# TAB 3 — SIGNAL LOG
# ══════════════════════════════════════════════
with tab_signals:
    sig_df = build_signal_table(processed_df)

    if sig_df.empty:
        st.info(
            "No SMC liquidity sweep signals detected in the current window. "
            "Try widening the lookback or reducing the Volume Multiplier in the sidebar."
        )
    else:
        total_buy  = int((sig_df["Signal"] == "BUY").sum())
        total_sell = int((sig_df["Signal"] == "SELL").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Signals",  len(sig_df))
        c2.metric("Bullish Sweeps", total_buy,  delta=f"+{total_buy}")
        c3.metric("Bearish Sweeps", total_sell, delta=f"−{total_sell}",
                  delta_color="inverse")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        def colour_signal(val):
            if val == "BUY":
                return "background-color:#1a3a2a; color:#a6e3a1; font-weight:700"
            if val == "SELL":
                return "background-color:#3a1a2a; color:#f38ba8; font-weight:700"
            return ""

        st.dataframe(
            sig_df.style.map(colour_signal, subset=["Signal"]),
            use_container_width=True,
            height=460,
        )

# ══════════════════════════════════════════════
# TAB 4 — RAW DATA
# ══════════════════════════════════════════════
with tab_data:
    display_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "support", "resistance", "vol_sma",
        "signal", "sl", "tp", "rel_volume",
    ]
    display_df = processed_df[display_cols].copy()

    # Stamp backtest result onto each signal bar for cross-reference
    display_df["bt_result"] = ""
    if not trades_df.empty:
        bt_result_map: dict = dict(
            zip(
                pd.to_datetime(trades_df["entry_time"]).dt.tz_localize(None),
                trades_df["result"],
            )
        )
        display_df["bt_result"] = display_df.index.map(
            lambda ts: bt_result_map.get(ts, "")
        )

    display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
    display_df = display_df.sort_index(ascending=False)

    fmt = {
        "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Close": "{:.2f}",
        "support": "{:.2f}", "resistance": "{:.2f}",
        "vol_sma": "{:,.0f}", "Volume": "{:,.0f}",
        "sl": "{:.2f}", "tp": "{:.2f}", "rel_volume": "{:.2f}",
    }

    def _colour_signal_raw(val):
        if val == "BUY":  return "background-color:#1a3a2a; color:#a6e3a1; font-weight:700"
        if val == "SELL": return "background-color:#3a1a2a; color:#f38ba8; font-weight:700"
        return ""

    def _colour_bt_result(val):
        if val == "Win":  return "color:#a6e3a1; font-weight:700"
        if val == "Loss": return "color:#f38ba8; font-weight:700"
        if val == "Open": return "color:#89dceb; font-weight:700"
        return "color:#45475a"

    def _colour_rel_vol(val):
        try:
            v = float(val)
            if v >= 2.0:                return "color:#f9e2af; font-weight:700"
            if v >= float(VOLUME_MULT): return "color:#cdd6f4"
        except (ValueError, TypeError):
            pass
        return "color:#45475a"

    st.dataframe(
        display_df.style
        .map(_colour_signal_raw, subset=["signal"])
        .map(_colour_bt_result,  subset=["bt_result"])
        .map(_colour_rel_vol,    subset=["rel_volume"])
        .format(fmt),
        use_container_width=True,
        height=500,
    )

    first_ts = processed_df.index[0].strftime("%Y-%m-%d %H:%M")
    last_ts  = processed_df.index[-1].strftime("%Y-%m-%d %H:%M")
    n_sigs   = int((processed_df["signal"] != "").sum())
    n_bt     = len(trades_df) if not trades_df.empty else 0

    st.markdown(
        f"<div style='font-size:0.62rem; color:#45475a; margin-top:4px; line-height:1.9'>"
        f"<b style='color:#6c7086'>Rows:</b> {len(processed_df):,} &nbsp;&middot;&nbsp; "
        f"<b style='color:#6c7086'>Period:</b> {first_ts} &rarr; {last_ts} &nbsp;&middot;&nbsp; "
        f"<b style='color:#6c7086'>Interval:</b> {INTERVAL} &nbsp;&middot;&nbsp; "
        f"<b style='color:#6c7086'>Signals:</b> {n_sigs} &nbsp;&middot;&nbsp; "
        f"<b style='color:#6c7086'>BT trades:</b> {n_bt} &nbsp;&middot;&nbsp; "
        f"bt_result: Win / Loss / Open / blank (non-signal bars)"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='font-size:0.62rem; color:#45475a; text-align:center; padding-bottom:8px'>"
    f"QuantBengal Pro SMC Terminal &nbsp;&middot;&nbsp; "
    f"Strategy: SMC Liquidity Sweep &nbsp;&middot;&nbsp; "
    f"Backtest: fixed {bt_target:.0f} pt target / {bt_sl:.0f} pt SL "
    f"(1&nbsp;:&nbsp;{bt_rr_display:.2f} RR) &nbsp;&middot;&nbsp; "
    f"All signals are probabilistic outcomes of historical pattern simulation. "
    f"Past statistics do not guarantee future performance. Not financial advice."
    f"</div>",
    unsafe_allow_html=True,
)

# ── Tab 1: Chart ─────────────────────────────
with tab_chart:
    # n_candles slider scoped to the chart tab — only re-renders the figure,
    # not the full app (sidebar sliders already force a full rerun).
    n_candles_input = st.slider(
        "Visible candles",
        min_value=50, max_value=300, value=100, step=10,
        help="Controls how many 15-min bars are rendered. "
             "Signal detection always uses the full downloaded history.",
    )

    fig = build_chart(processed_df, ticker, n_candles=n_candles_input)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    st.markdown("""
    <div style='font-size:0.68rem; color:#45475a; margin-top:-8px; line-height:1.9'>
    <span style='color:#a6e3a1'>&#8212;&#8212; green dashed</span> = Support (20p Low, shift 1) &mdash; retail buy-stop cluster floor
    &nbsp;&middot;&nbsp;
    <span style='color:#f38ba8'>&#8212;&#8212; red dashed</span> = Resistance (20p High, shift 1) &mdash; retail sell-stop cluster ceiling
    &nbsp;&middot;&nbsp;
    <span style='color:#a6e3a1'>&#9650;</span> Bullish Sweep below support
    &nbsp;&middot;&nbsp;
    <span style='color:#f38ba8'>&#9660;</span> Bearish Sweep above resistance
    &nbsp;&middot;&nbsp;
    SL / TP extension lines hidden by default &mdash; click legend entries to toggle.
    &nbsp;&middot;&nbsp;
    Historical simulation only. Not financial advice.
    </div>
    """, unsafe_allow_html=True)

# ── Tab 2: Signal Log ────────────────────────
with tab_signals:
    sig_df = build_signal_table(processed_df)

    if sig_df.empty:
        st.info("No SMC liquidity sweep signals detected in the current window. "
                "Try widening the lookback or reducing the volume multiplier.")
    else:
        total_buy  = int((sig_df["Signal"] == "BUY").sum())
        total_sell = int((sig_df["Signal"] == "SELL").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Signals",  len(sig_df))
        c2.metric("Bullish Sweeps", total_buy,  delta=f"+{total_buy}")
        c3.metric("Bearish Sweeps", total_sell, delta=f"-{total_sell}", delta_color="inverse")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Colour-code the Signal column
        def colour_signal(val):
            if val == "BUY":
                return "background-color: #1a3a2a; color: #a6e3a1; font-weight:700"
            elif val == "SELL":
                return "background-color: #3a1a2a; color: #f38ba8; font-weight:700"
            return ""

        styled = sig_df.style.map(colour_signal, subset=["Signal"])
        st.dataframe(styled, use_container_width=True, height=420)

# ── Tab 3: Backtest ──────────────────────────
with tab_backtest:
    if trades_df.empty:
        st.info(
            "No signals found in the current window — backtest has no trades to simulate. "
            "Try reducing the Volume Multiplier or Rolling Window in the sidebar."
        )
    else:
        # ── Stat card row ─────────────────────────────────────────────────
        s = bt_stats
        pnl_cls  = "pos" if s["total_pnl"] >= 0 else "neg"
        exp_cls  = "pos" if s["expectancy"] >= 0 else "neg"
        pf_cls   = "pos" if s["profit_factor"] >= 1.0 else "neg"
        wr_cls   = "pos" if s["win_rate"] >= 0.5 else "neg"

        stat_cards = [
            ("Total Trades",    str(s["n_total"]),                       "neu"),
            ("Closed",          str(s["n_closed"]),                      "neu"),
            ("Open",            str(s["n_open"]),                        "neu"),
            ("Wins",            str(s["n_wins"]),                        "pos"),
            ("Losses",          str(s["n_losses"]),                      "neg"),
            ("Win Rate",        f"{s['win_rate']*100:.1f}%",             wr_cls),
            ("Total P&L",       f"{s['total_pnl']:+.1f} pts",           pnl_cls),
            ("Avg Win",         f"{s['avg_win']:+.1f} pts",              "pos"),
            ("Avg Loss",        f"{s['avg_loss']:+.1f} pts",             "neg"),
            ("Expectancy",      f"{s['expectancy']:+.2f} pts/trade",     exp_cls),
            ("Profit Factor",   f"{s['profit_factor']:.2f}×",           pf_cls),
            ("Max Consec. Loss",str(s["max_consec_loss"]),               "neg"),
        ]

        html = '<div class="bt-stat-grid">'
        for lbl, val, cls in stat_cards:
            html += (
                f'<div class="bt-stat">'
                f'  <div class="s-label">{lbl}</div>'
                f'  <div class="s-value {cls}">{val}</div>'
                f'</div>'
            )
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Methodology callout ───────────────────────────────────────────
        st.markdown(
            f"""
            <div style='font-size:0.68rem; color:#6c7086; background:#0d1117;
                        border:1px solid #1e2a3a; border-radius:6px;
                        padding:8px 14px; line-height:1.8; margin-bottom:10px'>
            <b style='color:#89dceb'>Simulation rules</b> &nbsp;·&nbsp;
            Entry at <b>signal candle Close</b> &nbsp;·&nbsp;
            Target <b style='color:#a6e3a1'>+{bt_target:.0f} pts</b> (limit fill at exact level) &nbsp;·&nbsp;
            Stop <b style='color:#f38ba8'>−{bt_sl:.0f} pts</b> (stop fill at exact level) &nbsp;·&nbsp;
            Implied RR <b>1 : {bt_rr_display:.2f}</b> &nbsp;·&nbsp;
            Same-candle TP+SL breach → <b style='color:#f38ba8'>Loss</b> (conservative) &nbsp;·&nbsp;
            Max hold <b>{bt_max_bars} bars ({bt_max_bars*15//60} h)</b> &nbsp;·&nbsp;
            No slippage, commission, or position sizing modelled.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Charts: equity curve + donut ─────────────────────────────────
        col_eq, col_do = st.columns([0.68, 0.32])

        with col_eq:
            st.plotly_chart(
                build_equity_curve(trades_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col_do:
            st.plotly_chart(
                build_result_donut(bt_stats),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── Trade log table ───────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.72rem; color:#89dceb; "
            "letter-spacing:0.08em; text-transform:uppercase; margin:8px 0 4px'>Trade Log</div>",
            unsafe_allow_html=True,
        )

        display_trades = trades_df.copy()
        display_trades["entry_time"] = display_trades["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_trades["exit_time"]  = display_trades["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_trades.index = range(1, len(display_trades) + 1)
        display_trades.index.name = "#"
        display_trades.columns = [
            "Entry Time", "Exit Time", "Type",
            "Entry Px", "Exit Px", "P&L (pts)", "Result", "Cum P&L",
        ]

        def _style_trades(row):
            if row["Result"] == "Win":
                bg = "background-color:#1a3a2a"
            elif row["Result"] == "Loss":
                bg = "background-color:#3a1a1a"
            else:
                bg = "background-color:#1a1e2a"
            return [bg] * len(row)

        def _colour_pnl(val):
            try:
                return "color:#a6e3a1;font-weight:700" if float(val) >= 0 \
                       else "color:#f38ba8;font-weight:700"
            except (ValueError, TypeError):
                return ""

        def _colour_result(val):
            if val == "Win":
                return "color:#a6e3a1;font-weight:700"
            elif val == "Loss":
                return "color:#f38ba8;font-weight:700"
            return "color:#89dceb"

        styled_trades = (
            display_trades.style
            .apply(_style_trades, axis=1)
            .map(_colour_pnl,    subset=["P&L (pts)", "Cum P&L"])
            .map(_colour_result, subset=["Result"])
            .format({
                "Entry Px": "{:.2f}",
                "Exit Px":  "{:.2f}",
                "P&L (pts)": "{:+.2f}",
                "Cum P&L":   "{:+.2f}",
            })
        )

        st.dataframe(styled_trades, use_container_width=True, height=400)

        st.markdown(
            f"<div style='font-size:0.62rem; color:#45475a; margin-top:4px'>"
            f"Showing {len(trades_df)} trades  ·  "
            f"{s['n_closed']} closed ({s['n_wins']} W / {s['n_losses']} L)  ·  "
            f"{s['n_open']} still open at data end  ·  "
            f"Open trades excluded from win rate and expectancy."
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Tab 4: Raw Data ───────────────────────────
with tab_data:
    display_cols = ["Open", "High", "Low", "Close", "Volume",
                    "support", "resistance", "vol_sma", "signal", "sl", "tp", "rel_volume"]
    display_df = processed_df[display_cols].copy()

    # ── Cross-reference backtest result onto signal bars ──────────────────
    # For every entry_time in trades_df, stamp the result onto the
    # corresponding bar in display_df so the two views are self-consistent.
    display_df["bt_result"] = ""
    if not trades_df.empty:
        # Build a map: entry_time (Timestamp, tz-naive) → result string
        bt_result_map: dict = dict(
            zip(
                pd.to_datetime(trades_df["entry_time"]).dt.tz_localize(None),
                trades_df["result"],
            )
        )
        # Align on the DatetimeIndex of display_df (already tz-naive)
        display_df["bt_result"] = display_df.index.map(
            lambda ts: bt_result_map.get(ts, "")
        )

    display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
    display_df = display_df.sort_index(ascending=False)

    # ── Column-level formatters ───────────────────────────────────────────
    fmt = {
        "Open":       "{:.2f}",
        "High":       "{:.2f}",
        "Low":        "{:.2f}",
        "Close":      "{:.2f}",
        "support":    "{:.2f}",
        "resistance": "{:.2f}",
        "vol_sma":    "{:,.0f}",
        "Volume":     "{:,.0f}",
        "sl":         "{:.2f}",
        "tp":         "{:.2f}",
        "rel_volume": "{:.2f}",
    }

    # ── Cell-level colourisers ────────────────────────────────────────────
    def _colour_signal_raw(val: str) -> str:
        if val == "BUY":
            return "background-color:#1a3a2a; color:#a6e3a1; font-weight:700"
        if val == "SELL":
            return "background-color:#3a1a2a; color:#f38ba8; font-weight:700"
        return ""

    def _colour_bt_result(val: str) -> str:
        if val == "Win":
            return "color:#a6e3a1; font-weight:700"
        if val == "Loss":
            return "color:#f38ba8; font-weight:700"
        if val == "Open":
            return "color:#89dceb; font-weight:700"
        return "color:#45475a"

    def _colour_rel_vol(val) -> str:
        try:
            v = float(val)
            if v >= 2.0:
                return "color:#f9e2af; font-weight:700"   # amber — very high vol
            if v >= float(VOLUME_MULT):
                return "color:#cdd6f4"                    # white — above threshold
        except (ValueError, TypeError):
            pass
        return "color:#45475a"                            # grey — below threshold

    styled_raw = (
        display_df.style
        .map(_colour_signal_raw, subset=["signal"])
        .map(_colour_bt_result,  subset=["bt_result"])
        .map(_colour_rel_vol,    subset=["rel_volume"])
        .format(fmt)
    )

    st.dataframe(styled_raw, use_container_width=True, height=480)

    # ── Footer metadata strip ─────────────────────────────────────────────
    first_ts = processed_df.index[0].strftime("%Y-%m-%d %H:%M")
    last_ts  = processed_df.index[-1].strftime("%Y-%m-%d %H:%M")
    n_sigs   = int((processed_df["signal"] != "").sum())
    n_bt     = len(trades_df) if not trades_df.empty else 0

    st.markdown(
        f"<div style='font-size:0.62rem; color:#45475a; margin-top:4px; line-height:1.9'>"
        f"<b style='color:#6c7086'>Rows:</b> {len(processed_df):,} &nbsp;·&nbsp; "
        f"<b style='color:#6c7086'>Period:</b> {first_ts} → {last_ts} &nbsp;·&nbsp; "
        f"<b style='color:#6c7086'>Interval:</b> {INTERVAL} &nbsp;·&nbsp; "
        f"<b style='color:#6c7086'>Signals:</b> {n_sigs} &nbsp;·&nbsp; "
        f"<b style='color:#6c7086'>BT trades stamped:</b> {n_bt} &nbsp;·&nbsp; "
        f"<b style='color:#6c7086'>bt_result column:</b> Win / Loss / Open / blank"
        f"</div>",
        unsafe_allow_html=True,
        help="bt_result reflects the backtest outcome for that signal bar. "
             "Blank rows are non-signal bars. "
             "Data sourced from Yahoo Finance via yfinance.",
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='font-size:0.62rem; color:#45475a; text-align:center; padding-bottom:8px'>"
    f"QuantBengal Pro SMC Terminal &nbsp;·&nbsp; "
    f"Strategy: SMC Liquidity Sweep &nbsp;·&nbsp; "
    f"Backtest: fixed {bt_target:.0f} pt target / {bt_sl:.0f} pt SL "
    f"(1 : {bt_rr_display:.2f} RR) &nbsp;·&nbsp; "
    f"All signals are probabilistic outcomes of historical pattern simulation. "
    f"Past statistics do not guarantee future performance. Not financial advice."
    f"</div>",
    unsafe_allow_html=True,
)
