# ============================================================
# QuantBengal Pro SMC Terminal  |  app.py
# Senior Quant Dev & Market Microstructure Analyst build
# Strategy: Smart Money Concepts (SMC) Liquidity Sweep Detection
# Stack: Python 3.11+ | Streamlit | Plotly | Pandas | yfinance
# Theme: High-Visibility Light Terminal
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
# GLOBAL STYLE (Light Terminal Overhaul)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base Light Theme ── */
  html, body, [data-testid="stAppViewContainer"] {
      background-color: #f8fafc;
      color: #0f172a;
      font-family: 'JetBrains Mono', 'Courier New', monospace;
  }
  [data-testid="stSidebar"] {
      background-color: #f1f5f9;
      border-right: 1px solid #cbd5e1;
  }

  /* ── Header strip ── */
  .terminal-header {
      background: linear-gradient(90deg, #f1f5f9 0%, #e2e8f0 60%, #f1f5f9 100%);
      border-bottom: 1px solid #0284c7;
      padding: 14px 24px 10px 24px;
      display: flex;
      align-items: baseline;
      gap: 12px;
  }
  .terminal-header h1 {
      font-size: 1.45rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      color: #0284c7;
      margin: 0;
  }
  .terminal-header span {
      font-size: 0.72rem;
      color: #475569;
      letter-spacing: 0.15em;
      text-transform: uppercase;
  }

  /* ── KPI tiles ── */
  .kpi-grid { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
  .kpi-tile {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 10px 18px;
      min-width: 140px;
      flex: 1;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .kpi-tile .label {
      font-size: 0.65rem;
      color: #475569;
      letter-spacing: 0.12em;
      text-transform: uppercase;
  }
  .kpi-tile .value {
      font-size: 1.35rem;
      font-weight: 700;
      color: #0f172a;
      margin-top: 2px;
  }
  .kpi-tile .value.up   { color: #16a34a; }
  .kpi-tile .value.down { color: #dc2626; }
  .kpi-tile .value.neutral { color: #0284c7; }

  /* ── Signal badge ── */
  .badge {
      display: inline-block;
      border-radius: 4px;
      padding: 2px 8px;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.08em;
  }
  .badge.buy  { background: #dcfce7; color: #16a34a; border: 1px solid #16a34a; }
  .badge.sell { background: #fee2e2; color: #dc2626; border: 1px solid #dc2626; }
  .badge.none { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

  /* ── Table ── */
  .signal-table { font-size: 0.78rem; }
  [data-testid="stDataFrame"] { border: 1px solid #cbd5e1; border-radius: 6px; background-color: #ffffff; }

  /* ── Sidebar labels ── */
  .stSelectbox label, .stSlider label, .stNumberInput label {
      font-size: 0.72rem !important;
      color: #0284c7 !important;
      letter-spacing: 0.08em;
      text-transform: uppercase;
  }

  /* ── Tab strip ── */
  [data-testid="stTab"] { color: #475569 !important; font-size: 0.82rem; }
  [aria-selected="true"] { color: #0284c7 !important; border-bottom: 2px solid #0284c7 !important; }

  /* ── Divider ── */
  hr { border-color: #cbd5e1; margin: 8px 0; }

  /* ── Backtest stat cards ── */
  .bt-stat-grid { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
  .bt-stat {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 8px 14px;
      min-width: 120px;
      flex: 1;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
  }
  .bt-stat .s-label {
      font-size: 0.62rem;
      color: #475569;
      letter-spacing: 0.1em;
      text-transform: uppercase;
  }
  .bt-stat .s-value {
      font-size: 1.1rem;
      font-weight: 700;
      color: #0f172a;
      margin-top: 2px;
  }
  .bt-stat .s-value.pos { color: #16a34a; }
  .bt-stat .s-value.neg { color: #dc2626; }
  .bt-stat .s-value.neu { color: #0284c7; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ROLLING_WINDOW: int = 20          
VOLUME_MULT: float  = 1.2         
RR_RATIO: float     = 2.5         
LOOKBACK_DAYS: int  = 30          
INTERVAL: str       = "15m"       

BT_TARGET_PTS: float = 100.0      
BT_SL_PTS: float     = 40.0       
BT_MAX_BARS: int     = 96         

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

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.strip().capitalize() for c in raw.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(raw.columns)):
        return pd.DataFrame()

    df = raw[list(required)].copy()
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

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
    out = df.copy()

    out["support"]    = out["Low"].rolling(window).min().shift(1)
    out["resistance"] = out["High"].rolling(window).max().shift(1)
    out["vol_sma"]    = out["Volume"].rolling(window).mean().shift(1)
    out["vol_threshold"] = out["vol_sma"] * vol_mult

    high_vol = out["Volume"] > out["vol_threshold"]

    bull_mask = (
        (out["Low"]   < out["support"])     &   
        (out["Close"] > out["support"])     &   
        high_vol
    )

    bear_mask = (
        (out["High"]  > out["resistance"])  &   
        (out["Close"] < out["resistance"])  &   
        high_vol
    )

    out["signal"] = np.where(bull_mask, "BUY", np.where(bear_mask, "SELL", ""))

    buy_sl  = out["support"]
    buy_tp  = out["Close"] + (out["Close"] - buy_sl) * rr_ratio

    sell_sl = out["resistance"]
    sell_tp = out["Close"] - (sell_sl - out["Close"]) * rr_ratio

    out["sl"] = np.where(bull_mask, buy_sl,  np.where(bear_mask, sell_sl, np.nan))
    out["tp"] = np.where(bull_mask, buy_tp,  np.where(bear_mask, sell_tp, np.nan))

    out["candle_range"]  = out["High"] - out["Low"]
    out["rel_volume"]    = (out["Volume"] / out["vol_sma"]).round(2)

    return out


# ─────────────────────────────────────────────
# CHARTING ENGINE (Light Theme Optimized)
# ─────────────────────────────────────────────
def build_chart(df: pd.DataFrame, ticker: str, n_candles: int = 100) -> go.Figure:
    view = df.tail(n_candles).copy()
    mean_range: float = float(view["candle_range"].mean())
    marker_offset: float = mean_range * 0.35

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.018,
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
            name="Price",
            increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
            increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
            whiskerwidth=0.3, line=dict(width=1), hoverinfo="x+y",
        ),
        row=1, col=1,
    )

    # 2. Support Line
    fig.add_trace(
        go.Scatter(
            x=view.index, y=view["support"], mode="lines",
            name="Support (20p Low)",
            line=dict(color="#16a34a", width=1.5, dash="dash"),
            opacity=0.85, hovertemplate="Support: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 3. Resistance Line
    fig.add_trace(
        go.Scatter(
            x=view.index, y=view["resistance"], mode="lines",
            name="Resistance (20p High)",
            line=dict(color="#dc2626", width=1.5, dash="dash"),
            opacity=0.85, hovertemplate="Resistance: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 4. BUY signals
    buys = view[view["signal"] == "BUY"]
    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=(buys["Low"] - marker_offset) if not buys.empty else pd.Series(dtype=float),
            mode="markers+text", name="Bullish Sweep ▲",
            marker=dict(symbol="triangle-up", size=15, color="#16a34a", line=dict(color="#ffffff", width=1.5)),
            text=["BUY"] * len(buys), textposition="bottom center",
            textfont=dict(size=8, color="#16a34a", family="JetBrains Mono, monospace"),
            hovertemplate="<b>BULLISH SWEEP</b><br>Time : %{x}<br>Close: %{customdata[0]:.2f}<br>SL   : %{customdata[1]:.2f}<br>TP   : %{customdata[2]:.2f}<extra></extra>",
            customdata=buys[["Close", "sl", "tp", "rel_volume"]].values if not buys.empty else np.empty((0, 4)),
        ),
        row=1, col=1,
    )

    # 5. SELL signals
    sells = view[view["signal"] == "SELL"]
    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=(sells["High"] + marker_offset) if not sells.empty else pd.Series(dtype=float),
            mode="markers+text", name="Bearish Sweep ▼",
            marker=dict(symbol="triangle-down", size=15, color="#dc2626", line=dict(color="#ffffff", width=1.5)),
            text=["SELL"] * len(sells), textposition="top center",
            textfont=dict(size=8, color="#dc2626", family="JetBrains Mono, monospace"),
            hovertemplate="<b>BEARISH SWEEP</b><br>Time : %{x}<br>Close: %{customdata[0]:.2f}<br>SL   : %{customdata[1]:.2f}<br>TP   : %{customdata[2]:.2f}<extra></extra>",
            customdata=sells[["Close", "sl", "tp", "rel_volume"]].values if not sells.empty else np.empty((0, 4)),
        ),
        row=1, col=1,
    )

    # 6. SL / TP horizontal extensions (Legend-only default)
    right_edge = view.index[-1]
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
                x=sl_x, y=sl_y, mode="lines", name="Stop Loss levels",
                line=dict(color="#dc2626", width=0.9, dash="longdash"),
                opacity=0.55, visible="legendonly", hoverinfo="skip",
            ),
            row=1, col=1,
        )
    if tp_x:
        fig.add_trace(
            go.Scatter(
                x=tp_x, y=tp_y, mode="lines", name="Take Profit levels",
                line=dict(color="#16a34a", width=0.9, dash="longdash"),
                opacity=0.55, visible="legendonly", hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # 7. Volume bars
    vol_colors = np.where(view["Close"] >= view["Open"], "#22c55e", "#ef4444").tolist()
    fig.add_trace(
        go.Bar(
            x=view.index, y=view["Volume"], name="Volume",
            marker=dict(color=vol_colors, line=dict(width=0)),
            opacity=0.50, showlegend=False, hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # 8. Volume SMA reference line
    fig.add_trace(
        go.Scatter(
            x=view.index, y=view["vol_sma"], mode="lines",
            name=f"Vol SMA ({ROLLING_WINDOW})",
            line=dict(color="#0284c7", width=1.4),
            hovertemplate="Vol SMA: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Layout styling (Optimized for Light Theme)
    t_start = view.index[0].strftime("%d %b")
    t_end   = view.index[-1].strftime("%d %b '%y")
    n_buy_vis  = int((view["signal"] == "BUY").sum())
    n_sell_vis = int((view["signal"] == "SELL").sum())

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(
            text=(
                f"<b style='color:#0284c7'>{ticker}</b>"
                f"<span style='color:#475569'>  ·  SMC Sweep  ·  {INTERVAL}  ·  "
                f"last {n_candles} bars  ({t_start} → {t_end})  ·  "
                f"<span style='color:#16a34a'>{n_buy_vis}▲</span> "
                f"<span style='color:#dc2626'>{n_sell_vis}▼</span></span>"
            ),
            font=dict(family="JetBrains Mono, monospace", size=12),
            x=0.005, xanchor="left",
        ),
        legend=dict(orientation="h", x=0, y=1.055, font=dict(size=10, color="#475569", family="JetBrains Mono, monospace"), bgcolor="rgba(255,255,255,0.9)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#cbd5e1", font_size=11, font_family="JetBrains Mono, monospace", font_color="#0f172a"),
        xaxis=dict(rangeslider=dict(visible=False)),
    )

    fig.update_yaxes(
        row=1, col=1, gridcolor="#f1f5f9", gridwidth=0.5, zerolinecolor="#cbd5e1",
        tickfont=dict(size=10, color="#475569", family="JetBrains Mono, monospace"),
        tickformat=",.2f", side="right", showgrid=True,
    )
    fig.update_yaxes(
        row=2, col=1, gridcolor="#f1f5f9", gridwidth=0.5, zerolinecolor="#cbd5e1",
        tickfont=dict(size=9, color="#475569", family="JetBrains Mono, monospace"),
        tickformat=".2s", side="right", showgrid=True,
    )
    fig.update_xaxes(
        gridcolor="#f1f5f9", gridwidth=0.5,
        tickfont=dict(size=9, color="#475569", family="JetBrains Mono, monospace"),
        showgrid=True, zeroline=False, rangeslider=dict(visible=False),
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
    signal_rows = df[df["signal"] != ""]
    if signal_rows.empty:
        return pd.DataFrame()

    all_highs:  np.ndarray = df["High"].to_numpy(dtype=np.float64)
    all_lows:   np.ndarray = df["Low"].to_numpy(dtype=np.float64)
    all_closes: np.ndarray = df["Close"].to_numpy(dtype=np.float64)
    all_times:  np.ndarray = df.index.to_numpy()
    n_bars: int = len(df)

    time_to_pos: dict = {t: i for i, t in enumerate(df.index)}
    records: list[dict] = []

    for entry_time, sig_row in signal_rows.iterrows():
        sig_type: str    = sig_row["signal"]
        entry_px: float  = float(sig_row["Close"])

        if sig_type == "BUY":
            tp_price = entry_px + target_pts
            sl_price = entry_px - sl_pts
        else:  
            tp_price = entry_px - target_pts
            sl_price = entry_px + sl_pts

        entry_pos: int = time_to_pos[entry_time]
        scan_start: int = entry_pos + 1                
        scan_end: int   = min(scan_start + max_bars, n_bars)

        result:   str   = "Open"
        exit_px:  float = all_closes[-1]               
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

    closed = trades["result"].isin(["Win", "Loss"])
    trades["cum_pnl"] = trades.loc[closed, "pnl_pts"].cumsum().reindex(trades.index).ffill().fillna(0)

    return trades


# ─────────────────────────────────────────────
# BACKTEST SUMMARY STATS
# ─────────────────────────────────────────────
def compute_bt_stats(trades: pd.DataFrame) -> dict:
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
    avg_loss   = float(losses["pnl_pts"].mean()) if n_losses else 0.0   

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    gross_profit = float(wins["pnl_pts"].sum())   if n_wins   else 0.0
    gross_loss   = abs(float(losses["pnl_pts"].sum())) if n_losses else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

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
# EQUITY TIMESERIES (Light Theme)
# ─────────────────────────────────────────────
def build_equity_curve_timeseries(trades: pd.DataFrame) -> go.Figure:
    closed = trades[trades["result"].isin(["Win", "Loss"])].copy()
    open_  = trades[trades["result"] == "Open"].copy()

    if closed.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        return fig

    closed = closed.sort_values("entry_time").reset_index(drop=True)
    cum_pnl = closed["pnl_pts"].cumsum()
    last_pnl   = float(cum_pnl.iloc[-1])
    pnl_color  = "#16a34a" if last_pnl >= 0 else "#dc2626"
    point_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in cum_pnl]

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#cbd5e1", width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=closed["entry_time"], y=cum_pnl, mode="lines+markers", name="Cumulative P&L",
        line=dict(color="#0284c7", width=2.5, shape="hv"),
        fill="tozeroy", fillcolor="rgba(2,132,199,0.05)",
        marker=dict(color=point_colors, size=8, symbol=["triangle-up" if r == "Win" else "triangle-down" for r in closed["result"]], line=dict(color="#ffffff", width=1.2)),
        hovertemplate="<b>%{customdata[0]}</b>  %{x|%d %b %H:%M}<br>P&L this trade : <b>%{customdata[1]:+.1f} pts</b><br>Cumulative     : <b>%{y:+.1f} pts</b><extra></extra>",
        customdata=list(zip(closed["type"], closed["pnl_pts"])),
    ))

    bar_colors = ["#22c55e" if p > 0 else "#ef4444" for p in closed["pnl_pts"]]
    fig.add_trace(go.Bar(
        x=closed["entry_time"], y=closed["pnl_pts"], name="Per-trade P&L",
        marker=dict(color=bar_colors, opacity=0.35, line=dict(width=0)),
        yaxis="y2", hovertemplate="%{x|%d %b %H:%M}  <b>%{y:+.1f} pts</b><extra></extra>",
    ))

    if not open_.empty:
        fig.add_trace(go.Scatter(
            x=open_["entry_time"], y=[last_pnl] * len(open_),
            mode="markers+text", name="Open (unresolved)",
            marker=dict(symbol="circle-open", size=10, color="#0284c7", line=dict(color="#0284c7", width=2)),
            text=["OPEN"] * len(open_), textposition="top center",
            textfont=dict(size=8, color="#0284c7", family="JetBrains Mono, monospace"),
        ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(
            text=(
                f"Point Equity Curve  ·  30-day SMC Backtest  ·  "
                f"{len(closed)} closed trades  ·  "
                f"<span style='color:{pnl_color}'>{last_pnl:+.1f} pts net</span>"
            ),
            font=dict(family="JetBrains Mono, monospace", size=12, color="#0284c7"),
            x=0.005, xanchor="left",
        ),
        legend=dict(orientation="h", x=0, y=1.055, font=dict(size=10, color="#475569"), bgcolor="rgba(255,255,255,0.9)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#cbd5e1", font_size=11, font_family="JetBrains Mono, monospace", font_color="#0f172a"),
        xaxis=dict(gridcolor="#f1f5f9", tickfont=dict(size=9, color="#475569"), tickformat="%d %b\n%H:%M"),
        yaxis=dict(title="Cumulative P&L (pts)", gridcolor="#f1f5f9", tickfont=dict(size=9, color="#475569"), side="left", tickformat="+,.0f"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, tickfont=dict(size=9, color="#475569"), tickformat="+,.0f", title="Per-trade P&L (pts)"),
    )
    return fig


# ─────────────────────────────────────────────
# RESULT DONUT CHART (Light Theme)
# ─────────────────────────────────────────────
def build_result_donut(stats: dict) -> go.Figure:
    labels = ["Wins", "Losses", "Open"]
    values = [stats["n_wins"], stats["n_losses"], stats["n_open"]]
    colors = ["#22c55e", "#ef4444", "#0284c7"]

    if sum(values) == 0:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        fig.add_annotation(text="No closed trades", x=0.5, y=0.5, showarrow=False, font=dict(size=13, color="#475569"))
        return fig

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=3)),
        textinfo="label+percent", textfont=dict(size=11, family="JetBrains Mono, monospace"),
        hovertemplate="%{label}: %{value} trades (%{percent})<extra></extra>",
        sort=False,
    ))

    wr = stats["win_rate"] * 100
    wr_color = "#16a34a" if wr >= 50 else "#dc2626"
    fig.add_annotation(
        text=f"<b style='color:{wr_color}'>{wr:.1f}%</b><br><span style='color:#475569;font-size:10px'>Win Rate</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(size=16, family="JetBrains Mono, monospace"),
        xanchor="center", yanchor="middle",
    )

    fig.update_layout(
        template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=28, b=0),
        title=dict(text="Outcome Distribution", font=dict(family="JetBrains Mono, monospace", size=12, color="#0284c7"), x=0.5, xanchor="center"),
        showlegend=True,
    )
    return fig


# ─────────────────────────────────────────────
# LIVE RADAR UI HELPERS
# ─────────────────────────────────────────────
def render_radar_banner(last_row: pd.Series) -> None:
    sig = last_row.get("signal", "")

    if sig == "BUY":
        bg      = "linear-gradient(90deg, #f0fdf4 0%, #dcfce7 50%, #f0fdf4 100%)"
        border  = "#16a34a"
        icon    = "▲"
        label   = "ACTIVE BUY SIGNAL"
        sub     = "Bullish Sweep Confirmed — Institutional Demand Detected"
        txt_col = "#16a34a"
        pulse   = "#16a34a"
    elif sig == "SELL":
        bg      = "linear-gradient(90deg, #fef2f2 0%, #fee2e2 50%, #fef2f2 100%)"
        border  = "#dc2626"
        icon    = "▼"
        label   = "ACTIVE SELL SIGNAL"
        sub     = "Bearish Sweep Confirmed — Institutional Supply Detected"
        txt_col = "#dc2626"
        pulse   = "#dc2626"
    else:
        bg      = "#f8fafc"
        border  = "#cbd5e1"
        icon    = "◉"
        label   = "MONITORING — NO ACTIVE SIGNAL"
        sub     = "Last candle closed neutral. Watching for next SMC sweep."
        txt_col = "#475569"
        pulse   = "#cbd5e1"

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
          .radar-banner .rb-label {{ font-size:1.05rem; font-weight:700; color:{txt_col}; letter-spacing:0.1em; }}
          .radar-banner .rb-sub   {{ font-size:0.68rem; color:#475569; margin-top:3px; }}
          .radar-banner .rb-ts    {{ font-size:0.62rem; color:#64748b; }}
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
    sig        = last_row.get("signal", "")
    close      = float(last_row["Close"])
    prev_close = float(last_row.get("prev_close", close))
    delta_pts  = close - prev_close
    delta_pct  = (delta_pts / prev_close * 100) if prev_close != 0 else 0.0

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
        tp_display  = float(last_row["tp"])  if pd.notna(last_row.get("tp"))  else float("nan")
        sl_display  = float(last_row["sl"])  if pd.notna(last_row.get("sl"))  else float("nan")
        rr_display  = bt_target / bt_sl if bt_sl > 0 else 0.0
        tp_label    = "Last TP Level"
        sl_label    = "Last SL Level"

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(label="Live Price", value=f"{close:,.2f}", delta=f"{delta_pts:+.2f} ({delta_pct:+.2f}%)")
    c2.metric(label=tp_label, value=f"{tp_display:,.2f}" if not np.isnan(tp_display) else "—", delta=f"{'+' if sig == 'BUY' else ''}{bt_target:.0f} pts" if sig else None, delta_color="normal" if sig == "BUY" else ("inverse" if sig == "SELL" else "off"))
    c3.metric(label=sl_label, value=f"{sl_display:,.2f}" if not np.isnan(sl_display) else "—", delta=f"{'−' if sig == 'BUY' else '+'}{bt_sl:.0f} pts" if sig else None, delta_color="inverse")
    c4.metric(label="Risk : Reward", value=f"1 : {rr_display:.2f}", delta=f"{bt_target:.0f} / {bt_sl:.0f} pts", delta_color="off")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐯 QuantBengal Pro")
    st.markdown("**SMC Terminal**  `v1.0`")
    st.markdown("---")

    selected_label = st.selectbox("Asset", list(ASSET_MAP.keys()), index=0)
    ticker = ASSET_MAP[selected_label]
    st.markdown("---")

    st.markdown("**Strategy Parameters**")
    roll_window = st.slider("Rolling Window (bars)", 10, 50, ROLLING_WINDOW, step=5)
    vol_multiplier = st.slider("Volume Multiplier", 1.0, 2.5, VOLUME_MULT, step=0.1, format="%.1f×")
    rr = st.slider("Risk : Reward Ratio", 1.5, 5.0, RR_RATIO, step=0.5, format="1 : %.1f")
    st.markdown("---")

    st.markdown("**Backtest Parameters**")
    bt_target = st.number_input("Target (pts)", min_value=10.0, max_value=500.0, value=BT_TARGET_PTS, step=10.0)
    bt_sl     = st.number_input("Stop Loss (pts)", min_value=5.0, max_value=200.0, value=BT_SL_PTS, step=5.0)
    bt_rr_display = bt_target / bt_sl if bt_sl > 0 else 0
    st.markdown(f"<div style='font-size:0.68rem; color:#0284c7; margin-top:-6px'>Implied RR → 1 : {bt_rr_display:.2f}</div>", unsafe_allow_html=True)
    bt_max_bars = st.slider("Max Hold (bars)", 12, 200, BT_MAX_BARS, step=4)

    st.markdown("---")
    refresh = st.button("⟳  Refresh Data", use_container_width=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="terminal-header">
  <h1>🐯 QuantBengal Pro SMC Terminal</h1>
  <span>Smart Money Concepts · Liquidity Sweep Detection · {INTERVAL} bars</span>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA FETCH + PROCESSING
# ─────────────────────────────────────────────
with st.spinner(f"Fetching {ticker} · {INTERVAL} · 30d …"):
    raw_df = fetch_ohlcv(ticker, days=LOOKBACK_DAYS, interval=INTERVAL)

if raw_df.empty:
    st.error(f"**Data pipeline returned no usable rows for `{ticker}`.** Market may be closed or rate-limited.")
    st.stop()

processed_df = compute_smc_signals(raw_df, window=roll_window, vol_mult=vol_multiplier, rr_ratio=rr)
processed_df["prev_close"] = processed_df["Close"].shift(1)


# ─────────────────────────────────────────────
# KPI BAR
# ─────────────────────────────────────────────
st.markdown(build_kpi_html(processed_df), unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BACKTEST EXECUTION
# ─────────────────────────────────────────────
trades_df = run_backtest(processed_df, target_pts=bt_target, sl_pts=bt_sl, max_bars=bt_max_bars)
bt_stats = compute_bt_stats(trades_df) if not trades_df.empty else {}


# ─────────────────────────────────────────────
# MAIN TABS (Merged, light theme layout)
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
    render_radar_banner(last_bar)
    render_radar_metrics(last_bar, bt_target=bt_target, bt_sl=bt_sl)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    n_candles_radar = st.slider("Visible candles", min_value=50, max_value=300, value=100, step=10, key="radar_candles")
    
    st.plotly_chart(
        build_chart(processed_df, ticker, n_candles=n_candles_radar),
        use_container_width=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )

    st.markdown(
        """
        <div style='font-size:0.67rem; color:#64748b; margin-top:-8px; line-height:1.9'>
        <span style='color:#16a34a'>&#9650; green triangle</span> = Bullish Liquidity Sweep (BUY)
        &nbsp;&middot;&nbsp;
        <span style='color:#dc2626'>&#9660; red triangle</span> = Bearish Liquidity Sweep (SELL)
        &nbsp;&middot;&nbsp;
        <span style='color:#16a34a'>&#8212;&#8212; dashed green</span> = Support (20p rolling Low)
        &nbsp;&middot;&nbsp;
        <span style='color:#dc2626'>&#8212;&#8212; dashed red</span> = Resistance (20p rolling High)
        &nbsp;&middot;&nbsp;
        SL / TP extension lines hidden by default &mdash; toggle via chart legend.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════
# TAB 2 — SMC PERFORMANCE BACKTEST (Merged and Optimized)
# ══════════════════════════════════════════════
with tab_backtest:
    if trades_df.empty or not bt_stats:
        st.info("No SMC signals detected in the current window to backtest.")
    else:
        s  = bt_stats
        pf = s["profit_factor"]

        st.markdown("<div style='font-size:0.72rem; color:#0284c7; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px'>Performance Summary</div>", unsafe_allow_html=True)

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric(label="Total Trades", value=str(s["n_total"]), delta=f"{s['n_closed']} closed · {s['n_open']} open", delta_color="off")
        win_rate_pct = s["win_rate"] * 100
        kc2.metric(label="Win Rate", value=f"{win_rate_pct:.1f}%", delta=f"{s['n_wins']}W / {s['n_losses']}L", delta_color="normal" if win_rate_pct >= 50 else "inverse")
        kc3.metric(label="Net Points Accrued", value=f"{s['total_pnl']:+.1f} pts", delta=f"Expectancy {s['expectancy']:+.2f} pts", delta_color="normal" if s["total_pnl"] >= 0 else "inverse")
        kc4.metric(label="Profit Factor", value=f"{pf:.2f}×" if pf != float("inf") else "∞", delta_color="normal" if pf >= 1.0 else "inverse")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        sec_cards = [
            ("Avg Win",          f"{s['avg_win']:+.1f} pts",          "pos"),
            ("Avg Loss",         f"{s['avg_loss']:+.1f} pts",          "neg"),
            ("Expectancy",       f"{s['expectancy']:+.2f} pts/trade", "pos" if s["expectancy"] >= 0 else "neg"),
            ("Max Consec. Loss", str(s["max_consec_loss"]),             "neg"),
            ("Target / SL",      f"{bt_target:.0f} / {bt_sl:.0f} pts","neu"),
            ("Implied RR",       f"1 : {bt_rr_display:.2f}",          "neu"),
        ]
        sec_html = '<div class="bt-stat-grid">'
        for lbl, val, cls in sec_cards:
            sec_html += f'<div class="bt-stat"><div class="s-label">{lbl}</div><div class="s-value {cls}">{val}</div></div>'
        sec_html += '</div>'
        st.markdown(sec_html, unsafe_allow_html=True)

        # Methodology callout
        st.markdown(
            f"""
            <div style='font-size:0.68rem; color:#64748b; background:#f8fafc;
                        border:1px solid #cbd5e1; border-radius:6px;
                        padding:8px 14px; line-height:1.8; margin-bottom:10px'>
            <b style='color:#0284c7'>Simulation rules</b> &nbsp;·&nbsp;
            Entry at <b>signal candle Close</b> &nbsp;·&nbsp;
            Target <b style='color:#16a34a'>+{bt_target:.0f} pts</b> (limit fill at exact level) &nbsp;·&nbsp;
            Stop <b style='color:#dc2626'>−{bt_sl:.0f} pts</b> (stop fill at exact level) &nbsp;·&nbsp;
            Implied RR <b>1 : {bt_rr_display:.2f}</b> &nbsp;·&nbsp;
            Same-candle TP+SL breach → <b style='color:#dc2626'>Loss</b> (conservative) &nbsp;·&nbsp;
            Max hold <b>{bt_max_bars} bars ({bt_max_bars*15//60} h)</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Charts row (Time-series equity curve + outcome donut chart merged perfectly)
        col_eq, col_do = st.columns([0.68, 0.32])
        with col_eq:
            st.plotly_chart(build_equity_curve_timeseries(trades_df), use_container_width=True, config={"displayModeBar": False})
        with col_do:
            st.plotly_chart(build_result_donut(bt_stats), use_container_width=True, config={"displayModeBar": False})

        # Detailed Trade Log (Advanced Colored Trade-Styler table)
        st.markdown("<div style='font-size:0.72rem; color:#0284c7; letter-spacing:0.1em; text-transform:uppercase; margin:10px 0 4px'>Detailed Trade Log</div>", unsafe_allow_html=True)
        display_trades = trades_df.copy()
        display_trades["entry_time"] = display_trades["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_trades["exit_time"]  = display_trades["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_trades.index = range(1, len(display_trades) + 1)
        display_trades.index.name = "#"
        display_trades.columns = ["Entry Time", "Exit Time", "Type", "Entry Px", "Exit Px", "P&L (pts)", "Result", "Cum P&L"]

        def _style_trades(row):
            if row["Result"] == "Win":
                return ["background-color:#dcfce7; color:#15803d"] * len(row)
            if row["Result"] == "Loss":
                return ["background-color:#fee2e2; color:#b91c1c"] * len(row)
            return ["background-color:#f1f5f9; color:#475569"] * len(row)

        st.dataframe(
            display_trades.style.apply(_style_trades, axis=1).format({
                "Entry Px": "{:.2f}", "Exit Px": "{:.2f}", "P&L (pts)": "{:+.2f}", "Cum P&L": "{:+.2f}"
            }),
            use_container_width=True, height=400,
        )

# ══════════════════════════════════════════════
# TAB 3 — SIGNAL LOG
# ══════════════════════════════════════════════
with tab_signals:
    sig_df = build_signal_table(processed_df)

    if sig_df.empty:
        st.info("No SMC sweep signals detected in this lookback window.")
    else:
        total_buy  = int((sig_df["Signal"] == "BUY").sum())
        total_sell = int((sig_df["Signal"] == "SELL").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Signals",  len(sig_df))
        c2.metric("Bullish Sweeps", total_buy,  delta=f"+{total_buy}")
        c3.metric("Bearish Sweeps", total_sell, delta=f"−{total_sell}", delta_color="inverse")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        def colour_signal(val):
            if val == "BUY":  return "background-color:#dcfce7; color:#16a34a; font-weight:700"
            if val == "SELL": return "background-color:#fee2e2; color:#dc2626; font-weight:700"
            return ""

        st.dataframe(sig_df.style.map(colour_signal, subset=["Signal"]), use_container_width=True, height=460)

# ══════════════════════════════════════════════
# TAB 4 — RAW DATA
# ══════════════════════════════════════════════
with tab_data:
    display_cols = ["Open", "High", "Low", "Close", "Volume", "support", "resistance", "vol_sma", "signal", "sl", "tp", "rel_volume"]
    display_df = processed_df[display_cols].copy()

    display_df["bt_result"] = ""
    if not trades_df.empty:
        bt_result_map: dict = dict(zip(pd.to_datetime(trades_df["entry_time"]).dt.tz_localize(None), trades_df["result"]))
        display_df["bt_result"] = display_df.index.map(lambda ts: bt_result_map.get(ts, ""))

    display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
    display_df = display_df.sort_index(ascending=False)

    fmt = {
        "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Close": "{:.2f}",
        "support": "{:.2f}", "resistance": "{:.2f}", "vol_sma": "{:,.0f}", "Volume": "{:,.0f}",
        "sl": "{:.2f}", "tp": "{:.2f}", "rel_volume": "{:.2f}"
    }

    def _colour_signal_raw(val):
        if val == "BUY":  return "background-color:#dcfce7; color:#16a34a; font-weight:700"
        if val == "SELL": return "background-color:#fee2e2; color:#dc2626; font-weight:700"
        return ""

    def _colour_bt_result(val):
        if val == "Win":  return "color:#16a34a; font-weight:700"
        if val == "Loss": return "color:#dc2626; font-weight:700"
        return "color:#64748b"

    st.dataframe(
        display_df.style.map(_colour_signal_raw, subset=["signal"]).map(_colour_bt_result, subset=["bt_result"]).format(fmt),
        use_container_width=True, height=500,
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='font-size:0.62rem; color:#64748b; text-align:center; padding-bottom:8px'>"
    f"QuantBengal Pro SMC Terminal &nbsp;·&nbsp; "
    f"Strategy: SMC Liquidity Sweep &nbsp;·&nbsp; "
    f"Backtest: fixed {bt_target:.0f} pt target / {bt_sl:.0f} pt SL "
    f"(1 : {bt_rr_display:.2f} RR) &nbsp;·&nbsp; "
    f"All signals are probabilistic outcomes of historical pattern simulation. "
    f"Past statistics do not guarantee future performance. Not financial advice."
    f"</div>",
    unsafe_allow_html=True,
)
