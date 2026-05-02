from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.dates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

DEFAULT_MARKET_SYMBOLS: Dict[str, str] = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
}

SIGNIFICANT_CHANGE_PCT = float(os.environ.get("SIGNIFICANT_CHANGE_PCT", "5.0"))

CHART_STYLE = {
    "bg": "#1a1a2e",
    "panel": "#16213e",
    "text": "#e0e0e0",
    "grid": "#2a2a4a",
    "price_line": "#4fc3f7",
    "price_fill": "#4fc3f720",
    "gain": "#00e676",
    "loss": "#ff5252",
    "accent": "#ffd740",
}


def _parse_watchlist(watchlist_path: str | Path) -> List[Tuple[str, str]]:
    """Parse watchlist entries as either 'SYMBOL' or 'Label:SYMBOL'."""
    path = Path(watchlist_path)
    if not path.exists():
        return []

    entries: List[Tuple[str, str]] = []
    seen = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            label, symbol = [part.strip() for part in line.split(":", 1)]
            display_name = label or symbol
        else:
            symbol = line
            display_name = symbol

        key = symbol.upper()
        if key in seen:
            continue
        seen.add(key)
        entries.append((display_name, symbol))

    return entries


def _download_daily(symbol: str, period: str = "1mo") -> pd.DataFrame:
    return yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )


def _get_price_data(symbol: str, period: str = "1mo") -> Tuple[pd.Series, pd.Series] | None:
    """Download daily data and return (closing_prices, daily_pct_change)."""
    data = _download_daily(symbol, period)
    if data.empty:
        return None
    close = data["Close"]
    if getattr(close, "ndim", 1) > 1:
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 2:
        return None
    pct_change = close.pct_change().dropna() * 100
    if pct_change.empty:
        return None
    return close, pct_change


def _has_significant_change(daily_pct: pd.Series, threshold: float) -> bool:
    """True if the most recent trading day had an absolute move >= threshold."""
    if daily_pct.empty:
        return False
    return bool(abs(daily_pct.iloc[-1]) >= threshold)


def _build_chart_base64(
    title: str, symbol: str, close: pd.Series, daily_pct: pd.Series
) -> str | None:
    """Build a dual-panel chart: price line (top) + daily % change bars (bottom)."""
    if close.empty or daily_pct.empty:
        return None

    s = CHART_STYLE
    fig, (ax_price, ax_pct) = plt.subplots(
        2, 1, figsize=(10, 6), height_ratios=[2, 1],
        facecolor=s["bg"], sharex=False,
    )
    fig.subplots_adjust(hspace=0.35)

    # --- Top panel: closing price line ---
    ax_price.set_facecolor(s["panel"])
    ax_price.plot(
        close.index, close.values,
        color=s["price_line"], linewidth=1.8, zorder=3,
    )
    ax_price.fill_between(
        close.index, close.values, close.values.min(),
        color=s["price_line"], alpha=0.08,
    )

    first_price = close.iloc[0]
    last_price = close.iloc[-1]
    total_pct = (last_price / first_price - 1) * 100
    change_color = s["gain"] if total_pct >= 0 else s["loss"]
    ax_price.scatter(
        [close.index[-1]], [last_price],
        color=change_color, s=40, zorder=4,
    )
    ax_price.annotate(
        f"  ${last_price:,.2f} ({total_pct:+.1f}%)",
        xy=(close.index[-1], last_price),
        fontsize=9, fontweight="bold", color=change_color,
        va="center",
    )

    ax_price.set_title(
        f"{title} ({symbol})", fontsize=13, fontweight="bold",
        color=s["text"], pad=10,
    )
    ax_price.set_ylabel("Price", fontsize=10, color=s["text"])
    ax_price.tick_params(colors=s["text"], labelsize=8)
    ax_price.grid(True, alpha=0.25, color=s["grid"], linewidth=0.5)
    for spine in ax_price.spines.values():
        spine.set_color(s["grid"])

    ax_price.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%b %d")
    )
    fig.autofmt_xdate(rotation=0)

    # --- Bottom panel: daily % change bars ---
    ax_pct.set_facecolor(s["panel"])
    bar_colors = [s["gain"] if v >= 0 else s["loss"] for v in daily_pct.values]
    ax_pct.bar(
        daily_pct.index, daily_pct.values,
        color=bar_colors, width=0.7, alpha=0.85, edgecolor="none",
    )
    ax_pct.axhline(0, color=s["text"], linewidth=0.5, alpha=0.4)

    last_pct = daily_pct.iloc[-1]
    ax_pct.annotate(
        f"{last_pct:+.1f}%",
        xy=(daily_pct.index[-1], last_pct),
        fontsize=8, fontweight="bold",
        color=s["gain"] if last_pct >= 0 else s["loss"],
        ha="center",
        va="bottom" if last_pct >= 0 else "top",
        xytext=(0, 4 if last_pct >= 0 else -4),
        textcoords="offset points",
    )

    ax_pct.set_ylabel("Daily Chg %", fontsize=10, color=s["text"])
    ax_pct.tick_params(colors=s["text"], labelsize=8)
    ax_pct.grid(True, alpha=0.25, color=s["grid"], linewidth=0.5, axis="y")
    for spine in ax_pct.spines.values():
        spine.set_color(s["grid"])
    ax_pct.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%b %d")
    )

    buffer = BytesIO()
    fig.savefig(
        buffer, format="png", dpi=150, bbox_inches="tight",
        facecolor=s["bg"], edgecolor="none",
    )
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _combined_symbols(watchlist: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    combined = list(DEFAULT_MARKET_SYMBOLS.items()) + list(watchlist)
    unique: List[Tuple[str, str]] = []
    seen = set()
    for name, symbol in combined:
        key = symbol.upper()
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, symbol))
    return unique


def generate_market_charts(
    watchlist_path: str | Path = "watchlist.txt",
    period: str = "1mo",
    significance_threshold: float | None = None,
) -> tuple[Dict[str, str], List[Tuple[str, str, float]]]:
    """Generate dual-panel charts (price line + daily % change) as base64 PNGs.

    Charts are always generated for the 4 default indexes (S&P 500, Nasdaq 100,
    Crude Oil, Gold). Watchlist stocks are only charted if the previous trading
    day's change >= the significance threshold (default 5%).

    Returns:
        Tuple of:
            - Dict[str, str]: Keyed by symbol, values are raw base64 PNG strings.
            - List of (name, symbol, last_day_pct) for stocks that triggered the filter.
    """
    threshold = significance_threshold if significance_threshold is not None else SIGNIFICANT_CHANGE_PCT
    watchlist_entries = _parse_watchlist(watchlist_path)
    default_symbols = set(s.upper() for s in DEFAULT_MARKET_SYMBOLS.values())
    charts: Dict[str, str] = {}
    significant_movers: List[Tuple[str, str, float]] = []

    for name, symbol in _combined_symbols(watchlist_entries):
        is_default = symbol.upper() in default_symbols

        result = _get_price_data(symbol, period)
        if result is None:
            logger.info("Skipping %s (%s): no daily data available", symbol, name)
            continue

        close, daily_pct = result

        if not is_default:
            if not _has_significant_change(daily_pct, threshold):
                logger.info(
                    "Skipping %s (%s): previous day change %.1f%% below %.1f%% threshold",
                    symbol,
                    name,
                    abs(daily_pct.iloc[-1]),
                    threshold,
                )
                continue
            significant_movers.append((name, symbol, float(daily_pct.iloc[-1])))

        chart_b64 = _build_chart_base64(name, symbol, close, daily_pct)
        if chart_b64 is None:
            continue
        charts[symbol] = chart_b64

    if not charts:
        raise ValueError(
            "No charts could be generated: Yahoo Finance returned no usable daily data "
            "for any symbol (defaults + watchlist). Check symbols and network access."
        )

    return charts, significant_movers
