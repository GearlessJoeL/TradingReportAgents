from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
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
DISPLAY_DAYS = 5


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


def _download_daily(symbol: str) -> pd.DataFrame:
    return yf.download(
        symbol,
        period="1mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )


def _get_daily_changes(symbol: str) -> pd.Series | None:
    """Download daily data and return the last 5 trading days' percent change."""
    data = _download_daily(symbol)
    if data.empty:
        return None
    close = data["Close"]
    if getattr(close, "ndim", 1) > 1:
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 2:
        return None
    pct_change = (close.pct_change().dropna() * 100).tail(DISPLAY_DAYS)
    if pct_change.empty:
        return None
    return pct_change


def _has_significant_change(daily_pct: pd.Series, threshold: float) -> bool:
    """True if the most recent trading day had an absolute move >= threshold."""
    if daily_pct.empty:
        return False
    return bool(abs(daily_pct.iloc[-1]) >= threshold)


def _build_daily_chart_base64(title: str, symbol: str, daily_pct: pd.Series) -> str | None:
    """Build a daily gain/drop bar chart for the last 5 trading days."""
    if daily_pct.empty:
        return None

    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in daily_pct.values]
    date_labels = [d.strftime("%m/%d") for d in daily_pct.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(date_labels, daily_pct.values, color=colors, width=0.6, edgecolor="none")

    for bar, val in zip(bars, daily_pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.15 if val >= 0 else -0.35),
            f"{val:+.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"{title} ({symbol}) — Last {len(daily_pct)} Trading Days")
    ax.set_ylabel("Daily Change (%)")
    ax.grid(True, alpha=0.2, axis="y")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
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
    period: str = "6mo",
    significance_threshold: float | None = None,
) -> tuple[Dict[str, str], List[Tuple[str, str, float]]]:
    """Generate daily gain/drop bar charts (last 5 trading days) as base64 PNGs.

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

        daily_pct = _get_daily_changes(symbol)
        if daily_pct is None:
            logger.info("Skipping %s (%s): no daily data available", symbol, name)
            continue

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

        chart_b64 = _build_daily_chart_base64(name, symbol, daily_pct)
        if chart_b64 is None:
            continue
        charts[symbol] = chart_b64

    if not charts:
        raise ValueError(
            "No charts could be generated: Yahoo Finance returned no usable daily data "
            "for any symbol (defaults + watchlist). Check symbols and network access."
        )

    return charts, significant_movers
