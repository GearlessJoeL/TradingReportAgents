from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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


def _periods_to_try(primary: str) -> List[str]:
    """Prefer the configured period, then shorter windows Yahoo often still has."""
    fallbacks = ["3mo", "1mo", "5d", "1y", "max"]
    ordered: List[str] = []
    for p in [primary.strip()] + fallbacks:
        if p and p not in ordered:
            ordered.append(p)
    return ordered


def _download_weekly(symbol: str, period: str) -> pd.DataFrame:
    return yf.download(
        symbol,
        period=period,
        interval="1wk",
        auto_adjust=True,
        progress=False,
        threads=False,
    )


def _get_weekly_changes(symbol: str, period: str) -> pd.Series | None:
    """Download weekly data and return percent change per week."""
    for try_period in _periods_to_try(period):
        data = _download_weekly(symbol, try_period)
        if data.empty:
            continue
        close = data["Close"]
        if getattr(close, "ndim", 1) > 1:
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < 2:
            continue
        if try_period != period:
            logger.info(
                "Chart for %s: no data for period=%r, using period=%r instead",
                symbol,
                period,
                try_period,
            )
        pct_change = close.pct_change().dropna() * 100
        return pct_change
    return None


def _has_significant_change(weekly_pct: pd.Series, threshold: float) -> bool:
    """Check if any recent week had a move exceeding the threshold."""
    if weekly_pct.empty:
        return False
    return bool((weekly_pct.abs() >= threshold).any())


def _build_weekly_chart_base64(title: str, symbol: str, period: str = "6mo") -> str | None:
    """Build a weekly gain/drop bar chart and return base64 PNG."""
    weekly_pct = _get_weekly_changes(symbol, period)
    if weekly_pct is None or weekly_pct.empty:
        logger.warning(
            "Skipping chart for %s (%s): no usable weekly data",
            symbol,
            title,
        )
        return None

    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in weekly_pct.values]

    fig, ax = plt.subplots(figsize=(10, 5))
    dates = weekly_pct.index
    ax.bar(dates, weekly_pct.values, color=colors, width=5, edgecolor="none")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"{title} ({symbol}) — Weekly Change %")
    ax.set_xlabel("Week")
    ax.set_ylabel("Change (%)")
    ax.grid(True, alpha=0.2, axis="y")
    fig.autofmt_xdate()

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
) -> Dict[str, str]:
    """Generate weekly gain/drop bar charts and return base64-encoded PNGs.

    Charts are always generated for the 4 default indexes (S&P 500, Nasdaq 100,
    Crude Oil, Gold). Watchlist stocks are only charted if they had at least one
    week with a change exceeding the significance threshold.

    Returns:
        Dict[str, str]: Keyed by symbol, values are raw base64 PNG strings.
    """
    threshold = significance_threshold if significance_threshold is not None else SIGNIFICANT_CHANGE_PCT
    watchlist_entries = _parse_watchlist(watchlist_path)
    default_symbols = set(s.upper() for s in DEFAULT_MARKET_SYMBOLS.values())
    charts: Dict[str, str] = {}

    for name, symbol in _combined_symbols(watchlist_entries):
        is_default = symbol.upper() in default_symbols

        if not is_default:
            weekly_pct = _get_weekly_changes(symbol, period)
            if weekly_pct is None:
                logger.info("Skipping %s (%s): no weekly data available", symbol, name)
                continue
            if not _has_significant_change(weekly_pct, threshold):
                logger.info(
                    "Skipping %s (%s): no week exceeded %.1f%% threshold",
                    symbol,
                    name,
                    threshold,
                )
                continue

        chart_b64 = _build_weekly_chart_base64(name, symbol, period=period)
        if chart_b64 is None:
            continue
        charts[symbol] = chart_b64

    if not charts:
        raise ValueError(
            "No charts could be generated: Yahoo Finance returned no usable weekly data "
            "for any symbol (defaults + watchlist). Check symbols and CHART_PERIOD."
        )

    return charts
