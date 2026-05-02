from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


DEFAULT_MARKET_SYMBOLS: Dict[str, str] = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
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


def _rolling_average(close_series, window: int):
    if close_series.shape[0] < window:
        return None
    return close_series.rolling(window=window).mean()


def _periods_to_try(primary: str) -> List[str]:
    """Prefer the configured period, then shorter windows Yahoo often still has."""
    fallbacks = ["3mo", "1mo", "5d", "1y", "max"]
    ordered: List[str] = []
    for p in [primary.strip()] + fallbacks:
        if p and p not in ordered:
            ordered.append(p)
    return ordered


def _download_daily(symbol: str, period: str):
    return yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )


def _build_chart_base64(title: str, symbol: str, period: str = "6mo") -> str | None:
    for try_period in _periods_to_try(period):
        data = _download_daily(symbol, try_period)
        if data.empty:
            continue
        close = data["Close"]
        if getattr(close, "ndim", 1) > 1:
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            continue
        if try_period != period:
            logger.info(
                "Chart for %s: no data for period=%r, using period=%r instead",
                symbol,
                period,
                try_period,
            )
        break
    else:
        logger.warning(
            "Skipping chart for %s (%s): no Yahoo Finance daily data for any tried period",
            symbol,
            title,
        )
        return None

    ma20 = _rolling_average(close, 20)
    ma50 = _rolling_average(close, 50)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(close.index, close.values, label="Close", linewidth=1.8)
    if ma20 is not None:
        ax.plot(ma20.index, ma20.values, label="MA 20", linewidth=1.2, linestyle="--")
    if ma50 is not None:
        ax.plot(ma50.index, ma50.values, label="MA 50", linewidth=1.2, linestyle="-.")

    ax.set_title(f"{title} ({symbol})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.2)
    ax.legend()
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
) -> Dict[str, str]:
    """Generate market charts and return base64-encoded PNGs.

    Returns:
        Dict[str, str]: Keyed by symbol, values are raw base64 PNG strings.
    """
    watchlist_entries = _parse_watchlist(watchlist_path)
    charts: Dict[str, str] = {}

    for name, symbol in _combined_symbols(watchlist_entries):
        chart_b64 = _build_chart_base64(name, symbol, period=period)
        if chart_b64 is None:
            continue
        charts[symbol] = chart_b64

    if not charts:
        raise ValueError(
            "No charts could be generated: Yahoo Finance returned no usable daily data "
            "for any symbol (defaults + watchlist). Check symbols and CHART_PERIOD."
        )

    return charts
