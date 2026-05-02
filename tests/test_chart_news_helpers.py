"""Unit tests for chart-linked news ticker mapping and fetch formatting."""

from unittest.mock import patch

import main as tr_main


def test_news_query_ticker_proxies_defaults() -> None:
    assert tr_main._news_query_ticker_for_chart("^GSPC") == "SPY"
    assert tr_main._news_query_ticker_for_chart("^NDX") == "QQQ"
    assert tr_main._news_query_ticker_for_chart("CL=F") == "USO"
    assert tr_main._news_query_ticker_for_chart("GC=F") == "GLD"


def test_news_query_ticker_plain_equity_unchanged() -> None:
    assert tr_main._news_query_ticker_for_chart("AAPL") == "AAPL"


@patch.object(tr_main, "route_to_vendor")
def test_fetch_chart_symbol_news_concatenates(mock_route) -> None:
    mock_route.return_value = "## X News\n\n### Headline\nLink: https://example.com\n"
    out = tr_main._fetch_chart_symbol_news(
        ["AAPL", "^GSPC"],
        trade_date="2026-01-15",
        lookback_days=3,
        max_chars_per_symbol=10_000,
    )
    assert "Chart-linked news" in out
    assert "AAPL" in out
    assert "^GSPC" in out
    assert "SPY" in out  # proxy note
    assert mock_route.call_count == 2
    calls = [c[0] for c in mock_route.call_args_list]
    assert calls[0] == ("get_news", "AAPL", "2026-01-12", "2026-01-15")
    assert calls[1] == ("get_news", "SPY", "2026-01-12", "2026-01-15")
