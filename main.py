from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from agents.vision_analyst import VisionAnalyst
from skills.chart_generator import generate_market_charts
from skills.notifier import notify_report
from tradingagents.agents.researchers.prompts import (
    DebatePromptContext,
    build_bear_prompt,
    build_bull_prompt,
)
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_llm_client


def _message_text(response) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _fetch_news(ticker: str, trade_date: str, lookback_days: int) -> tuple[str, str, str]:
    trade_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_date = (trade_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    company_news = route_to_vendor("get_news", ticker, start_date, trade_date)
    global_news = route_to_vendor("get_global_news", trade_date, lookback_days, 5)
    merged = "\n\n".join(
        [
            "## Company News",
            company_news.strip(),
            "## Global News",
            global_news.strip(),
        ]
    )
    return company_news, global_news, merged


def _write_chart_files(charts_base64: dict[str, str], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []
    for symbol, image_b64 in charts_base64.items():
        file_path = output_dir / f"{symbol.replace('/', '_')}.png"
        file_path.write_bytes(base64.b64decode(image_b64))
        chart_paths.append(file_path)
    return chart_paths


def main() -> None:
    load_dotenv()

    ticker = os.environ.get("REPORT_TICKER", "NVDA").strip().upper()
    trade_date = os.environ.get("REPORT_DATE", datetime.utcnow().strftime("%Y-%m-%d")).strip()
    lookback_days = int(os.environ.get("NEWS_LOOKBACK_DAYS", "7"))
    watchlist_path = os.environ.get("WATCHLIST_PATH", "watchlist.txt")
    chart_period = os.environ.get("CHART_PERIOD", "6mo")
    charts_output_dir = Path(os.environ.get("CHART_OUTPUT_DIR", "output/charts"))

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }
    set_config(config)

    provider = config["llm_provider"]
    base_url = config.get("backend_url")
    deep_llm = create_llm_client(
        provider=provider,
        model=config["deep_think_llm"],
        base_url=base_url,
    ).get_llm()
    quick_llm = create_llm_client(
        provider=provider,
        model=config["quick_think_llm"],
        base_url=base_url,
    ).get_llm()

    # Step A: fetch company and global news.
    company_news, global_news, merged_news_report = _fetch_news(
        ticker=ticker,
        trade_date=trade_date,
        lookback_days=lookback_days,
    )

    # Step B: generate charts and run vision analysis.
    charts_base64 = generate_market_charts(watchlist_path=watchlist_path, period=chart_period)
    chart_paths = _write_chart_files(charts_base64, charts_output_dir)
    chart_analysis = VisionAnalyst().analyze_charts(charts_base64)

    # Step C: run one bull/bear debate pass using news context.
    debate_context = DebatePromptContext(
        market_research_report="(not provided in this synchronous pipeline)",
        sentiment_report="(not provided in this synchronous pipeline)",
        news_report=merged_news_report,
        fundamentals_report="(not provided in this synchronous pipeline)",
        history="",
        current_response="",
    )
    bull_argument = _message_text(deep_llm.invoke(build_bull_prompt(debate_context)))
    debate_context = DebatePromptContext(
        market_research_report=debate_context.market_research_report,
        sentiment_report=debate_context.sentiment_report,
        news_report=debate_context.news_report,
        fundamentals_report=debate_context.fundamentals_report,
        history=f"Bull Analyst: {bull_argument}",
        current_response=bull_argument,
    )
    bear_argument = _message_text(deep_llm.invoke(build_bear_prompt(debate_context)))

    # Step D: synthesize into final markdown report.
    synthesis_prompt = f"""You are preparing a final client-facing trading markdown report.

Ticker: {ticker}
Analysis date: {trade_date}

Company news:
{company_news}

Global news:
{global_news}

Chart visual summary:
{chart_analysis["visual_summary"]}

Technical chart analysis:
{chart_analysis["technical_analysis"]}

Bull argument:
{bull_argument}

Bear argument:
{bear_argument}

Write a concise markdown report with sections:
1) Executive Summary
2) News Takeaways
3) Chart + Technical Read
4) Bull vs Bear Debate
5) Final Stance (Buy/Overweight/Hold/Underweight/Sell with rationale)
6) Risks and What to Watch Next Week
"""
    final_report = _message_text(quick_llm.invoke(synthesis_prompt))
    print(final_report)

    image_buffers = [path.read_bytes() for path in chart_paths]
    notify_report(
        subject=f"{ticker} Trading Report - {trade_date}",
        markdown_text=final_report,
        image_buffers=image_buffers,
    )


if __name__ == "__main__":
    main()
