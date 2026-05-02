from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timedelta, timezone
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
from tradingagents.default_config import DEFAULT_CONFIG, apply_llm_env_overrides
from tradingagents.llm_clients import create_llm_client


def _message_text(response) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _fetch_global_news(trade_date: str, lookback_days: int) -> str:
    global_news = route_to_vendor("get_global_news", trade_date, lookback_days, 10)
    return global_news.strip()


def _write_chart_files(charts_base64: dict[str, str], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []
    for symbol, image_b64 in charts_base64.items():
        file_path = output_dir / f"{symbol.replace('/', '_')}.png"
        file_path.write_bytes(base64.b64decode(image_b64))
        chart_paths.append(file_path)
    return chart_paths


def _format_movers_section(significant_movers: list[tuple[str, str, float]]) -> str:
    if not significant_movers:
        return "No watchlist stocks moved more than the significance threshold yesterday."
    lines = []
    for name, symbol, pct in significant_movers:
        direction = "up" if pct >= 0 else "down"
        lines.append(f"- **{name}** ({symbol}): {direction} {abs(pct):.1f}%")
    return "Significant movers (previous trading day):\n" + "\n".join(lines)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report_date_raw = os.environ.get("REPORT_DATE", "").strip()
    trade_date = report_date_raw or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lookback_days = int(os.environ.get("NEWS_LOOKBACK_DAYS", "7"))
    watchlist_path = os.environ.get("WATCHLIST_PATH", "watchlist.txt")
    chart_period = os.environ.get("CHART_PERIOD", "6mo")
    charts_output_dir = Path(os.environ.get("CHART_OUTPUT_DIR", "output/charts"))

    config = DEFAULT_CONFIG.copy()
    apply_llm_env_overrides(config)
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

    # Step A: fetch global market news.
    global_news = _fetch_global_news(trade_date=trade_date, lookback_days=lookback_days)

    # Step B: generate charts and run vision analysis.
    charts_base64, significant_movers = generate_market_charts(
        watchlist_path=watchlist_path, period=chart_period
    )
    chart_paths = _write_chart_files(charts_base64, charts_output_dir)
    movers_section = _format_movers_section(significant_movers)

    chart_analysis = VisionAnalyst(
        base_url=config.get("backend_url"),
        text_model=config["deep_think_llm"],
        chart_model=config["chart_llm"],
    ).analyze_charts(charts_base64)

    # Step C: run one bull/bear debate pass on the overall market.
    news_context = f"## Global Market News\n{global_news}\n\n## Significant Stock Movers\n{movers_section}"
    debate_context = DebatePromptContext(
        market_research_report="(not provided in this synchronous pipeline)",
        sentiment_report="(not provided in this synchronous pipeline)",
        news_report=news_context,
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
    synthesis_system = """\
You are an elite, autonomous Financial System Analyst and Portfolio Manager. \
Your objective is to synthesize daily financial news, market data, and technical analysis \
into a high-quality, actionable, and noise-free Daily Market Report for an institutional investor.

You must be objective, concise, and focus exclusively on high-impact events. \
Do not provide financial advice or execute trades. Your goal is strictly information synthesis and risk assessment.

## Core Responsibilities

### Step 1: The Macro Market Summary (The "Top Down" View)
Review the news context and overall index performance. Provide a single, concise paragraph \
summarizing the overarching theme of yesterday's market (e.g. "Risk-on sentiment driven by \
dovish Fed commentary," or "Tech sector drag due to semiconductor supply chain fears").

### Step 2: Significant News Filtering & Micro Analysis
Review the provided news items. Ruthlessly filter out noise — ignore standard PR announcements, \
minor analyst upgrades, and low-impact geopolitical noise. Select only the most significant news \
items (maximum 3-5) that have a direct, material impact on the broader market or specific sectors. \
For each selected item, provide a 2-3 sentence analysis detailing why it matters and its potential \
ripple effects. When a source link is available for a news item, include it as a markdown hyperlink \
on the headline (e.g. **[Headline](url)**). If no link is available, just bold the headline.

### Step 3: Watchlist Volatility Trigger & Deep Dive
Review the watchlist price action data. Identify any equity with a daily price change (positive or \
negative) of >= 5%. For every triggered equity: state the ticker and the exact percentage change; \
cross-reference the news context to explain the catalyst for the move; integrate any provided \
technical analysis to state whether the move broke key support/resistance levels. Use the bull and \
bear perspectives to add depth to your catalyst analysis.
"""

    synthesis_user = f"""\
Analysis date: {trade_date}

## Input 1: Macro News Context
{global_news}

## Input 2: Watchlist Price Action
{movers_section}

## Input 3: Visual Technical Analysis
Chart visual summary:
{chart_analysis["visual_summary"]}

Technical chart analysis:
{chart_analysis["technical_analysis"]}

## Input 4: Bull/Bear Market Perspectives
Bull argument:
{bull_argument}

Bear argument:
{bear_argument}

## Output Format (Strict Adherence Required)
Format your final output strictly in Markdown using this structure:

# 📈 Daily Market Report - {trade_date}

## 🌐 Macro Market Overview
[A concise, 3-4 sentence summary of yesterday's overall market action and primary drivers.]

## 📰 High-Impact Catalysts
* **[Event/Headline 1](source_url_if_available)**: [Your brief, sharp analysis of its impact.]
* **[Event/Headline 2](source_url_if_available)**: [Your brief, sharp analysis of its impact.]
*(Use markdown hyperlinks on headlines when a source Link is provided in the news context. Omit the link markup if none is available.)*

## 🚨 Watchlist Volatility Alerts (>5% Move)
*(If no stocks triggered the 5% threshold, output: "No watchlist equities experienced a >5% move yesterday.")*

* **[$TICKER]**: [+X% / -Y%]
    * *Catalyst:* [Explanation of the move]
    * *Technical Context:* [Integration of chart data, if applicable]

---
*Disclaimer: This report is auto-generated for research purposes only and does not constitute financial advice.*
"""
    final_report = _message_text(
        quick_llm.invoke(
            [
                ("system", synthesis_system),
                ("human", synthesis_user),
            ]
        )
    )
    print(final_report)

    image_buffers = [path.read_bytes() for path in chart_paths]
    notify_report(
        subject=f"Daily Market Report - {trade_date}",
        markdown_text=final_report,
        image_buffers=image_buffers,
    )


if __name__ == "__main__":
    main()
