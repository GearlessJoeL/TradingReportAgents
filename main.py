from __future__ import annotations

import base64
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from agents.vision_analyst import VisionAnalyst
from skills.chart_generator import DEFAULT_MARKET_SYMBOLS, generate_market_charts
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


def _symbol_to_cid(symbol: str) -> str:
    """Convert a ticker symbol to a safe Content-ID name (e.g. '^GSPC' → 'chart_gspc')."""
    return "chart_" + re.sub(r"[^a-zA-Z0-9]", "", symbol).lower()


def _build_image_map(charts_base64: dict[str, str]) -> dict[str, bytes]:
    """Build a CID → raw image bytes mapping from base64-encoded chart data."""
    return {
        _symbol_to_cid(sym): base64.b64decode(b64)
        for sym, b64 in charts_base64.items()
    }


def _block_matches_chart(block: str, symbol: str, display_name: str) -> bool:
    """Check whether an analysis text block corresponds to a given chart symbol."""
    lower = block.lower()
    if display_name.lower() in lower:
        return True
    if symbol in block:
        return True
    clean = re.sub(r"[^a-zA-Z0-9]", "", symbol).lower()
    return clean in lower


def _inject_chart_images(
    report: str,
    charts_base64: dict[str, str],
    significant_movers: list[tuple[str, str, float]],
) -> str:
    """Place each chart image directly above its matching analysis block in section 4."""
    symbol_names: dict[str, str] = {v: k for k, v in DEFAULT_MARKET_SYMBOLS.items()}
    for name, sym, _ in significant_movers:
        symbol_names.setdefault(sym, name)

    if not charts_base64:
        return report

    sec4_header = re.search(r"(## 4\.[^\n]*\n)", report)
    if not sec4_header:
        lines = [
            f"![{symbol_names.get(s, s)}({s})](cid:{_symbol_to_cid(s)})"
            for s in charts_base64
        ]
        return report + "\n\n" + "\n\n".join(lines)

    sec4_start = sec4_header.end()
    sec5 = re.search(r"\n## 5\.", report[sec4_start:])
    sec4_end = (sec4_start + sec5.start()) if sec5 else len(report)
    sec4_body = report[sec4_start:sec4_end]

    # Split into top-level bullet blocks (each starts with "* **")
    blocks = re.split(r"(?=^\* \*\*)", sec4_body, flags=re.MULTILINE)

    used: set[str] = set()
    new_parts: list[str] = []

    for block in blocks:
        matched_sym: str | None = None
        for sym in charts_base64:
            if sym in used:
                continue
            if _block_matches_chart(block, sym, symbol_names.get(sym, sym)):
                matched_sym = sym
                break

        if matched_sym:
            used.add(matched_sym)
            name = symbol_names.get(matched_sym, matched_sym)
            img = f"![{name}({matched_sym})](cid:{_symbol_to_cid(matched_sym)})\n\n"
            new_parts.append(img + block)
        else:
            new_parts.append(block)

    for sym in charts_base64:
        if sym not in used:
            name = symbol_names.get(sym, sym)
            new_parts.append(f"\n![{name}({sym})](cid:{_symbol_to_cid(sym)})\n\n")

    return report[:sec4_start] + "".join(new_parts) + report[sec4_end:]


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
Your objective is to synthesize daily financial news, market data, and visual technical analysis \
into a high-quality, actionable, and structured Daily Market Report for institutional investors.

You must be objective, concise, and focus exclusively on high-impact events. \
Do not hallucinate data, prices, or source links. Do not provide financial advice. \
Your goal is strictly information synthesis and risk assessment.

You will receive raw inputs: news & macro context, watchlist price action, \
and vision analyst output (technical chart summaries). Synthesize them into the exact 7-section \
Markdown format specified in the user prompt. Do not deviate from that structure.

Key rules:
- News Takeaways must be a Markdown table with columns: Theme, Key Stories, Bullish/Bearish Impact.
- Deep Dive must include original source links from the raw data where available.
- Chart & Technical Read must use the bulleted sub-format for each ticker/index analyzed.
- Bull vs Bear Debate table is ONLY for equities with >= 5% daily moves. If none, state so.
- Final Stance must be a single bolded phrase with brief rationale.
- Risks section must be 2-3 forward-looking bullet points.
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
Synthesize ALL inputs above and format your response STRICTLY in Markdown using these exact 7 sections. Do not deviate.

# 📈 Daily Market Report — {trade_date}

## 1. 🌐 General Market Summary
[Single concise paragraph: overarching theme, dominant sentiment (risk-on/off), major index performance, primary macro drivers.]

## 2. 📰 News Takeaways

| Theme | Key Stories | Bullish/Bearish Impact |
| :--- | :--- | :--- |
| [Theme] | [Brief summary] | [Bullish / Bearish / Neutral] |

## 3. 🔍 Deep Dive: Major Catalysts
[1-3 most significant news items with deeper fundamental analysis. MUST append original [Source Link] from Input 1 for each story where available.]
* **[Headline/Topic]**: [Fundamental analysis] - [Source Link]

## 4. 📈 Chart & Technical Read
[Technical breakdown for major indices and key watchlist tickers using Input 3. For EACH chart/ticker use this bulleted format:]
* **[$TICKER / Index Name]**
    * **Explanation:** [Brief narrative of chart pattern/trend]
    * **Current Level:** [Value]
    * **Key Support:** [Value/Zone]
    * **Key Resistance:** [Value/Zone]
    * **50-day MA:** [Value or relationship]
    * **RSI:** [Value or status]
    * **Volume:** [Status]
    * **Bias:** [Bullish / Bearish / Neutral]

## 5. ⚔️ Bull vs Bear Debate (Significant Movers)
[ONLY equities with >= 5% daily move from Input 2. If none, write "No significant outliers in the watchlist today."]

| Ticker | Price Change | The Bull Case | The Bear Case |
| :--- | :--- | :--- | :--- |
| [$TICKER] | [+X% / -Y%] | [Arguments] | [Arguments] |

## 6. 🎯 Final Stance
* **Stance: [Hold / Overweight (with caution) / Underweight / Aggressive Buy]**
* **Rationale:** [1-2 sentences justifying stance based on today's data.]

## 7. 🔮 Risks & What to Watch Next Week
* [Risk/Watch item 1]
* [Risk/Watch item 2]
* [Risk/Watch item 3]

---
*Disclaimer: This report is auto-generated by the TradingAgents Pipeline for research purposes only and does not constitute financial advice.*
"""
    final_report = _message_text(
        quick_llm.invoke(
            [
                ("system", synthesis_system),
                ("human", synthesis_user),
            ]
        )
    )
    final_report = _inject_chart_images(final_report, charts_base64, significant_movers)
    print(final_report)

    image_map = _build_image_map(charts_base64)
    notify_report(
        subject=f"Daily Market Report - {trade_date}",
        markdown_text=final_report,
        image_map=image_map,
    )


if __name__ == "__main__":
    main()
