"""Stateless linear debate runtime.

This module is the migration target that replaces the old LangGraph-driven
runtime path while keeping the external `TradingAgentsGraph.propagate()` API
usable through a compatibility wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.prompts import (
    DebatePromptContext,
    build_bear_prompt,
    build_bull_prompt,
)
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.default_config import DEFAULT_CONFIG, apply_llm_env_overrides
from tradingagents.llm_clients import create_llm_client

from .runtime_interfaces import DebateContext, DebateTranscript, run_bull_bear_debate

try:
    from skills.notifier import notify_report
except ModuleNotFoundError:
    def notify_report(subject: str, markdown_text: str, image_map: dict | None = None) -> None:  # noqa: ARG001
        return None


@dataclass(frozen=True)
class LinearRunResult:
    """Result payload returned by the linear runtime."""

    final_state: dict[str, Any]
    recommendation: str


class LinearDebateRuntime:
    """Fetch news, run bull/bear rounds, then summarize in one pass."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = DEFAULT_CONFIG.copy() if not config else dict(config)
        apply_llm_env_overrides(self.config)
        set_config(self.config)

        provider = self.config["llm_provider"]
        base_url = self.config.get("backend_url")
        deep_model = self.config["deep_think_llm"]
        quick_model = self.config["quick_think_llm"]
        chart_model = self.config.get("chart_llm")

        self.deep_llm = create_llm_client(
            provider=provider,
            model=deep_model,
            base_url=base_url,
        ).get_llm()
        self.quick_llm = create_llm_client(
            provider=provider,
            model=quick_model,
            base_url=base_url,
        ).get_llm()
        self.chart_llm = (
            create_llm_client(provider=provider, model=chart_model, base_url=base_url).get_llm()
            if chart_model
            else None
        )

    def run(self, ticker: str, trade_date: str) -> LinearRunResult:
        company_news, global_news = self._fetch_news(ticker=ticker, trade_date=trade_date)
        news_report = "\n\n".join(
            [
                "## Company News",
                company_news.strip(),
                "## Global News",
                global_news.strip(),
            ]
        )

        context = DebateContext(
            ticker=ticker,
            trade_date=trade_date,
            market_report="(not included in linear runtime)",
            sentiment_report="(not included in linear runtime)",
            news_report=news_report,
            fundamentals_report="(not included in linear runtime)",
        )

        transcript = run_bull_bear_debate(
            context=context,
            rounds=max(1, int(self.config.get("max_debate_rounds", 1))),
            bull_turn=self._bull_turn,
            bear_turn=self._bear_turn,
            transcript=DebateTranscript(),
        )

        investment_plan = self._summarize_debate(ticker=ticker, transcript=transcript)
        notify_report(
            subject=f"{ticker} debate summary - {trade_date}",
            markdown_text=investment_plan,
        )

        final_state = {
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "market_report": context.market_report,
            "sentiment_report": context.sentiment_report,
            "news_report": context.news_report,
            "fundamentals_report": context.fundamentals_report,
            "investment_debate_state": {
                "bull_history": transcript.bull_history,
                "bear_history": transcript.bear_history,
                "history": transcript.history,
                "current_response": transcript.current_response,
                "judge_decision": investment_plan,
            },
            "investment_plan": investment_plan,
            "final_trade_decision": investment_plan,
        }

        recommendation = _extract_rating(investment_plan)
        return LinearRunResult(final_state=final_state, recommendation=recommendation)

    def _fetch_news(self, ticker: str, trade_date: str) -> tuple[str, str]:
        trade_dt = datetime.strptime(trade_date, "%Y-%m-%d")
        look_back_days = int(self.config.get("news_lookback_days", 7))
        start_date = (trade_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

        company_news = route_to_vendor("get_news", ticker, start_date, trade_date)
        global_news = route_to_vendor("get_global_news", trade_date, look_back_days, 5)
        return company_news, global_news

    def _bull_turn(self, context: DebateContext, transcript: DebateTranscript) -> str:
        prompt_context = DebatePromptContext(
            market_research_report=context.market_report,
            sentiment_report=context.sentiment_report,
            news_report=context.news_report,
            fundamentals_report=context.fundamentals_report,
            history=transcript.history,
            current_response=transcript.current_response,
        )
        prompt = build_bull_prompt(prompt_context)
        return _message_text(self.deep_llm.invoke(prompt))

    def _bear_turn(self, context: DebateContext, transcript: DebateTranscript) -> str:
        prompt_context = DebatePromptContext(
            market_research_report=context.market_report,
            sentiment_report=context.sentiment_report,
            news_report=context.news_report,
            fundamentals_report=context.fundamentals_report,
            history=transcript.history,
            current_response=transcript.current_response,
        )
        prompt = build_bear_prompt(prompt_context)
        return _message_text(self.deep_llm.invoke(prompt))

    def _summarize_debate(self, ticker: str, transcript: DebateTranscript) -> str:
        state = {
            "company_of_interest": ticker,
            "investment_debate_state": {
                "history": transcript.history,
                "bull_history": transcript.bull_history,
                "bear_history": transcript.bear_history,
                "current_response": transcript.current_response,
                "count": transcript.count,
            },
        }
        return create_research_manager(self.deep_llm)(state)["investment_plan"]


def _message_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _extract_rating(markdown_text: str) -> str:
    from tradingagents.agents.utils.rating import parse_rating

    try:
        return parse_rating(markdown_text)
    except Exception:
        return "Hold"
