"""Compatibility wrapper for the linear runtime.

This file intentionally preserves the historical import path
`tradingagents.graph.trading_graph.TradingAgentsGraph` while routing execution
to the new stateless linear debate pipeline.
"""

from __future__ import annotations

from typing import Any

from tradingagents.default_config import DEFAULT_CONFIG, apply_llm_env_overrides
from tradingagents.linear.pipeline import LinearDebateRuntime


class TradingAgentsGraph:
    """Backward-compatible facade over the linear runtime."""

    def __init__(
        self,
        selected_analysts: list[str] | None = None,
        debug: bool = False,
        config: dict[str, Any] | None = None,
        callbacks: list[Any] | None = None,
    ):
        self.selected_analysts = selected_analysts or ["market", "social", "news", "fundamentals"]
        self.debug = debug
        self.callbacks = callbacks or []
        self.config = DEFAULT_CONFIG.copy() if not config else dict(config)
        apply_llm_env_overrides(self.config)
        self._runtime = LinearDebateRuntime(self.config)
        self.curr_state: dict[str, Any] | None = None

    def propagate(self, company_name: str, trade_date: str):
        result = self._runtime.run(ticker=company_name, trade_date=trade_date)
        self.curr_state = result.final_state
        return result.final_state, result.recommendation

    def process_signal(self, full_signal: str) -> str:
        from tradingagents.agents.utils.rating import parse_rating

        return parse_rating(full_signal)
