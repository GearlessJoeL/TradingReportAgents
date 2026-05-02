from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.linear import pipeline as linear_pipeline


class _DummyClient:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


class _DummyDeepLLM:
    def __init__(self):
        self._responses = iter(
            [
                "Bull case: accelerating demand and margin resilience.",
                "Bear case: concentration risk and valuation compression.",
            ]
        )

    def invoke(self, _prompt):
        return SimpleNamespace(content=next(self._responses))


@pytest.mark.smoke
def test_linear_runtime_smoke_run(monkeypatch):
    for key in (
        "LLM_PROVIDER",
        "LLM_DEEP_THINK_MODEL",
        "LLM_QUICK_THINK_MODEL",
        "LLM_CHART_MODEL",
        "LLM_BACKEND_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    deep_llm = _DummyDeepLLM()
    quick_llm = object()

    def _fake_create_llm_client(provider, model, base_url=None):  # noqa: ARG001
        if model == "deep-model":
            return _DummyClient(deep_llm)
        return _DummyClient(quick_llm)

    def _fake_route_to_vendor(tool_name, *args):  # noqa: ARG001
        if tool_name == "get_news":
            return "- Company headline A\n- Company headline B"
        if tool_name == "get_global_news":
            return "- Macro headline A\n- Macro headline B"
        raise AssertionError(f"Unexpected vendor call: {tool_name}")

    def _fake_research_manager(_llm):
        def _runner(_state):
            return {
                "investment_plan": (
                    "**Recommendation**: Hold\n\n"
                    "**Rationale**: Debate is balanced.\n\n"
                    "**Strategic Actions**: Maintain current sizing."
                )
            }

        return _runner

    monkeypatch.setattr(linear_pipeline, "create_llm_client", _fake_create_llm_client)
    monkeypatch.setattr(linear_pipeline, "route_to_vendor", _fake_route_to_vendor)
    monkeypatch.setattr(linear_pipeline, "create_research_manager", _fake_research_manager)

    config = DEFAULT_CONFIG.copy()
    config["deep_think_llm"] = "deep-model"
    config["quick_think_llm"] = "quick-model"
    config["max_debate_rounds"] = 1

    runtime = linear_pipeline.LinearDebateRuntime(config=config)
    result = runtime.run(ticker="NVDA", trade_date="2026-05-01")

    assert result.recommendation == "Hold"
    assert result.final_state["company_of_interest"] == "NVDA"
    assert "## Company News" in result.final_state["news_report"]
    assert "## Global News" in result.final_state["news_report"]
    assert "Bull Analyst:" in result.final_state["investment_debate_state"]["history"]
    assert "Bear Analyst:" in result.final_state["investment_debate_state"]["history"]
    assert "**Recommendation**: Hold" in result.final_state["final_trade_decision"]
