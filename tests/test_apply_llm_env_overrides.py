from __future__ import annotations

import pytest

from tradingagents.default_config import DEFAULT_CONFIG, apply_llm_env_overrides


def test_apply_llm_env_overrides_empty_env_unchanged(monkeypatch):
    for key in (
        "LLM_PROVIDER",
        "LLM_DEEP_THINK_MODEL",
        "LLM_QUICK_THINK_MODEL",
        "LLM_CHART_MODEL",
        "LLM_BACKEND_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = DEFAULT_CONFIG.copy()
    apply_llm_env_overrides(cfg)
    assert cfg["deep_think_llm"] == DEFAULT_CONFIG["deep_think_llm"]
    assert cfg["chart_llm"] == DEFAULT_CONFIG["chart_llm"]
    assert cfg["backend_url"] is None


def test_apply_llm_env_overrides_sets_models(monkeypatch):
    cfg = DEFAULT_CONFIG.copy()
    monkeypatch.setenv("LLM_DEEP_THINK_MODEL", "provider/custom-deep")
    monkeypatch.setenv("LLM_CHART_MODEL", "provider/custom-vision")
    monkeypatch.setenv("LLM_BACKEND_URL", "https://example.com/v1")
    apply_llm_env_overrides(cfg)
    assert cfg["deep_think_llm"] == "provider/custom-deep"
    assert cfg["chart_llm"] == "provider/custom-vision"
    assert cfg["backend_url"] == "https://example.com/v1"


def test_apply_llm_backend_url_blank_clears(monkeypatch):
    cfg = DEFAULT_CONFIG.copy()
    cfg["backend_url"] = "https://keep.example/v1"
    monkeypatch.setenv("LLM_BACKEND_URL", "   ")
    apply_llm_env_overrides(cfg)
    assert cfg["backend_url"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("LLM_DEEP_THINK_MODEL", "   "),
        ("LLM_QUICK_THINK_MODEL", ""),
    ],
)
def test_apply_llm_model_blank_ignored(monkeypatch, key, value):
    cfg = DEFAULT_CONFIG.copy()
    monkeypatch.setenv(key, value)
    apply_llm_env_overrides(cfg)
    assert cfg["deep_think_llm"] == DEFAULT_CONFIG["deep_think_llm"]
    assert cfg["quick_think_llm"] == DEFAULT_CONFIG["quick_think_llm"]
