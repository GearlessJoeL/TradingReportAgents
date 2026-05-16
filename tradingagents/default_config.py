import os
from typing import Any, Dict

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Upstream TRADINGAGENTS_* env-var overlay (applied at import).
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER": "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM": "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM": "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL": "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE": "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS": "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS": "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED": "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER": "benchmark_ticker",
}

# Fork-specific LLM_* keys (applied at runtime via apply_llm_env_overrides).
_LLM_ENV_TO_CONFIG: tuple[tuple[str, str], ...] = (
    ("LLM_PROVIDER", "llm_provider"),
    ("LLM_DEEP_THINK_MODEL", "deep_think_llm"),
    ("LLM_QUICK_THINK_MODEL", "quick_think_llm"),
    ("LLM_CHART_MODEL", "chart_llm"),
    ("LLM_BACKEND_URL", "backend_url"),
)


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        config[key] = _coerce(raw, config.get(key))
    return config


def apply_llm_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay LLM_* environment variables onto a config dict (mutates in place).

    Empty or whitespace-only model/provider values are ignored (keeps prior config).
    ``LLM_BACKEND_URL`` may be set to empty to force ``backend_url`` to ``None``.
    """
    for env_key, cfg_key in _LLM_ENV_TO_CONFIG:
        raw = os.getenv(env_key)
        if raw is None:
            continue
        stripped = raw.strip()
        if cfg_key == "backend_url":
            config[cfg_key] = stripped if stripped else None
        elif stripped:
            config[cfg_key] = stripped
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    # LLM settings — all models routed through OpenRouter by default.
    "llm_provider": "openrouter",
    "deep_think_llm": "deepseek/deepseek-v4-pro",
    "quick_think_llm": "deepseek/deepseek-v4-flash",
    "chart_llm": "moonshotai/kimi-k2.6",
    "backend_url": None,
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,
    "checkpoint_enabled": False,
    "output_language": "English",
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "news_article_limit": 20,
    "global_news_article_limit": 10,
    "global_news_lookback_days": 7,
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    },
    "tool_vendors": {},
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS": "^NSEI",
        ".BO": "^BSESN",
        ".T": "^N225",
        ".HK": "^HSI",
        ".L": "^FTSE",
        ".TO": "^GSPTSE",
        ".AX": "^AXJO",
        "": "SPY",
    },
})
