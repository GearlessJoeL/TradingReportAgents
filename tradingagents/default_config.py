import os
from typing import Any, Dict

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Map .env keys (LLM_*) onto DEFAULT_CONFIG keys. Values in .env.example match defaults.
_LLM_ENV_TO_CONFIG: tuple[tuple[str, str], ...] = (
    ("LLM_PROVIDER", "llm_provider"),
    ("LLM_DEEP_THINK_MODEL", "deep_think_llm"),
    ("LLM_QUICK_THINK_MODEL", "quick_think_llm"),
    ("LLM_CHART_MODEL", "chart_llm"),
    ("LLM_BACKEND_URL", "backend_url"),
)


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


DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    # LLM settings — all models routed through OpenRouter by default.
    # Override llm_provider / model names to use a different gateway.
    "llm_provider": "openrouter",
    "deep_think_llm": "deepseek/deepseek-v4-flash",
    "quick_think_llm": "deepseek/deepseek-v4-flash",
    # Vision-in via chat/completions; use a model that reliably accepts image_url payloads.
    "chart_llm": "openai/gpt-4o-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (OpenRouter → openrouter.ai/api/v1, OpenAI → api.openai.com, etc.).
    # The CLI overrides this per provider when the user picks one.
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}
