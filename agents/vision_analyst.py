from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import httpx

from tradingagents.default_config import DEFAULT_CONFIG, apply_llm_env_overrides

logger = logging.getLogger(__name__)


class VisionAnalyst:
    """Analyze chart images with OpenRouter vision-capable models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        text_model: Optional[str] = None,
        chart_model: Optional[str] = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for VisionAnalyst.")

        merged = DEFAULT_CONFIG.copy()
        apply_llm_env_overrides(merged)
        resolved_base = (
            base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or merged.get("backend_url")
            or "https://openrouter.ai/api/v1"
        )
        self.base_url = resolved_base.rstrip("/")
        self.text_model = (
            text_model
            or os.getenv("OPENROUTER_VISION_TEXT_MODEL")
            or merged["deep_think_llm"]
        )
        self.chart_model = (
            chart_model
            or os.getenv("OPENROUTER_CHART_MODEL")
            or merged.get("chart_llm")
            or "moonshotai/kimi-k2.6"
        )
        self.timeout_seconds = timeout_seconds

    def _request_chat_completion(
        self,
        *,
        model: str,
        messages: List[dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "https://localhost").strip() or "https://localhost"
        app_title = os.getenv("OPENROUTER_APP_TITLE", "TradingReportAgents").strip() or "TradingReportAgents"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": referer,
                    "X-Title": app_title,
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
        if response.is_error:
            body_preview = (response.text or "")[:2000]
            logger.error(
                "OpenRouter chat/completions failed model=%s status=%s body=%s",
                model,
                response.status_code,
                body_preview,
            )
            raise httpx.HTTPStatusError(
                f"{response.status_code} for {response.url} model={model!r} — {body_preview}",
                request=response.request,
                response=response,
            )
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    def _build_vision_messages(self, prompt: str, charts_base64: Dict[str, str]) -> List[dict]:
        content = [{"type": "text", "text": prompt}]
        for symbol, chart_base64 in charts_base64.items():
            content.append({"type": "text", "text": f"Chart symbol: {symbol}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{chart_base64}"},
                }
            )
        return [{"role": "user", "content": content}]

    @staticmethod
    def _build_text_only_messages(prompt: str) -> List[dict]:
        """Single user message without images (for models that are not vision-capable)."""
        return [{"role": "user", "content": prompt}]

    def generate_visual_summary(self, charts_base64: Dict[str, str]) -> str:
        """First pass: vision model extracts visual chart signals."""
        prompt = (
            "Review each chart and describe key visual patterns only. "
            "For each symbol, identify trend direction, momentum behavior, "
            "volatility regime, and notable inflection points."
        )
        messages = self._build_vision_messages(prompt, charts_base64)
        return self._request_chat_completion(
            model=self.chart_model,
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
        )

    def analyze_charts(
        self,
        charts_base64: Dict[str, str],
        *,
        news_context: str = "",
    ) -> Dict[str, str]:
        """Two-stage analysis: vision chart read, then text model technical write-up."""
        if not charts_base64:
            raise ValueError("charts_base64 cannot be empty.")

        visual_summary = self.generate_visual_summary(charts_base64)
        symbols = ", ".join(charts_base64)
        news_block = ""
        if news_context.strip():
            news_block = (
                "\n\nRetrieved headlines and summaries (use only to contextualize moves; "
                "do not invent stories beyond this text):\n"
                f"{news_context.strip()}\n"
            )
        prompt = (
            "You are a technical analyst. The text below is a vision-only summary of "
            f"market charts (symbols covered: {symbols}). Expand it into a market "
            "technician report. Do not claim you are looking at raw images; rely on the summary.\n\n"
            "Return sections for each symbol with:\n"
            "1) Trend (short and medium term)\n"
            "2) Support levels\n"
            "3) Resistance levels\n"
            "4) Momentum / weakness signals\n"
            "5) A concise trade bias (bullish, bearish, neutral)\n"
            "6) Where headline text is provided below, add 2-4 bullet 'Pivotal news' items "
            "per symbol that plausibly relate to price action (quote or paraphrase titles only).\n\n"
            "Then provide an overall cross-asset view for equities, oil, and gold.\n\n"
            f"Visual summary from chart model:\n{visual_summary}"
            f"{news_block}"
        )
        messages = self._build_text_only_messages(prompt)
        tech_max_tokens = 2200 if news_block else 1500
        technical_analysis = self._request_chat_completion(
            model=self.text_model,
            messages=messages,
            temperature=0.1,
            max_tokens=tech_max_tokens,
        )
        return {
            "visual_summary": visual_summary,
            "technical_analysis": technical_analysis,
        }
