from __future__ import annotations

import os
from typing import Dict, List, Optional

import httpx


class VisionAnalyst:
    """Analyze chart images with OpenRouter vision-capable models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        text_model: str = "deepseek/deepseek-v4-flash",
        chart_model: str = "banana2",
        timeout_seconds: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for VisionAnalyst.")

        self.base_url = base_url.rstrip("/")
        self.text_model = text_model
        self.chart_model = chart_model
        self.timeout_seconds = timeout_seconds

    def _request_chat_completion(
        self,
        *,
        model: str,
        messages: List[dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
        response.raise_for_status()
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

    def generate_visual_summary(self, charts_base64: Dict[str, str]) -> str:
        """Use banana2 to extract visual chart signals."""
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

    def analyze_charts(self, charts_base64: Dict[str, str]) -> Dict[str, str]:
        """Run a two-stage analysis: banana2 visual pass + deepseek technical analysis."""
        if not charts_base64:
            raise ValueError("charts_base64 cannot be empty.")

        visual_summary = self.generate_visual_summary(charts_base64)
        prompt = (
            "You are a technical analyst. Use the provided charts and visual summary "
            "to produce a market technician report.\n\n"
            "Return sections for each symbol with:\n"
            "1) Trend (short and medium term)\n"
            "2) Support levels\n"
            "3) Resistance levels\n"
            "4) Momentum / weakness signals\n"
            "5) A concise trade bias (bullish, bearish, neutral)\n\n"
            "Then provide an overall cross-asset view for equities, oil, and gold.\n\n"
            f"Visual summary from chart model:\n{visual_summary}"
        )
        messages = self._build_vision_messages(prompt, charts_base64)
        technical_analysis = self._request_chat_completion(
            model=self.text_model,
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
        )
        return {
            "visual_summary": visual_summary,
            "technical_analysis": technical_analysis,
        }
