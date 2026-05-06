from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _message_content_text(message: Any) -> str:
    """Extract plain text from an AIMessage-style object (handles string or block lists)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
                elif block.get("type") == "output_text" and block.get("output_text") is not None:
                    parts.append(str(block["output_text"]))
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


class VisionAnalyst:
    """Analyze chart images with the same LangChain chat model stack as the daily pipeline."""

    def __init__(self, *, chart_llm: Any, text_llm: Any) -> None:
        self.chart_llm = chart_llm
        self.text_llm = text_llm

    def _build_image_content(self, prompt: str, charts_base64: Dict[str, str]) -> List[dict]:
        content: List[dict] = [{"type": "text", "text": prompt}]
        for symbol, chart_base64 in charts_base64.items():
            content.append({"type": "text", "text": f"Chart symbol: {symbol}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{chart_base64}"},
                }
            )
        return content

    def generate_visual_summary(self, charts_base64: Dict[str, str]) -> str:
        """First pass: vision model extracts visual chart signals."""
        prompt = (
            "Review each chart and describe key visual patterns only. "
            "For each symbol, identify trend direction, momentum behavior, "
            "volatility regime, and notable inflection points."
        )
        msg = self.chart_llm.invoke(
            [{"role": "user", "content": self._build_image_content(prompt, charts_base64)}]
        )
        text = _message_content_text(msg)
        if not text:
            logger.error(
                "Vision chart_llm returned empty content — is LLM_CHART_MODEL vision-capable?"
            )
            raise RuntimeError(
                "Vision model returned empty content. Set LLM_CHART_MODEL to a model that accepts "
                "image input on your provider (e.g. openai/gpt-4o, google/gemini-2.0-flash-001 "
                "on OpenRouter)."
            )
        return text

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
        msg = self.text_llm.invoke([{"role": "user", "content": prompt}])
        technical_analysis = _message_content_text(msg)
        if not technical_analysis:
            raise RuntimeError(
                "Text model returned empty technical analysis (LLM_DEEP_THINK_MODEL response was empty)."
            )
        return {
            "visual_summary": visual_summary,
            "technical_analysis": technical_analysis,
        }
