"""Tests for vision analyst message parsing (provider content shapes)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.vision_analyst import VisionAnalyst, _message_content_text


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("  hello  ", "hello"),
        ("", ""),
        ([{"type": "text", "text": "a"}], "a"),
        ([{"type": "output_text", "output_text": "b"}], "b"),
        ([{"text": "c"}], "c"),
        (["plain"], "plain"),
        (
            [{"type": "reasoning"}, {"type": "text", "text": "final"}],
            "final",
        ),
    ],
)
def test_message_content_text(content, expected: str) -> None:
    msg = SimpleNamespace(content=content)
    assert _message_content_text(msg) == expected


def test_analyze_charts_requires_non_empty_dict() -> None:
    analyst = VisionAnalyst(chart_llm=object(), text_llm=object())
    with pytest.raises(ValueError, match="cannot be empty"):
        analyst.analyze_charts({})
