from tradingagents.agents.researchers.prompts import (
    DebatePromptContext,
    build_bear_prompt,
    build_bull_prompt,
)


def _sample_context() -> DebatePromptContext:
    return DebatePromptContext(
        market_research_report="market-data",
        sentiment_report="sentiment-data",
        news_report="news-data",
        fundamentals_report="fundamentals-data",
        history="history-data",
        current_response="last-response",
    )


def test_build_bull_prompt_preserves_verbatim_template():
    prompt = build_bull_prompt(_sample_context())

    expected = """You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
Market research report: market-data
Social media sentiment report: sentiment-data
Latest world affairs news: news-data
Company fundamentals report: fundamentals-data
Conversation history of the debate: history-data
Last bear argument: last-response
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.
"""

    assert prompt == expected


def test_build_bear_prompt_preserves_verbatim_template():
    prompt = build_bear_prompt(_sample_context())

    expected = """You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

Market research report: market-data
Social media sentiment report: sentiment-data
Latest world affairs news: news-data
Company fundamentals report: fundamentals-data
Conversation history of the debate: history-data
Last bull argument: last-response
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock.
"""

    assert prompt == expected
