import pytest

from tradingagents.linear.runtime_interfaces import (
    DebateContext,
    DebateTranscript,
    FetchedNews,
    SummaryOutput,
    run_bull_bear_debate,
)


def test_fetched_news_merges_to_markdown_sections():
    payload = FetchedNews(company_news="Company headline", global_news="Macro headline")

    merged = payload.merged_context()

    assert "## Company News" in merged
    assert "Company headline" in merged
    assert "## Global News" in merged
    assert "Macro headline" in merged


def test_run_bull_bear_debate_produces_ordered_transcript():
    context = DebateContext(
        ticker="AAPL",
        trade_date="2026-05-01",
        market_report="market",
        sentiment_report="sentiment",
        news_report="news",
        fundamentals_report="fundamentals",
    )

    transcript = run_bull_bear_debate(
        context=context,
        rounds=2,
        bull_turn=lambda _ctx, t: f"bull turn {t.count + 1}",
        bear_turn=lambda _ctx, t: f"bear turn {t.count + 1}",
    )

    assert transcript.count == 4
    assert transcript.turns[0].role == "bull"
    assert transcript.turns[1].role == "bear"
    assert transcript.turns[2].role == "bull"
    assert transcript.turns[3].role == "bear"
    assert transcript.current_response.startswith("Bear Analyst:")
    assert "Bull Analyst: bull turn 1" in transcript.bull_history
    assert "Bear Analyst: bear turn 2" in transcript.bear_history


def test_run_bull_bear_debate_appends_existing_transcript():
    context = DebateContext(
        ticker="AAPL",
        trade_date="2026-05-01",
        market_report="market",
        sentiment_report="sentiment",
        news_report="news",
        fundamentals_report="fundamentals",
    )
    existing = DebateTranscript()
    existing.append_turn(
        run_bull_bear_debate(
            context=context,
            rounds=1,
            bull_turn=lambda _ctx, _t: "seed bull",
            bear_turn=lambda _ctx, _t: "seed bear",
        ).turns[0]
    )

    updated = run_bull_bear_debate(
        context=context,
        rounds=1,
        bull_turn=lambda _ctx, _t: "next bull",
        bear_turn=lambda _ctx, _t: "next bear",
        transcript=existing,
    )

    assert updated.count == 3
    assert updated.history.count("Analyst:") == 3


def test_run_bull_bear_debate_requires_positive_rounds():
    context = DebateContext(
        ticker="AAPL",
        trade_date="2026-05-01",
        market_report="market",
        sentiment_report="sentiment",
        news_report="news",
        fundamentals_report="fundamentals",
    )

    with pytest.raises(ValueError, match="rounds must be >= 1"):
        run_bull_bear_debate(
            context=context,
            rounds=0,
            bull_turn=lambda _ctx, _t: "bull",
            bear_turn=lambda _ctx, _t: "bear",
        )


def test_summary_output_contract_keeps_required_fields():
    summary = SummaryOutput(
        recommendation="Buy",
        rationale="Bull arguments outweigh near-term macro risks.",
        strategic_actions="Scale in over 3 tranches.",
        markdown_report="# Debate Summary\n\n- Recommendation: Buy",
    )

    assert summary.recommendation == "Buy"
    assert "Recommendation" in summary.markdown_report
