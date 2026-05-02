"""Linear runtime contracts and helpers."""

from .runtime_interfaces import (
    DebateContext,
    DebateSummarizer,
    DebateTranscript,
    DebateTurn,
    DebaterTurnFn,
    FetchNewsRequest,
    FetchedNews,
    NewsFetcher,
    SummarizerInput,
    SummaryOutput,
    run_bull_bear_debate,
)
from .pipeline import LinearDebateRuntime, LinearRunResult

__all__ = [
    "DebateContext",
    "DebateSummarizer",
    "DebateTranscript",
    "DebateTurn",
    "DebaterTurnFn",
    "FetchNewsRequest",
    "FetchedNews",
    "NewsFetcher",
    "SummarizerInput",
    "SummaryOutput",
    "run_bull_bear_debate",
    "LinearDebateRuntime",
    "LinearRunResult",
]
