"""Minimal contracts for the stateless linear debate runtime.

These interfaces intentionally avoid LangGraph-specific state and checkpoint
concepts. They define the smallest typed contract needed for:
1) Fetching company and macro news context
2) Running a bull/bear debate loop for N rounds
3) Returning a stable summarizer output payload
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol


DebaterRole = Literal["bull", "bear"]
DebaterTurnFn = Callable[["DebateContext", "DebateTranscript"], str]


@dataclass(frozen=True)
class FetchNewsRequest:
    """Request contract for fetching debate news context."""

    ticker: str
    trade_date: str
    look_back_days: int = 7
    company_news_limit: int = 10
    global_news_limit: int = 5


@dataclass(frozen=True)
class FetchedNews:
    """Normalized output contract for news retrieval."""

    company_news: str
    global_news: str

    def merged_context(self) -> str:
        """Return a single markdown block suitable for debate context."""
        return "\n\n".join(
            [
                "## Company News",
                self.company_news.strip(),
                "## Global News",
                self.global_news.strip(),
            ]
        )


@dataclass(frozen=True)
class DebateContext:
    """Inputs required by bull/bear prompt turns."""

    ticker: str
    trade_date: str
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str


@dataclass(frozen=True)
class DebateTurn:
    """One debater turn generated in the loop."""

    role: DebaterRole
    argument: str
    round_index: int

    def prefixed_argument(self) -> str:
        speaker = "Bull Analyst" if self.role == "bull" else "Bear Analyst"
        return f"{speaker}: {self.argument.strip()}"


@dataclass
class DebateTranscript:
    """Debate history contract decoupled from legacy AgentState."""

    turns: list[DebateTurn] = field(default_factory=list)
    history: str = ""
    bull_history: str = ""
    bear_history: str = ""
    current_response: str = ""

    def append_turn(self, turn: DebateTurn) -> None:
        """Append a turn and keep string histories synchronized."""
        message = turn.prefixed_argument()
        self.turns.append(turn)
        self.history = _append_line(self.history, message)
        if turn.role == "bull":
            self.bull_history = _append_line(self.bull_history, message)
        else:
            self.bear_history = _append_line(self.bear_history, message)
        self.current_response = message

    @property
    def count(self) -> int:
        return len(self.turns)


@dataclass(frozen=True)
class SummarizerInput:
    """Payload passed to the summarizer after the debate loop."""

    context: DebateContext
    transcript: DebateTranscript


@dataclass(frozen=True)
class SummaryOutput:
    """Stable output contract for the minimal linear runtime."""

    recommendation: str
    rationale: str
    strategic_actions: str
    markdown_report: str


class NewsFetcher(Protocol):
    """Fetch company and global news for a single run."""

    def fetch(self, request: FetchNewsRequest) -> FetchedNews:
        ...


class DebateSummarizer(Protocol):
    """Turn a debate transcript into a single report contract."""

    def summarize(self, payload: SummarizerInput) -> SummaryOutput:
        ...


def run_bull_bear_debate(
    context: DebateContext,
    rounds: int,
    bull_turn: DebaterTurnFn,
    bear_turn: DebaterTurnFn,
    transcript: DebateTranscript | None = None,
) -> DebateTranscript:
    """Run a deterministic bull-then-bear loop for the requested rounds."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    active = transcript or DebateTranscript()
    for round_index in range(1, rounds + 1):
        active.append_turn(
            DebateTurn(
                role="bull",
                argument=bull_turn(context, active),
                round_index=round_index,
            )
        )
        active.append_turn(
            DebateTurn(
                role="bear",
                argument=bear_turn(context, active),
                round_index=round_index,
            )
        )
    return active


def _append_line(text: str, new_line: str) -> str:
    if not text.strip():
        return new_line
    return f"{text}\n{new_line}"
