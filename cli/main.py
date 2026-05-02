from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from tradingagents.default_config import DEFAULT_CONFIG, apply_llm_env_overrides
from tradingagents.graph.trading_graph import TradingAgentsGraph

load_dotenv()
load_dotenv(".env.enterprise", override=False)

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Stateless linear bull/bear debate runtime",
    add_completion=True,
)
console = Console()


@app.command()
def analyze(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol, e.g. NVDA"),
    analysis_date: str = typer.Option(
        datetime.now().strftime("%Y-%m-%d"),
        "--date",
        "-d",
        help="Analysis date (YYYY-MM-DD).",
    ),
    rounds: int = typer.Option(1, "--rounds", "-r", min=1, help="Bull/Bear rounds."),
    save_report: bool = typer.Option(
        True,
        "--save-report/--no-save-report",
        help="Write markdown report to the configured results directory.",
    ),
):
    config = DEFAULT_CONFIG.copy()
    apply_llm_env_overrides(config)
    config["max_debate_rounds"] = rounds

    graph = TradingAgentsGraph(config=config, debug=False)
    final_state, recommendation = graph.propagate(ticker, analysis_date)
    markdown_report = final_state["final_trade_decision"]

    console.print(f"[bold green]Recommendation:[/bold green] {recommendation}")
    console.print()
    console.print(Markdown(markdown_report))

    if save_report:
        report_dir = Path(config["results_dir"]) / ticker / analysis_date / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / "linear_report.md"
        report_file.write_text(markdown_report, encoding="utf-8")
        console.print(f"\n[cyan]Saved report:[/cyan] {report_file}")


if __name__ == "__main__":
    app()
