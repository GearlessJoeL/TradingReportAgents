# TradingReportAgents

**TradingReportAgents** is a small, **stateless daily research pipeline** built on top of the [TradingAgents](https://github.com/TauricResearch/TradingAgents) codebase. It pulls market news and chart snapshots, runs a **vision-style chart read**, a **single-round bull vs bear debate**, synthesizes a **client-facing Markdown report**, and optionally delivers it via **Telegram** and/or **SMTP email**. It does **not** place trades or connect to brokers.

> This project is for **research and reporting**, not financial advice. Model output can be wrong or outdated; verify facts and consult a professional before acting.

---

## What it does

1. **News** — Company and global news for a ticker and date (default data path: **yfinance**).
2. **Charts** — Renders watchlist symbols to PNGs (`skills/chart_generator.py`).
3. **Vision-style analysis** — `agents/vision_analyst.py` summarizes charts via an OpenRouter-compatible vision model.
4. **Debate** — One pass of bull and bear prompts (`tradingagents.agents.researchers.prompts`) over the news context.
5. **Report** — Final Markdown (executive summary, news, technical read, debate, stance, risks).
6. **Notify** — `skills/notifier.py` sends the text (and chart images) when `NOTIFY_TELEGRAM` / `NOTIFY_EMAIL` are enabled.

Entry point for the daily job: **`python main.py`**.

---

## Quick start (local)

**Prerequisites:** Python 3.10+ (see `.github/workflows/daily_agent.yml` for CI version).

```bash
cd TradingReportAgents
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
```

Edit `.env`: set **`OPENROUTER_API_KEY`** (default LLM gateway in `tradingagents/default_config.py`). Add a **`watchlist.txt`** (one ticker symbol per line) for chart generation.

```bash
python main.py
```

The report is printed to stdout; Telegram/email run according to `NOTIFY_*` flags in `.env`.

### Environment variables

| Variable | Role |
|----------|------|
| `OPENROUTER_API_KEY` | Default gateway for LLMs and vision (see `.env.example`). |
| `REPORT_TICKER` | Primary ticker (default `NVDA`). |
| `REPORT_DATE` | Analysis date `YYYY-MM-DD` (default: today UTC). |
| `NEWS_LOOKBACK_DAYS` | News window (default `7`). |
| `WATCHLIST_PATH` | File of tickers for charts (default `watchlist.txt`). |
| `CHART_PERIOD` | yfinance period, e.g. `6mo`. |
| `CHART_OUTPUT_DIR` | Where PNGs are written (default `output/charts`). |
| `NOTIFY_TELEGRAM`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram delivery. |
| `NOTIFY_EMAIL`, `EMAILS_FILE`, `SMTP_*` | Email delivery. |

Full template: **`.env.example`**.

---

## GitHub Actions (scheduled report)

Workflow: **`.github/workflows/daily_agent.yml`**

- Runs on a **weekday UTC schedule** and **`workflow_dispatch`**.
- Installs from **`requirements.txt`**, then runs **`python main.py`**.
- Configure **repository secrets** (API keys, Telegram, SMTP) and **variables** (`REPORT_TICKER`, `REPORT_DATE`, paths, notifier flags) to match your `.env` layout.

If you add imports beyond what `requirements.txt` lists, update that file (or change the workflow to `pip install -e .`) so CI stays green.

---

## Repository layout (high level)

| Path | Purpose |
|------|---------|
| `main.py` | Daily pipeline: news → charts → vision → debate → synthesis → `notify_report`. |
| `agents/vision_analyst.py` | Chart image analysis via OpenRouter. |
| `skills/chart_generator.py` | Market chart PNG generation. |
| `skills/notifier.py` | Telegram + SMTP helpers. |
| `tradingagents/linear/` | Stateless linear runtime (news + debate + summary); used by tests and related tooling. |
| `tradingagents/` | Shared config, dataflows, LLM clients, researcher prompts, etc. |
| `cli/` | Optional **`tradingagents`** Typer CLI (full upstream-style analysis flows). |

---

## Tests and verification

Focused checks for the linear runtime and CLI contract:

```bash
pytest -q \
  tests/test_linear_pipeline_smoke.py \
  tests/test_researcher_prompts.py \
  tests/test_cli_linear_contract.py
```

See **`POST_REFACTOR_VERIFICATION.md`** for criteria. Release notes for the underlying framework live in **`CHANGELOG.md`**.

---

## Optional: full `TradingAgents` package

This tree still contains the broader **TradingAgents** graph, CLI, and configuration surface (`pyproject.toml`, `cli/main.py`). For programmatic use of the full graph, see upstream docs and `tradingagents/default_config.py`. This README focuses on the **TradingReportAgents** daily report path.

---

## Upstream and citation

Derived from **TradingAgents** (Tauric Research). If you use their research in academic work, cite their paper as in the upstream [TradingAgents README](https://github.com/TauricResearch/TradingAgents/blob/main/README.md#citation).

---

## Contributing

Issues and PRs welcome for the report pipeline, notifier, CI, and docs. Large changes to shared `tradingagents/` modules should stay compatible with existing tests.
