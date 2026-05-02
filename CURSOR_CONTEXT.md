# Target Architecture: Serverless Daily Financial Agent
We are modifying the `TauricResearch/TradingAgents` repository. 
**Goal:** Transform this heavy, stateful, multi-agent trading framework into a lightweight, stateless, serverless (GitHub Actions) Daily Financial News & Charting Agent.

**Core Rules:**
1. **No Auto-Trading:** Strip out any broker execution logic (Alpaca, IBKR).
2. **Stateless:** Remove LangGraph checkpointers and SQLite memory. The agent runs once, sends a report, and dies.
3. **LLM Gateway:** Replace default OpenAI/Ollama with `OpenRouter` API (defaulting to `openai/gpt-4o-mini`).
4. **New Output:** The final output is a Markdown report sent via Telegram Bot API, not just printed to the console.
5. **New Skill (Vision):** We will add a "Vision Analyst" that reads `yfinance` generated charts (Mag7, S&P500, NDX).