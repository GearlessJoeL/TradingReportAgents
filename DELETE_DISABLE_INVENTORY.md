# Delete/Disable Inventory for Stateless Linear Pipeline

This inventory is scoped to the linear runtime target:
`Fetch News -> Debate (Bull/Bear prompts) -> Summarize`.

It groups removals by subsystem and distinguishes:
- **Delete**: safe to remove module/file directly.
- **Disable then delete**: remove call sites/flags first, then delete module.

## 1) Checkpointing (LangGraph + SQLite)

### Delete
- `tradingagents/graph/checkpointer.py`
- `tests/test_checkpoint_resume.py`

### Disable Then Delete (code references)
- `tradingagents/graph/trading_graph.py`
  - remove `from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id`
  - remove `_checkpointer_ctx` lifecycle and `workflow.compile(checkpointer=...)` branch
  - remove `checkpoint_step(...)` resume logging branch
  - remove `args["config"]["configurable"]["thread_id"]` injection
  - remove `clear_checkpoint(...)` on successful completion
- `cli/main.py`
  - remove `analyze()` options `--checkpoint` and `--clear-checkpoints`
  - remove `clear_all_checkpoints` import/call
  - remove `run_analysis(checkpoint=checkpoint)` coupling
- `tradingagents/default_config.py`
  - remove `checkpoint_enabled` config key and related comments

### Dependency References
- `pyproject.toml`
  - remove `langgraph-checkpoint-sqlite>=2.0.0`

### Docs/Changelog References To Update
- `README.md`
  - remove "Checkpoint resume" section
  - remove CLI examples: `tradingagents analyze --checkpoint` and `--clear-checkpoints`
  - remove python config example setting `config["checkpoint_enabled"] = True`
- `CHANGELOG.md`
  - update entries describing `--checkpoint`, per-ticker SQLite DBs, and `--clear-checkpoints`
- `CURSOR_CONTEXT.md`
  - aligns with removal already (stateless target); keep as guiding context

## 2) Memory/Reflection Persistence (cross-run state)

### Delete
- `tradingagents/agents/utils/memory.py`
- `tradingagents/graph/reflection.py`
- `tests/test_memory_log.py` (after extracting/relocating any still-needed non-memory assertions)

### Disable Then Delete (code references)
- `tradingagents/graph/trading_graph.py`
  - remove `TradingMemoryLog` import/initialization (`self.memory_log`)
  - remove `Reflector` import/initialization (`self.reflector`)
  - remove `_fetch_returns(...)`
  - remove `_resolve_pending_entries(...)`
  - remove pre-run call `self._resolve_pending_entries(company_name)`
  - remove `past_context = self.memory_log.get_past_context(...)`
  - remove `past_context` injection into `create_initial_state(...)`
  - remove post-run `self.memory_log.store_decision(...)`
- `tradingagents/graph/__init__.py`
  - remove `Reflector` export/import
- `tradingagents/agents/utils/agent_states.py`
  - remove `past_context` field from `AgentState`
- `tradingagents/graph/propagation.py`
  - remove `past_context` parameter/assignment from `create_initial_state(...)`
- `tradingagents/agents/managers/portfolio_manager.py`
  - remove prompt section that injects `state.get("past_context", "")`
- `scripts/smoke_structured_output.py`
  - remove `past_context` field from synthetic PM state

### Config References
- `tradingagents/default_config.py`
  - remove `memory_log_path`
  - remove `memory_log_max_entries`

### Docs/Changelog References To Update
- `README.md`
  - remove "Decision log" persistence section and `TRADINGAGENTS_MEMORY_LOG_PATH` mention
  - remove persistence framing tied to PM memory injection
- `CHANGELOG.md`
  - update/remove entries describing persistent decision log, reflection, memory caps, and PM memory usage

## 3) Trading-Stage Chain (Trader + Risk Debate + Portfolio Manager)

### Delete
- `tradingagents/agents/trader/trader.py`
- `tradingagents/agents/risk_mgmt/aggressive_debator.py`
- `tradingagents/agents/risk_mgmt/conservative_debator.py`
- `tradingagents/agents/risk_mgmt/neutral_debator.py`
- `tradingagents/agents/managers/portfolio_manager.py`

### Disable Then Delete (code references)
- `tradingagents/graph/setup.py`
  - remove node creation for Trader / Aggressive / Neutral / Conservative / Portfolio Manager
  - remove all edges from `Research Manager -> Trader -> risk loop -> Portfolio Manager -> END`
  - reconnect graph tail so output ends after debate summarization stage
- `tradingagents/graph/conditional_logic.py`
  - remove risk-loop routing paths that return/use `"Portfolio Manager"`
- `tradingagents/graph/propagation.py`
  - remove initial state construction for `risk_debate_state`
- `tradingagents/graph/trading_graph.py`
  - remove imports for `RiskDebateState`
  - remove saved-log fields tied to trader/risk/portfolio:
    - `trader_investment_decision`
    - `risk_debate_state`
    - `final_trade_decision` (replace with debate summary output field)
  - replace `process_signal(final_trade_decision)` return path with markdown report output
- `tradingagents/agents/__init__.py`
  - remove exports/imports for trader, risk debators, and portfolio manager
- `tradingagents/agents/schemas.py`
  - remove trader/portfolio schema types and renderers:
    - `TraderAction`, `TraderProposal`, `render_trader_proposal`
    - `PortfolioDecision`, `render_pm_decision`
  - keep research summarization schema only (if still used)
- `tradingagents/agents/utils/agent_states.py`
  - remove `RiskDebateState`, `trader_investment_plan`, and `final_trade_decision` fields
- `tradingagents/graph/signal_processing.py`
  - remove if no longer needed after portfolio-rating output is dropped

### Tests/Scripts References To Remove Or Rewrite
- `tests/test_structured_agents.py`
  - remove Trader test classes/helpers; keep/adapt Research Manager coverage
- `tests/test_signal_processing.py`
  - remove if `SignalProcessor` is removed with PM rating path
- `tests/test_memory_log.py`
  - remove PM injection and portfolio-manager behavior tests along with memory deletion
- `scripts/smoke_structured_output.py`
  - rewrite to only exercise retained summarize stage, or remove script

### Docs/Changelog References To Update
- `README.md`
  - remove/replace architecture sections:
    - "Trader Agent"
    - "Risk Management and Portfolio Manager"
    - statements about simulated exchange execution
  - revise project description to debate-and-summary output (no transaction decision chain)
- `CHANGELOG.md`
  - keep historical entries but add forward note in next release section that trader/risk/portfolio runtime path was removed for linear stateless mode

## Recommended Deletion Order (to avoid broken imports)

1. Route runtime and CLI entrypoint to the new linear orchestrator.
2. Remove checkpoint and memory call sites/config flags.
3. Remove trader/risk/portfolio graph edges and state fields.
4. Delete now-unused modules/tests/scripts/dependency/docs references.
