# Post-Refactor Verification Plan

This checklist defines the minimum post-refactor validation gates for the
stateless linear runtime (`Fetch News -> Bull/Bear Debate -> Summarize`).

## 1) Smoke Run Checks

Run an isolated smoke test for the linear pipeline:

```bash
pytest -q tests/test_linear_pipeline_smoke.py
```

Pass criteria:
- Runtime finishes one full cycle without LangGraph state/checkpoint machinery.
- Final state includes merged company/global news sections.
- Debate transcript includes both Bull and Bear turns.
- Final markdown includes a recommendation line.

## 2) Prompt Regression Checks

Run prompt snapshot checks to protect canonical Bull/Bear wording:

```bash
pytest -q tests/test_researcher_prompts.py
```

Pass criteria:
- `build_bull_prompt()` output exactly matches the preserved Bull template.
- `build_bear_prompt()` output exactly matches the preserved Bear template.

## 3) Docs + CLI Validation Checks

Validate CLI contract and user-facing help text:

```bash
pytest -q tests/test_cli_linear_contract.py
tradingagents --help
tradingagents analyze --help
```

Pass criteria:
- CLI help describes the stateless linear runtime.
- `analyze` help exposes linear controls (`--rounds`, `--save-report/--no-save-report`).
- Deprecated checkpoint flags (`--checkpoint`, `--clear-checkpoints`) are absent.

## 4) Full Verification Bundle

Run all post-refactor verification checks together:

```bash
pytest -q \
  tests/test_linear_pipeline_smoke.py \
  tests/test_researcher_prompts.py \
  tests/test_cli_linear_contract.py
```
