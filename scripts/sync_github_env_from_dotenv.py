#!/usr/bin/env python3
"""Push values from a local .env file into GitHub Actions secrets and variables.

Requires: GitHub CLI (`gh`) installed and authenticated (`gh auth login`).

Mappings match `.github/workflows/daily_agent.yml`:
  - Secrets: API keys, Telegram tokens, SMTP credentials.
  - Variables: report paths, notifier toggles, EMAILS_FILE, SMTP_USE_TLS.
  - Local `emails.txt` / `watchlist.txt` (if present and non-empty) are pushed to secrets
    `EMAILS_TXT` / `WATCHLIST_TXT` by default so CI can recreate them. GitHub never shows
    secret values in the UI—only that the name exists. Use `--no-runtime-files` to skip.

Usage:
  python scripts/sync_github_env_from_dotenv.py              # uses .env in cwd
  python scripts/sync_github_env_from_dotenv.py path/to/.env
  python scripts/sync_github_env_from_dotenv.py --dry-run
  python scripts/sync_github_env_from_dotenv.py -R owner/repo
  python scripts/sync_github_env_from_dotenv.py --runtime-files-only
  python scripts/sync_github_env_from_dotenv.py --no-runtime-files

Empty values are skipped by default (use --include-empty to sync them anyway).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Must stay aligned with `.github/workflows/daily_agent.yml`.
SECRET_KEYS: frozenset[str] = frozenset(
    {
        "OPENROUTER_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
        "ZHIPU_API_KEY",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
        "SMTP_FROM_EMAIL",
    }
)

VARIABLE_KEYS: frozenset[str] = frozenset(
    {
        "REPORT_TICKER",
        "REPORT_DATE",
        "NEWS_LOOKBACK_DAYS",
        "WATCHLIST_PATH",
        "CHART_PERIOD",
        "CHART_OUTPUT_DIR",
        "NOTIFY_TELEGRAM",
        "NOTIFY_EMAIL",
        "EMAILS_FILE",
        "SMTP_USE_TLS",
    }
)


def _strip_quotes(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser (no multiline values)."""
    if not path.is_file():
        raise FileNotFoundError(f"Env file not found: {path}")

    out: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            print(f"Warning: line {line_no} has no '=': skipping", file=sys.stderr)
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not key:
            print(f"Warning: line {line_no} has empty key: skipping", file=sys.stderr)
            continue
        out[key] = _strip_quotes(value)
    return out


def _gh_base_args(repo: str | None) -> list[str]:
    return ["gh", "-R", repo] if repo else ["gh"]


def set_secret(name: str, value: str, *, repo: str | None, dry_run: bool) -> None:
    cmd = [*_gh_base_args(repo), "secret", "set", name]
    if dry_run:
        preview = "(empty)" if value == "" else f"{len(value)} char(s)"
        print(f"DRY-RUN secret: {name}  {preview}")
        return
    subprocess.run(cmd, input=value.encode("utf-8"), check=True)


def push_runtime_file_secrets(
    *,
    emails_path: Path,
    watchlist_path: Path,
    repo: str | None,
    dry_run: bool,
    include_empty: bool,
) -> None:
    """Upload local emails.txt / watchlist.txt to EMAILS_TXT / WATCHLIST_TXT secrets."""
    mapping = {
        "EMAILS_TXT": emails_path,
        "WATCHLIST_TXT": watchlist_path,
    }
    for secret_name, path in mapping.items():
        if not path.is_file():
            print(f"Notice: skip {secret_name}: file not found ({path})", file=sys.stderr)
            continue
        body = path.read_text(encoding="utf-8")
        if not include_empty and not body.strip():
            print(f"Notice: skip {secret_name}: file is empty ({path})", file=sys.stderr)
            continue
        set_secret(secret_name, body, repo=repo, dry_run=dry_run)
        if not dry_run:
            print(f"Updated GitHub secret {secret_name} from {path} ({len(body)} bytes).")


def set_variable(name: str, value: str, *, repo: str | None, dry_run: bool) -> None:
    cmd = [*_gh_base_args(repo), "variable", "set", name, "--body", value]
    if dry_run:
        preview = repr(value) if len(value) < 80 else repr(value[:77] + "...")
        print(f"DRY-RUN variable: {name} = {preview}")
        return
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "env_file",
        nargs="?",
        default=None,
        type=Path,
        help="Path to .env (default: .env in current directory; not used with --runtime-files-only)",
    )
    parser.add_argument("-R", "--repo", help="GitHub repository (owner/name). Passed to gh -R.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not call gh.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Also sync keys whose value is empty in the .env file.",
    )
    parser.add_argument(
        "--secrets-only",
        action="store_true",
        help="Only create/update secrets.",
    )
    parser.add_argument(
        "--variables-only",
        action="store_true",
        help="Only create/update variables.",
    )
    parser.add_argument(
        "--no-runtime-files",
        action="store_true",
        help="Do not push local emails.txt / watchlist.txt to EMAILS_TXT / WATCHLIST_TXT.",
    )
    parser.add_argument(
        "--runtime-files-only",
        action="store_true",
        help="Only push emails.txt and watchlist.txt to EMAILS_TXT / WATCHLIST_TXT (no .env sync).",
    )
    parser.add_argument(
        "--emails-file",
        type=Path,
        default=Path("emails.txt"),
        help="Path to emails file for --runtime-files (default: emails.txt)",
    )
    parser.add_argument(
        "--watchlist-file",
        type=Path,
        default=Path("watchlist.txt"),
        help="Path to watchlist file for --runtime-files (default: watchlist.txt)",
    )
    args = parser.parse_args()

    if args.secrets_only and args.variables_only:
        print("error: --secrets-only and --variables-only are mutually exclusive", file=sys.stderr)
        return 2

    if args.runtime_files_only and (args.secrets_only or args.variables_only):
        print("error: --runtime-files-only cannot be combined with --secrets-only/--variables-only", file=sys.stderr)
        return 2

    if args.runtime_files_only:
        push_runtime_file_secrets(
            emails_path=args.emails_file.resolve(),
            watchlist_path=args.watchlist_file.resolve(),
            repo=args.repo,
            dry_run=args.dry_run,
            include_empty=args.include_empty,
        )
        return 0

    env_path = args.env_file if args.env_file is not None else Path(".env")
    env_path = env_path.resolve()

    try:
        data = parse_dotenv(env_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    unknown = sorted(k for k in data if k not in SECRET_KEYS and k not in VARIABLE_KEYS)
    if unknown:
        for k in unknown:
            print(f"Notice: key not mapped to GitHub Actions (skipped): {k}", file=sys.stderr)

    def should_sync(value: str) -> bool:
        if args.include_empty:
            return True
        return value != ""

    secret_hits = [(k, data[k]) for k in SECRET_KEYS if k in data and should_sync(data[k])]
    variable_hits = [(k, data[k]) for k in VARIABLE_KEYS if k in data and should_sync(data[k])]

    if not args.variables_only:
        for name, value in secret_hits:
            set_secret(name, value, repo=args.repo, dry_run=args.dry_run)

    if not args.secrets_only:
        for name, value in variable_hits:
            set_variable(name, value, repo=args.repo, dry_run=args.dry_run)

    if not args.dry_run and not args.variables_only:
        skipped_secrets = [k for k in SECRET_KEYS if k in data and not should_sync(data[k])]
        if skipped_secrets:
            print(
                "Skipped empty secret(s) (use --include-empty to push them): "
                + ", ".join(skipped_secrets),
                file=sys.stderr,
            )
    if not args.dry_run and not args.secrets_only:
        skipped_vars = [k for k in VARIABLE_KEYS if k in data and not should_sync(data[k])]
        if skipped_vars:
            print(
                "Skipped empty variable(s) (use --include-empty to push them): "
                + ", ".join(skipped_vars),
                file=sys.stderr,
            )

    if not args.no_runtime_files and not args.variables_only:
        push_runtime_file_secrets(
            emails_path=args.emails_file.resolve(),
            watchlist_path=args.watchlist_file.resolve(),
            repo=args.repo,
            dry_run=args.dry_run,
            include_empty=args.include_empty,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
