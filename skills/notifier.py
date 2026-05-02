from __future__ import annotations

import io
import logging
import os
import re
import smtplib
import ssl
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import BinaryIO

import httpx
import markdown as md

logger = logging.getLogger(__name__)

_TELEGRAM_MAX_LENGTH = 4096


def _markdown_to_email_html(text: str) -> str:
    """Convert markdown to a full HTML email document with inline styles."""
    body_html = md.markdown(text, extensions=["tables", "fenced_code", "nl2br"])
    return f"""\
<html>
<head>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         line-height: 1.6; color: #1a1a1a; max-width: 700px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #0d47a1; border-bottom: 2px solid #0d47a1; padding-bottom: 6px; }}
  h2 {{ color: #1565c0; margin-top: 24px; }}
  h3 {{ color: #1976d2; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #e3f2fd; }}
  code {{ background: #f5f5f5; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }}
  pre {{ background: #263238; color: #eeffff; padding: 14px; border-radius: 6px;
         overflow-x: auto; }}
  pre code {{ background: none; color: inherit; padding: 0; }}
  blockquote {{ border-left: 4px solid #90caf9; margin: 12px 0; padding: 8px 16px;
               background: #e3f2fd; }}
  strong {{ color: #0d47a1; }}
  img {{ max-width: 100%; height: auto; border-radius: 6px; margin: 12px 0 4px 0;
         display: block; }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


def _markdown_to_telegram_html(text: str) -> str:
    """Convert markdown to the HTML subset Telegram supports.

    Telegram supports: <b>, <i>, <u>, <s>, <code>, <pre>, <a href="">.
    Headers are not supported natively, so we render them as bold lines.
    """
    lines = text.split("\n")
    result: list[str] = []
    in_code_block = False

    for line in lines:
        if line.startswith("```"):
            if in_code_block:
                result.append("</pre>")
                in_code_block = False
            else:
                lang = line[3:].strip()
                if lang:
                    result.append(f'<pre language="{_escape_html(lang)}">')
                else:
                    result.append("<pre>")
                in_code_block = True
            continue

        if in_code_block:
            result.append(_escape_html(line))
            continue

        converted = _convert_inline_markdown(line)
        result.append(converted)

    if in_code_block:
        result.append("</pre>")

    return "\n".join(result)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _convert_inline_markdown(line: str) -> str:
    """Convert a single markdown line to Telegram HTML."""
    header_match = re.match(r"^(#{1,6})\s+(.*)", line)
    if header_match:
        content = _escape_html(header_match.group(2))
        content = _apply_inline_formatting(content)
        return f"\n<b>{content}</b>"

    if re.match(r"^[-*]\s+", line):
        content = re.sub(r"^[-*]\s+", "", line)
        content = _escape_html(content)
        content = _apply_inline_formatting(content)
        return f"• {content}"

    if re.match(r"^\d+\.\s+", line):
        content = _escape_html(line)
        content = _apply_inline_formatting(content)
        return content

    if line.startswith("---") or line.startswith("***"):
        return "————————————————"

    escaped = _escape_html(line)
    return _apply_inline_formatting(escaped)


def _apply_inline_formatting(text: str) -> str:
    """Apply bold, italic, and inline code formatting (text must already be HTML-escaped)."""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^\*]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"__(.+?)__", r"<u>\1</u>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    return text


def _split_text(text: str, max_len: int = _TELEGRAM_MAX_LENGTH) -> list[str]:
    """Split text into chunks that fit within Telegram's message limit."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def _env_truthy(name: str, default: str = "false") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in ("1", "true", "yes", "on")


def _to_bytes(image: bytes | bytearray | memoryview | BinaryIO) -> bytes | None:
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)

    if hasattr(image, "seek"):
        image.seek(0)
    if hasattr(image, "read"):
        raw = image.read()
        if isinstance(raw, str):
            return raw.encode("utf-8")
        if isinstance(raw, bytes):
            return raw
    return None


def send_telegram_report(markdown_text: str, image_map: dict[str, bytes] | None = None) -> bool:
    """Send the final markdown report (and optional images) to Telegram.

    Returns True on success and False on any validation/network/API error.
    """
    if not _env_truthy("NOTIFY_TELEGRAM", "false"):
        logger.info("Telegram notifier skipped: NOTIFY_TELEGRAM is not enabled.")
        return True

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        logger.info("Telegram notifier skipped: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not configured.")
        return False

    image_map = image_map or {}
    base_url = f"https://api.telegram.org/bot{bot_token}"

    telegram_html = _markdown_to_telegram_html(markdown_text)

    try:
        with httpx.Client(timeout=30.0) as client:
            for chunk in _split_text(telegram_html):
                message_response = client.post(
                    f"{base_url}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
                message_response.raise_for_status()

            for cid, image_bytes in image_map.items():
                photo_response = client.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": chat_id},
                    files={"photo": (f"{cid}.png", io.BytesIO(image_bytes), "image/png")},
                )
                photo_response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.exception("Failed to send Telegram report: %s", exc)
        return False
    except Exception as exc:
        logger.exception("Unexpected Telegram notifier failure: %s", exc)
        return False

    return True


def _load_recipient_emails(emails_file: str) -> list[str]:
    path = Path(emails_file)
    if not path.is_file():
        logger.warning("EMAILS_FILE does not exist or is not a file: %s", emails_file)
        return []

    recipients: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line and "@" in line:
            recipients.append(line)
    return recipients


def send_email_report(
    subject: str,
    markdown_text: str,
    image_map: dict[str, bytes] | None = None,
) -> bool:
    """Send the report by SMTP to addresses listed in EMAILS_FILE.

    Images referenced via ``![...](cid:<cid>)`` in *markdown_text* are embedded
    inline using MIME ``Content-ID`` headers so they render inside the HTML body
    rather than appearing as separate attachments.

    Uses NOTIFY_EMAIL, EMAILS_FILE, SMTP_* and SMTP_USE_TLS from the environment.
    Returns True when skipped (not an error), or after a successful send.
    Returns False on misconfiguration or SMTP failure.
    """
    if not _env_truthy("NOTIFY_EMAIL", "false"):
        logger.info("Email notifier skipped: NOTIFY_EMAIL is not enabled.")
        return True

    host = os.environ.get("SMTP_HOST", "").strip()
    from_email = os.environ.get("SMTP_FROM_EMAIL", "").strip()
    if not host or not from_email:
        logger.warning("Email notifier skipped: SMTP_HOST or SMTP_FROM_EMAIL is empty.")
        return False

    emails_path = os.environ.get("EMAILS_FILE", "emails.txt").strip() or "emails.txt"
    recipients = _load_recipient_emails(emails_path)
    if not recipients:
        logger.warning("Email notifier skipped: no recipients in %s", emails_path)
        return False

    port_raw = os.environ.get("SMTP_PORT", "587").strip() or "587"
    try:
        port = int(port_raw)
    except ValueError:
        logger.error("Invalid SMTP_PORT: %r", port_raw)
        return False

    username = os.environ.get("SMTP_USERNAME", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    use_tls = _env_truthy("SMTP_USE_TLS", "true")

    image_map = image_map or {}

    # related → alternative (plain + html) + inline CID images
    msg = MIMEMultipart("related")
    msg["Subject"] = str(Header(subject, "utf-8"))
    msg["From"] = from_email
    msg["To"] = ", ".join(recipients)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(markdown_text, "plain", "utf-8"))
    alt.attach(MIMEText(_markdown_to_email_html(markdown_text), "html", "utf-8"))
    msg.attach(alt)

    for cid, image_bytes in image_map.items():
        part = MIMEImage(image_bytes, _subtype="png")
        part.add_header("Content-ID", f"<{cid}>")
        part.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(part)

    try:
        if port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, timeout=60, context=context) as smtp:
                if username:
                    smtp.login(username, password)
                smtp.sendmail(from_email, recipients, msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=60) as smtp:
                smtp.ehlo()
                if use_tls:
                    context = ssl.create_default_context()
                    smtp.starttls(context=context)
                    smtp.ehlo()
                if username:
                    smtp.login(username, password)
                smtp.sendmail(from_email, recipients, msg.as_string())
    except (OSError, smtplib.SMTPException) as exc:
        logger.exception("Failed to send email report: %s", exc)
        return False

    logger.info("Email report sent successfully to %d recipient(s).", len(recipients))
    return True


def notify_report(
    subject: str,
    markdown_text: str,
    image_map: dict[str, bytes] | None = None,
) -> None:
    """Send via Telegram and/or email according to NOTIFY_* env flags."""
    send_telegram_report(markdown_text=markdown_text, image_map=image_map)
    send_email_report(subject=subject, markdown_text=markdown_text, image_map=image_map)
