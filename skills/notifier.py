from __future__ import annotations

import io
import logging
import os
import smtplib
import ssl
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import BinaryIO

import httpx

logger = logging.getLogger(__name__)


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


def send_telegram_report(markdown_text: str, image_buffers: list | None = None) -> bool:
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

    image_buffers = image_buffers or []
    base_url = f"https://api.telegram.org/bot{bot_token}"

    try:
        with httpx.Client(timeout=30.0) as client:
            message_response = client.post(
                f"{base_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": markdown_text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
            )
            message_response.raise_for_status()

            for index, image in enumerate(image_buffers, start=1):
                image_bytes = _to_bytes(image)
                if not image_bytes:
                    logger.warning("Skipping invalid image buffer at index %s", index - 1)
                    continue

                photo_response = client.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": chat_id},
                    files={"photo": (f"chart_{index}.png", io.BytesIO(image_bytes), "image/png")},
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
    image_buffers: list | None = None,
) -> bool:
    """Send the report by SMTP to addresses listed in EMAILS_FILE.

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

    image_buffers = image_buffers or []
    msg = MIMEMultipart("mixed")
    msg["Subject"] = str(Header(subject, "utf-8"))
    msg["From"] = from_email
    msg["To"] = ", ".join(recipients)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(markdown_text, "plain", "utf-8"))
    msg.attach(alt)

    for index, image in enumerate(image_buffers, start=1):
        image_bytes = _to_bytes(image)
        if not image_bytes:
            logger.warning("Skipping invalid image buffer at index %s", index - 1)
            continue
        part = MIMEImage(image_bytes, _subtype="png")
        part.add_header("Content-Disposition", "attachment", filename=f"chart_{index}.png")
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

    return True


def notify_report(
    subject: str,
    markdown_text: str,
    image_buffers: list | None = None,
) -> None:
    """Send via Telegram and/or email according to NOTIFY_* env flags."""
    send_telegram_report(markdown_text=markdown_text, image_buffers=image_buffers)
    send_email_report(subject=subject, markdown_text=markdown_text, image_buffers=image_buffers)
