from __future__ import annotations

import io
import logging
import os
from typing import BinaryIO

import httpx

logger = logging.getLogger(__name__)


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
